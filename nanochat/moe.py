"""Mixture of Experts in the style of Kimi K3 (github.com/MoonshotAI/Kimi-K3) as an MLP sublayer: SharedExperts (a
standard MLP), RoutedExperts (Triton kernels: bfloat16 on CUDA) and MoE, their sum."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
from torch.multiprocessing.reductions import StorageWeakRef
from torch.optim.optimizer import register_optimizer_step_pre_hook, register_optimizer_step_post_hook
import triton
import triton.language as tl

ROWS = dict(BM=128, BN=128, BK=32, num_warps=4, num_stages=3)  # tile of _rows
WGRAD = dict(BM=128, BN=128, BK=64, num_warps=4, num_stages=2)  # tile of _wgrad
WGRAD_LONG = dict(BK=128, num_stages=2)  # _wgrad on GPUs with >= 160 KB shared memory per block (A100, H100): faster there
OVERLAP = True  # the backward runs its two independent GEMM pairs on two CUDA streams
NONE, ACT, DZ = (tl.constexpr(i) for i in range(3))  # epilogues of _rows


@triton.jit
def _rows(A, W, OUT, P, HID, DOTS, ROWSLOT, TILES, KD, ND, NP,
          K: tl.constexpr, GATHER: tl.constexpr, SCATTER: tl.constexpr, WT: tl.constexpr, EPI: tl.constexpr, SILU: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # OUT[rows] = epilogue(A[rows] @ W[e]) for one row tile of expert e. slot = ROWSLOT[row] = token * K + k; GATHER reads A
    # by token, SCATTER writes OUT by slot, WT uses W[e] transposed. The epilogues, with a = p act(z):
    #   ACT (forward, the product is z): a; with SILU also z into HID, which silu's derivative needs
    #   DZ (backward, the product is da): da * da/dz. relu^2: 2 sqrt(p a) from HID = a. SILU: from HID = z, and this program's
    #   share of <da, silu(z)> = dL/dp into DOTS[slot]
    t, pn = tl.program_id(0) // NP, tl.program_id(0) % NP  # a tile's NP programs run together: its rows of A stay cached
    cnt = tl.load(TILES + 3 * t + 2)
    if cnt > 0:
        e = tl.load(TILES + 3 * t)
        om = tl.arange(0, BM)
        mm = om < cnt
        rows = tl.load(TILES + 3 * t + 1) + om
        slot = tl.load(ROWSLOT + rows, mask=mm, other=0).to(tl.int64)
        arow = slot // K if GATHER else rows
        orow = slot if SCATTER else rows
        rn = pn * BN + tl.arange(0, BN)
        mn = rn < ND
        w = W + e * KD * ND
        acc = tl.zeros((BM, BN), dtype=tl.float32)
        for k in range(0, KD, BK):
            rk = k + tl.arange(0, BK)
            mk = rk < KD
            a = tl.load(A + arow[:, None] * KD + rk[None, :], mask=mm[:, None] & mk[None, :], other=0.0)
            if WT:
                b = tl.trans(tl.load(w + rn[:, None] * KD + rk[None, :], mask=mn[:, None] & mk[None, :], other=0.0))
            else:
                b = tl.load(w + rk[:, None] * ND + rn[None, :], mask=mk[:, None] & mn[None, :], other=0.0)
            acc = tl.dot(a, b.to(a.dtype), acc)
        mask = mm[:, None] & mn[None, :]
        if EPI != NONE:
            pv = tl.load(P + slot, mask=mm, other=1.0).to(tl.float32)[:, None]
        if EPI == ACT:
            if SILU:
                tl.store(HID + rows[:, None] * ND + rn[None, :], acc.to(HID.dtype.element_ty), mask=mask)
                acc = acc * tl.sigmoid(acc) * pv
            else:
                acc = tl.maximum(acc, 0.0)
                acc = acc * acc * pv
        if EPI == DZ:
            hid = tl.load(HID + rows[:, None] * ND + rn[None, :], mask=mask, other=0.0).to(tl.float32)
            if SILU:
                s = tl.sigmoid(hid)
                tl.store(DOTS + slot * NP + pn, tl.sum(acc * hid * s, 1), mask=mm)
                acc = acc * pv * s * (1.0 + hid * (1.0 - s))
            else:
                acc = acc * 2.0 * tl.sqrt(hid * pv)
        tl.store(OUT + orow[:, None] * ND + rn[None, :], acc.to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _wgrad(L, Rt, OUT, ROWSLOT, OFF, MD, ND, K: tl.constexpr, LGATHER: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # OUT[e] (MD, ND) = sum over expert e's rows of L[row]^T Rt[row], summed in float32: a weight gradient, L or Rt gathered by token
    pm, pn, e = tl.program_id(0), tl.program_id(1), tl.program_id(2)  # an expert's programs run together: its rows stay cached
    r1 = tl.load(OFF + e + 1).to(tl.int32)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mm, mn = rm < MD, rn < ND
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for r in range(tl.load(OFF + e).to(tl.int32), r1, BK):
        rows = (r + tl.arange(0, BK)).to(tl.int64)  # (32-bit loop counter: faster)
        mr = rows < r1
        tok = tl.load(ROWSLOT + rows, mask=mr, other=0) // K
        lrow = tok if LGATHER else rows
        rrow = rows if LGATHER else tok
        a = tl.load(L + lrow[None, :] * MD + rm[:, None], mask=mm[:, None] & mr[None, :], other=0.0)
        b = tl.load(Rt + rrow[:, None] * ND + rn[None, :], mask=mr[:, None] & mn[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(OUT + (e.to(tl.int64) * MD + rm[:, None]) * ND + rn[None, :], acc.to(OUT.dtype.element_ty), mask=mm[:, None] & mn[None, :])


@triton.jit
def _tiles(COUNTS, OFF, TILES, E, EP: tl.constexpr, BM: tl.constexpr):
    # OFF[e] = first row of expert e; TILES[t] = (expert, first row, row count) of row tile t, count 0 past the last tile
    t = tl.program_id(0)
    ie = tl.arange(0, EP)
    c = tl.load(COUNTS + ie, mask=ie < E, other=0)
    if t <= E:
        tl.store(OFF + t, tl.sum(tl.where(ie < t, c, 0)))
    cum = tl.cumsum((c + BM - 1) // BM, 0)
    done = (cum <= t) & (ie < E)  # experts whose tiles all come before t
    e = tl.sum(done.to(tl.int64))
    used = e < E
    e = tl.minimum(e, E - 1)
    j = t - tl.max(tl.where(done, cum, 0))
    tl.store(TILES + 3 * t, e)
    tl.store(TILES + 3 * t + 1, tl.sum(tl.where(done, c, 0)) + j * BM)
    tl.store(TILES + 3 * t + 2, tl.where(used, tl.minimum(tl.load(COUNTS + e) - j * BM, BM), 0))


@triton.jit
def _combine(O, Y, X, DOTS, T, D, NP, K: tl.constexpr, DOT: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    # Y[t] = sum_k O[t * K + k]; with DOT also each program's share of <O[slot], X[t]> in DOTS[slot, program]
    pd = tl.program_id(1)
    rt = (tl.program_id(0) * BT + tl.arange(0, BT)).to(tl.int64)
    rd = pd * BD + tl.arange(0, BD)
    mt = rt < T
    mask = mt[:, None] & (rd < D)[None, :]
    acc = tl.zeros((BT, BD), dtype=tl.float32)
    if DOT:
        x = tl.load(X + rt[:, None] * D + rd[None, :], mask=mask, other=0.0).to(tl.float32)
    for k in range(K):  # (a real loop: unrolled, 12 or more slots spill the registers and the sum runs 8x slower)
        o = tl.load(O + (rt * K + k)[:, None] * D + rd[None, :], mask=mask, other=0.0).to(tl.float32)
        acc += o
        if DOT:
            tl.store(DOTS + (rt * K + k) * NP + pd, tl.sum(o * x, 1), mask=mt)
    tl.store(Y + rt[:, None] * D + rd[None, :], acc.to(Y.dtype.element_ty), mask=mask)


def make_plan(sel, E):
    # routing sel (T, K) -> rowslot (slot of each row, rows sorted by expert), off, tiles; all of routing-independent shape
    T, K = sel.shape
    assert E < 2 ** 15, "MoE: at most 32767 experts (the plan sorts 16-bit keys)"
    keys, rowslot = torch.sort(sel.reshape(-1).to(torch.int16), stable=True)  # (16-bit keys: the sort takes half the time)
    counts = torch.diff(torch.searchsorted(keys, torch.arange(E + 1, device=sel.device, dtype=torch.int16)))  # (bincount syncs the host)
    off, tiles = counts.new_empty(E + 1), counts.new_empty(triton.cdiv(T * K, ROWS["BM"]) + E, 3)
    _tiles[(len(tiles),)](counts, off, tiles, E, EP=triton.next_power_of_2(E), BM=ROWS["BM"])
    return rowslot, off, tiles


def _block(n, cfg, key):
    b = min(cfg[key], triton.next_power_of_2(n))
    return b, triton.cdiv(n, b)


def _rows_gemm(a, w, out, p, plan, epi=NONE, gather=False, scatter=False, wt=False, silu=False, hid=None, dots=None):
    rowslot, off, tiles = plan
    hid = out if hid is None else hid  # (what an epilogue does not touch: any pointer)
    dots = out if dots is None else dots
    kd, nd = a.shape[1], out.shape[1]
    bn, n = _block(nd, ROWS, "BN")
    cfg = dict(ROWS, BN=bn, BK=min(ROWS["BK"], triton.next_power_of_2(kd)))
    _rows[(len(tiles) * n,)](a, w, out, p, hid, dots, rowslot, tiles, kd, nd, n, K=p.shape[1],
                           GATHER=gather, SCATTER=scatter, WT=wt, EPI=epi, SILU=silu, **cfg)


def _weight_grad(l, r, out, plan, lgather):
    rowslot, off, tiles = plan
    E, md, nd = out.shape
    big_smem = torch.cuda.get_device_properties(l.device).shared_memory_per_block_optin >= 160 * 1024
    cfg = dict(WGRAD, **WGRAD_LONG) if big_smem else WGRAD
    bm, m = _block(md, cfg, "BM")
    bn, n = _block(nd, cfg, "BN")
    _wgrad[(m, n, E)](l, r, out, rowslot, off, md, nd, K=len(rowslot) // (l if lgather else r).shape[0], LGATHER=lgather, **dict(cfg, BM=bm, BN=bn))


def _sum_slots(o, T, K, x=None):
    D = o.shape[1]
    if K == 1 and x is None:
        return o
    n = triton.cdiv(D, 256)
    y = o.new_empty(T, D)
    dots = y if x is None else o.new_empty(T * K, n, dtype=torch.float32)
    _combine[(triton.cdiv(T, 32), n)](o, y, y if x is None else x, dots, T, D, n, K=K, DOT=x is not None, BT=32, BD=256, num_warps=4)
    return y if x is None else (y, dots.sum(1).view(T, K))


_COPIES = {}  # (where wu and wd are stored) -> (their versions and shape then, weak references to their storages, their copies)


def _kernel_weights(wu, wd, dtype):
    # The weights in the activations' dtype, as the kernels read them. Weights of another dtype are copied, and the copies
    # kept until the weights change: one cast per optimizer step, not one per micro-batch. A change shows in the weights'
    # version counters. Writes through .data do not move those, so every optimizer step also drops the copies (below:
    # before the step, which frees their memory for it, and after it).
    cast = lambda w: w.to(dtype).contiguous()
    if wu.dtype == dtype or torch.cuda.is_current_stream_capturing():  # (a CUDA graph replays the kernels, not this function: the cast has to be in it)
        return cast(wu), cast(wd)
    key, state = (wu.data_ptr(), wd.data_ptr()), (wu._version, wd._version, wu.shape, wu.stride(), dtype)
    hit = _COPIES.get(key)
    if hit is None or hit[0] != state or hit[1].expired() or hit[2].expired():
        for gone in [k for k, v in _COPIES.items() if v[1].expired() or v[2].expired()]:
            del _COPIES[gone]
        hit = _COPIES[key] = (state, StorageWeakRef(wu.untyped_storage()), StorageWeakRef(wd.untyped_storage()), cast(wu), cast(wd))
    return hit[3], hit[4]


register_optimizer_step_pre_hook(lambda *_: _COPIES.clear())
register_optimizer_step_post_hook(lambda *_: _COPIES.clear())


# forward and backward as custom ops, so torch.compile compiles the model around them; the forward also returns what the
# backward reuses (gated activations a, for silu the pre-activations z, the plan).
@torch.library.custom_op("moe::routed_experts", mutates_args=())
def _experts(x: Tensor, wu: Tensor, wd: Tensor, sel: Tensor, p: Tensor, silu: bool) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    (T, K), (E, H, D) = sel.shape, wu.shape
    x, p = x.contiguous(), p.contiguous()
    wub, wdb = _kernel_weights(wu, wd, x.dtype)
    plan = make_plan(sel, E)
    a, z = x.new_empty(T * K, H), x.new_empty(T * K if silu else 0, H)
    _rows_gemm(x, wub, a, p, plan, ACT, gather=True, wt=True, silu=silu, hid=z if silu else None)  # z = x @ wu^T, a = p * act(z)
    o = x.new_empty(T * K, D)
    _rows_gemm(a, wdb, o, p, plan, scatter=True, wt=True)  # o = a @ wd^T
    return _sum_slots(o, T, K), a, z, *plan


_SIDE = {}  # device -> its second stream


@torch.library.custom_op("moe::routed_experts_backward", mutates_args=())
def _experts_backward(dy: Tensor, x: Tensor, p: Tensor, a: Tensor, z: Tensor, wu: Tensor, wd: Tensor,
                      rowslot: Tensor, off: Tensor, tiles: Tensor, silu: bool) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    (T, K), D = p.shape, x.shape[1]
    x, p, dy = x.contiguous(), p.contiguous(), dy.contiguous().to(x.dtype)
    wub, wdb = _kernel_weights(wu, wd, x.dtype)
    plan = rowslot, off, tiles
    dz, dxs = torch.empty_like(a), x.new_empty(T * K, D)
    dwu, dwd = wu.new_empty(wu.shape), wd.new_empty(wd.shape)
    dots = p.new_empty(T * K, _block(a.shape[1], ROWS, "BN")[1]) if silu else None  # per slot, each column program's share of dL/dp
    # Two pairs that do not depend on each other, one per stream: with few rows a launch's last wave of tiles is part
    # empty, and the other stream's kernel fills it. Side stream: dwd, then dwu once dz is there.
    main = torch.cuda.current_stream()
    side = _SIDE.setdefault(x.device, torch.cuda.Stream(x.device)) if OVERLAP else main
    side.wait_stream(main)
    with torch.cuda.stream(side):
        _weight_grad(dy, a, dwd, plan, lgather=True)  # dy^T a
    _rows_gemm(dy, wdb, dz, p, plan, DZ, gather=True, silu=silu, hid=z if silu else a, dots=dots)  # da = dy @ wd, dz = da * da/dz
    side.wait_stream(main)
    with torch.cuda.stream(side):
        _weight_grad(dz, x, dwu, plan, lgather=False)  # dz^T x
    _rows_gemm(dz, wub, dxs, p, plan, scatter=True)  # dx per slot = dz @ wu
    main.wait_stream(side)  # (before the caller's stream can free or reuse what the side stream reads)
    if silu:  # dL/dp = <da, silu(z)>: the dots of dz's epilogue
        return _sum_slots(dxs, T, K), dwu, dwd, dots.sum(1).view(T, K)
    # relu^2: a is of degree 2 in z, so dL/dp = <da, a> / p = <dz @ wu, x> / (2 p): the slot's dx dotted with x, in the slot sum
    dx, dots = _sum_slots(dxs, T, K, x)
    return dx, dwu, dwd, dots / (2 * p)


@_experts.register_fake
def _(x, wu, wd, sel, p, silu):
    R, (E, H, D) = sel.numel(), wu.shape
    return (x.new_empty(x.shape), x.new_empty(R, H), x.new_empty(R if silu else 0, H),
            sel.new_empty(R), sel.new_empty(E + 1), sel.new_empty(triton.cdiv(R, ROWS["BM"]) + E, 3))


@_experts_backward.register_fake
def _(dy, x, p, a, z, wu, wd, rowslot, off, tiles, silu):
    return x.new_empty(x.shape), wu.new_empty(wu.shape), wd.new_empty(wd.shape), p.new_empty(p.shape)


def _setup(ctx, inputs, output):
    x, wu, wd, sel, p, silu = inputs
    ctx.save_for_backward(x, p, *output[1:3], wu, wd, *output[3:])
    ctx.silu = silu


def _backward(ctx, dy, *unused):
    dx, dwu, dwd, dp = _experts_backward(dy, *ctx.saved_tensors, ctx.silu)
    return dx, dwu, dwd, None, dp, None


_experts.register_autograd(_backward, setup_context=_setup)


def relu2(h):
    return F.relu(h).square()


ACTIVATIONS = {"relu2": relu2, "silu": F.silu}  # of an expert's hidden units: relu squared (nanochat's MLP) or silu


def routed_experts(x, wu, wd, sel, p, activation):
    # y = sum_k p[:, k] * expert_{sel[:, k]}(x), expert_e(x) = act(x @ wu[e]^T) @ wd[e]^T, act one of ACTIVATIONS;
    # x (T, D) bf16, wu (E, H, D), wd (E, D, H), sel (T, K), p (T, K) float32. Any D and H, but the kernels load a row as
    # vectors only if its length is a multiple of 16: at other widths the GEMMs run at half speed.
    assert activation in ACTIVATIONS, f"activation must be one of {list(ACTIVATIONS)}, not {activation!r}"
    return _experts(x, wu, wd, sel, p, activation == "silu")[0]


# Quantile Balancing (Kimi K3 tech report, 2.3.3): after each batch, per expert, the bias that would have given it exactly
# k/E of the tokens with every token's cutoff held fixed, read off a histogram of the routing margins (score + bias - the
# token's cutoff). Histograms add, so DDP all-reduces them. A token's cutoff may lie anywhere between its k-th and (k+1)-th
# largest biased scores; the report takes the (k+1)-th, RoutedExperts.forward their midpoint (see there).
BINS, SPAN = 512, 0.5


def margin_counts(s, cutoff, bias):
    T, E = s.shape
    idx = (((s + bias) - cutoff[:, None] + SPAN) * (BINS / (2 * SPAN))).long().clamp_(0, BINS - 1)
    return torch.zeros(E, BINS, device=s.device).scatter_add_(1, idx.t(), torch.ones(E, T, device=s.device))


@torch.library.custom_op("moe::quantile_balance", mutates_args=())
def quantile_balance(counts: Tensor, bias: Tensor, k: int) -> Tensor:
    E = counts.shape[0]
    if dist.is_available() and dist.is_initialized():
        counts = counts.clone()
        dist.all_reduce(counts)
    above = counts.flip(1).cumsum(1)  # margins in each expert's top i + 1 bins
    target = counts.sum(1, keepdim=True) * k / E
    i = (above < target).sum(1, keepdim=True).clamp_(max=BINS - 1)  # the bin where the target is crossed
    below = above.gather(1, (i - 1).clamp_(min=0)) * (i > 0)
    frac = ((target - below) / (above.gather(1, i) - below).clamp_(min=1)).clamp_(0, 1)
    bias = bias - (SPAN - (i + frac).squeeze(1) * (2 * SPAN / BINS))
    return bias - bias.mean()


@quantile_balance.register_fake
def _(counts, bias, k):
    return torch.empty_like(bias)


class SharedExperts(nn.Module):
    """Experts that every token goes through: y = sum over the experts of act(x @ up^T) @ down^T. Any device and dtype.
    One expert 4 * model_dim wide is the standard MLP.

    model_dim                  width of x and y
    num_shared_experts         experts
    shared_intermediate_size   hidden width of one expert
    activation                 "relu2" (relu squared, nanochat's) or "silu"
    """

    def __init__(self, model_dim: int, num_shared_experts: int, shared_intermediate_size: int, activation: str):
        super().__init__()
        assert num_shared_experts > 0 and shared_intermediate_size > 0, "SharedExperts: num_shared_experts > 0 and shared_intermediate_size > 0"
        assert activation in ACTIVATIONS, f"activation must be one of {list(ACTIVATIONS)}, not {activation!r}"
        self.model_dim, self.act = model_dim, ACTIVATIONS[activation]
        # the experts' hidden units side by side: one MLP num_shared_experts * shared_intermediate_size wide
        self.up = nn.Parameter(torch.empty(num_shared_experts * shared_intermediate_size, model_dim))
        self.down = nn.Parameter(torch.empty(model_dim, num_shared_experts * shared_intermediate_size))

    def init_weights(self):
        """nanochat's MLP init: up uniform with std 0.4 / sqrt(model_dim), down zero."""
        s = 3 ** 0.5 * self.model_dim ** -0.5  # (uniform on (-s, s) has std 1 / sqrt(model_dim))
        torch.nn.init.uniform_(self.up, -0.4 * s, 0.4 * s)
        torch.nn.init.zeros_(self.down)

    def num_active_params(self):
        """The parameters a token goes through: all of them."""
        return self.up.numel() + self.down.numel()

    def forward(self, x):
        return F.linear(self.act(F.linear(x, self.up.to(x.dtype))), self.down.to(x.dtype))


class RoutedExperts(nn.Module):
    """Experts of which a token goes through its own k: y = sum over them of p * act(x @ up^T) @ down^T.

    s = sigmoid(x @ router^T)                    router scores, one per expert
    the token's experts: top-k of s + bias       the bias only steers the selection; Quantile Balancing sets it in training
    p = s[experts] / sum(s[experts])             mixture weights

    model_dim                width of x and y
    num_experts              experts
    num_experts_per_token    k, the experts of a token
    moe_intermediate_size    hidden width of one expert
    activation               "relu2" (relu squared, nanochat's) or "silu"

    Triton kernels: the input is bfloat16 on CUDA, model_dim and moe_intermediate_size are multiples of 16, and every
    (token, expert) pair is computed: nothing is dropped. In training, each forward also sets the bias for the next batch
    from this one's routing, and the kernels read their own bfloat16 copies of the weights, made again when the weights
    change: after an optimizer step or an in-place write, not after a write through .data.
    """

    def __init__(self, model_dim: int, num_experts: int, num_experts_per_token: int, moe_intermediate_size: int, activation: str):
        super().__init__()
        assert 0 < num_experts_per_token < num_experts, "RoutedExperts: 0 < num_experts_per_token < num_experts"
        assert moe_intermediate_size > 0 and model_dim % 16 == 0 and moe_intermediate_size % 16 == 0, \
            "RoutedExperts: model_dim and moe_intermediate_size must be multiples of 16 (at other widths the kernels' GEMMs run at half speed)"
        assert activation in ACTIVATIONS, f"activation must be one of {list(ACTIVATIONS)}, not {activation!r}"
        self.model_dim, self.num_experts, self.num_experts_per_token, self.activation = model_dim, num_experts, num_experts_per_token, activation
        self.router = nn.Parameter(torch.empty(num_experts, model_dim))
        self.up = nn.Parameter(torch.empty(num_experts, moe_intermediate_size, model_dim))
        self.down = nn.Parameter(torch.empty(num_experts, model_dim, moe_intermediate_size))
        self.register_buffer("bias", torch.zeros(num_experts))  # selection bias, set by Quantile Balancing
        self.register_buffer("load", torch.zeros(num_experts), persistent=False)  # selections per expert since they were last read and zeroed

    def init_weights(self):
        """The experts as nanochat's MLP (up uniform with std 0.4 / sqrt(model_dim), down zero); the router uniform with
        std 1 / sqrt(model_dim); the balancing bias and the load counts zero."""
        s = 3 ** 0.5 * self.model_dim ** -0.5  # (uniform on (-s, s) has std 1 / sqrt(model_dim))
        torch.nn.init.uniform_(self.router, -s, s)
        torch.nn.init.uniform_(self.up, -0.4 * s, 0.4 * s)
        torch.nn.init.zeros_(self.down)
        self.bias.zero_()
        self.load.zero_()

    def num_active_params(self):
        """The parameters a token goes through: k of the experts (the router aside)."""
        return (self.up.numel() + self.down.numel()) * self.num_experts_per_token // self.num_experts

    def forward(self, x):
        assert x.is_cuda and x.dtype == torch.bfloat16, "RoutedExperts runs on Triton kernels: its input must be bfloat16 on CUDA"
        shape, k = x.shape, self.num_experts_per_token
        x = x.reshape(-1, shape[-1])
        s = torch.sigmoid(F.linear(x, self.router.to(x.dtype)).float())
        top = (s + self.bias).topk(k + 1, dim=1)
        sel = top.indices[:, :k]
        if self.training:  # balance for the next batch on this one's margins
            self.load.scatter_add_(0, sel.reshape(-1), torch.ones(sel.numel(), device=sel.device))
            # The cutoff: midway between the token's k-th and (k+1)-th largest biased scores. With few experts per token
            # the two are far apart, and at the (k+1)-th an expert the token did not select looks closer to getting in
            # than it is: loads then stay 10-30 % off an even split, against a few % at the midpoint.
            cutoff = top.values[:, k - 1:].mean(1)
            self.bias.copy_(quantile_balance(margin_counts(s.detach(), cutoff.detach(), self.bias), self.bias, k))
        g = s.gather(1, sel)
        # In training the op takes the weights as they are and keeps its own copies in x's dtype from one optimizer step to
        # the next (_kernel_weights). In eval they are cast here on every call, as a Linear's are.
        up, down = (self.up, self.down) if self.training else (self.up.to(x.dtype), self.down.to(x.dtype))
        return routed_experts(x, up, down, sel, g / g.sum(1, keepdim=True), self.activation).view(shape)


class MoE(nn.Module):
    """An MLP sublayer of shared and routed experts: y = shared(x) + routed(x). The arguments are SharedExperts' and
    RoutedExperts', with one activation for both."""

    def __init__(self, model_dim: int, num_experts: int, num_experts_per_token: int, moe_intermediate_size: int,
                 num_shared_experts: int, shared_intermediate_size: int, activation: str):
        super().__init__()
        self.shared = SharedExperts(model_dim, num_shared_experts, shared_intermediate_size, activation)
        self.routed = RoutedExperts(model_dim, num_experts, num_experts_per_token, moe_intermediate_size, activation)

    def init_weights(self):
        self.shared.init_weights()
        self.routed.init_weights()

    def num_active_params(self):
        """The parameters a token goes through: the shared experts and k of the routed ones (the router aside)."""
        return self.shared.num_active_params() + self.routed.num_active_params()

    def forward(self, x):
        return self.routed(x) + self.shared(x)
