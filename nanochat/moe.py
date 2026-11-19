"""
Mixture of Experts in the style of Kimi K3 (config names as in moonshotai/Kimi-K3): MoE. Dropless, with fused Triton
kernels: the activations must be bfloat16 on CUDA.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch import Tensor
import triton
import triton.language as tl

ROWS = dict(BM=128, BN=128, BK=32, num_warps=4, num_stages=3)  # tile of _rows
WGRAD = dict(BM=128, BN=128, BK=32, num_warps=4, num_stages=3)  # tile of _wgrad
WGRAD_LONG = dict(BK=128, num_stages=2)  # _wgrad on GPUs with >= 160 KB shared memory per block (A100, H100): faster there
NONE, ACT, DZ = (tl.constexpr(i) for i in range(3))  # epilogues of _rows


@triton.jit
def _rows(A, W, OUT, P, HID, ROWSLOT, TILES, KD, ND, NP,
          K: tl.constexpr, GATHER: tl.constexpr, SCATTER: tl.constexpr, WT: tl.constexpr, EPI: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # OUT[rows] = epilogue(A[rows] @ W[e]) for one row tile of expert e. slot = ROWSLOT[row] = token * K + k; GATHER reads A
    # by token, SCATTER writes OUT by slot, WT uses W[e] transposed. ACT: a = p * relu(.)^2 (forward), DZ: g * 2 sqrt(p a)
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
            pv = tl.load(P + slot, mask=mm, other=1.0).to(tl.float32)
        if EPI == ACT:
            acc = tl.maximum(acc, 0.0)
            acc = acc * acc * pv[:, None]
        if EPI == DZ:
            hid = tl.load(HID + rows[:, None] * ND + rn[None, :], mask=mask, other=0.0).to(tl.float32)
            acc = acc * 2.0 * tl.sqrt(hid * pv[:, None])
        tl.store(OUT + orow[:, None] * ND + rn[None, :], acc.to(OUT.dtype.element_ty), mask=mask)


@triton.jit
def _wgrad(L, Rt, OUT, ROWSLOT, OFF, MD, ND, K: tl.constexpr, LGATHER: tl.constexpr, BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    # OUT[e] = sum over expert e's rows of L[row]^T Rt[row] (float32): a weight gradient, L or Rt gathered by token
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
    tl.store(OUT + (e.to(tl.int64) * MD + rm[:, None]) * ND + rn[None, :], acc, mask=mm[:, None] & mn[None, :])


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
    for k in tl.static_range(K):
        o = tl.load(O + (rt * K + k)[:, None] * D + rd[None, :], mask=mask, other=0.0).to(tl.float32)
        acc += o
        if DOT:
            tl.store(DOTS + (rt * K + k) * NP + pd, tl.sum(o * x, 1), mask=mt)
    tl.store(Y + rt[:, None] * D + rd[None, :], acc.to(Y.dtype.element_ty), mask=mask)


def make_plan(sel, E):
    # routing sel (T, K) -> rowslot (slot of each row, rows sorted by expert), off, tiles; all of routing-independent shape
    T, K = sel.shape
    flat = sel.reshape(-1)
    keys, rowslot = torch.sort(flat, stable=True)
    counts = torch.diff(torch.searchsorted(keys, torch.arange(E + 1, device=flat.device)))  # (bincount syncs the host)
    off, tiles = counts.new_empty(E + 1), counts.new_empty(triton.cdiv(T * K, ROWS["BM"]) + E, 3)
    _tiles[(len(tiles),)](counts, off, tiles, E, EP=triton.next_power_of_2(E), BM=ROWS["BM"])
    return rowslot, off, tiles


def _block(n, cfg, key):
    b = min(cfg[key], triton.next_power_of_2(n))
    return b, triton.cdiv(n, b)


def _rows_gemm(a, w, out, p, plan, epi=NONE, gather=False, scatter=False, wt=False, hid=None):
    rowslot, off, tiles = plan
    hid = out if hid is None else hid
    kd, nd = a.shape[1], out.shape[1]
    bn, n = _block(nd, ROWS, "BN")
    cfg = dict(ROWS, BN=bn, BK=min(ROWS["BK"], triton.next_power_of_2(kd)))
    _rows[(len(tiles) * n,)](a, w, out, p, hid, rowslot, tiles, kd, nd, n, K=p.shape[1],
                           GATHER=gather, SCATTER=scatter, WT=wt, EPI=epi, **cfg)


def _weight_grad(l, r, E, plan, lgather):
    rowslot, off, tiles = plan
    md, nd = l.shape[1], r.shape[1]
    out = torch.empty(E, md, nd, device=l.device, dtype=torch.float32)
    big_smem = torch.cuda.get_device_properties(l.device).shared_memory_per_block_optin >= 160 * 1024
    cfg = dict(WGRAD, **WGRAD_LONG) if big_smem else WGRAD
    bm, m = _block(md, cfg, "BM")
    bn, n = _block(nd, cfg, "BN")
    _wgrad[(m, n, E)](l, r, out, rowslot, off, md, nd, K=len(rowslot) // (l if lgather else r).shape[0], LGATHER=lgather, **dict(cfg, BM=bm, BN=bn))
    return out


def _sum_slots(o, T, K, x=None):
    D = o.shape[1]
    if K == 1 and x is None:
        return o
    n = triton.cdiv(D, 256)
    y = o.new_empty(T, D)
    dots = y if x is None else o.new_empty(T * K, n, dtype=torch.float32)
    _combine[(triton.cdiv(T, 32), n)](o, y, y if x is None else x, dots, T, D, n, K=K, DOT=x is not None, BT=32, BD=256, num_warps=4)
    return y if x is None else (y, dots.sum(1).view(T, K))


# forward and backward as custom ops, so torch.compile compiles the model around them; the forward also returns what the
# backward reuses (gated activations a, bf16 weights, plan)
@torch.library.custom_op("moe::routed_experts", mutates_args=())
def _experts(x: Tensor, wu: Tensor, wd: Tensor, sel: Tensor, p: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    (T, K), (E, H, D) = sel.shape, wu.shape
    x, p = x.contiguous(), p.contiguous()
    wu = wu.to(x.dtype, copy=True, memory_format=torch.contiguous_format)  # (copy: outputs may not alias inputs)
    wd = wd.to(x.dtype, copy=True, memory_format=torch.contiguous_format)
    plan = make_plan(sel, E)
    a = x.new_empty(T * K, H)
    _rows_gemm(x, wu, a, p, plan, ACT, gather=True, wt=True)  # a = p * relu(x @ wu^T)^2
    o = x.new_empty(T * K, D)
    _rows_gemm(a, wd, o, p, plan, scatter=True, wt=True)  # o = a @ wd^T
    return _sum_slots(o, T, K), a, wu, wd, *plan


@torch.library.custom_op("moe::routed_experts_backward", mutates_args=())
def _experts_backward(dy: Tensor, x: Tensor, p: Tensor, a: Tensor, wu: Tensor, wd: Tensor,
                      rowslot: Tensor, off: Tensor, tiles: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    (T, K), (E, H, D) = p.shape, wu.shape
    x, p, dy = x.contiguous(), p.contiguous(), dy.contiguous().to(x.dtype)
    plan = rowslot, off, tiles
    dz = torch.empty_like(a)
    _rows_gemm(dy, wd, dz, p, plan, DZ, gather=True, hid=a)  # dz = (dy @ wd) * da/dz
    dxs = x.new_empty(T * K, D)
    _rows_gemm(dz, wu, dxs, p, plan, scatter=True)  # dx per slot = dz @ wu
    dwu = _weight_grad(dz, x, E, plan, lgather=False)  # dz^T x
    dwd = _weight_grad(dy, a, E, plan, lgather=True)  # dy^T a
    # a is of degree 2 in z, so dL/dp = <da, a> / p = <dz @ wu, x> / (2 p): the slot's dx dotted with x
    dx, dots = _sum_slots(dxs, T, K, x)
    return dx, dwu, dwd, dots / (2 * p)


@_experts.register_fake
def _(x, wu, wd, sel, p):
    R, E = sel.numel(), wu.shape[0]
    return (x.new_empty(x.shape[0], wd.shape[1]), x.new_empty(R, wu.shape[1]), x.new_empty(wu.shape), x.new_empty(wd.shape),
            sel.new_empty(R), sel.new_empty(E + 1), sel.new_empty(triton.cdiv(R, ROWS["BM"]) + E, 3))


@_experts_backward.register_fake
def _(dy, x, p, a, wu, wd, rowslot, off, tiles):
    f32 = dict(dtype=torch.float32)
    return torch.empty_like(x), wu.new_empty(wu.shape, **f32), wd.new_empty(wd.shape, **f32), p.new_empty(p.shape, **f32)


def _backward(ctx, dy, *unused):
    dx, dwu, dwd, dp = _experts_backward(dy, *ctx.saved_tensors)
    return dx, dwu, dwd, None, dp


_experts.register_autograd(_backward, setup_context=lambda ctx, inputs, output: ctx.save_for_backward(inputs[0], inputs[4], *output[1:]))


def routed_experts(x, wu, wd, sel, p):
    # y = sum_k p[:, k] * expert_{sel[:, k]}(x); x (T, D) bf16, wu (E, H, D), wd (E, D, H), sel (T, K), p (T, K) float32
    return _experts(x, wu, wd, sel, p)[0]


# Quantile Balancing (Kimi K3 tech report, 2.3.3): after each batch, per expert, the bias that would have given it exactly
# k/E of the tokens with every token's cutoff held fixed, read off a histogram of the routing margins (score + bias - the
# token's cutoff). Histograms add, so DDP all-reduces them. A token's cutoff may lie anywhere between its k-th and (k+1)-th
# largest biased scores; the report takes the (k+1)-th, MoE.forward their midpoint (see there).
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


class MoE(nn.Module):
    """In place of a layer's MLP: y = shared(x) + sum over the token's k experts of p_k expert_k(x).

    s = sigmoid(x @ router^T)                    router scores, one per expert
    the token's experts: top-k of s + bias       the bias only steers the selection; Quantile Balancing sets it in training
    p = s[experts] / sum(s[experts])             mixture weights
    expert(x) = relu(x @ wu^T)^2 @ wd^T          relu^2 as in the dense MLP; the shared experts are one such MLP

    config.num_experts              routed experts
    config.num_experts_per_token    k
    config.moe_intermediate_size    hidden width of one expert
    config.num_shared_experts       shared experts, on every token, each moe_intermediate_size wide (0: none)
    config.n_embd                   model dimension

    The input is bfloat16 on CUDA (Triton kernels). Every routed (token, expert) pair is computed: nothing is dropped.
    """

    def __init__(self, config):
        super().__init__()
        for name in ("num_experts_per_token", "moe_intermediate_size", "num_shared_experts"):
            assert getattr(config, name) is not None, f"MoE: set {name} explicitly (no defaults, no derived widths)"
        E, D, H = config.num_experts, config.n_embd, config.moe_intermediate_size
        self.k = config.num_experts_per_token
        self.router = nn.Parameter(torch.empty(E, D))
        self.wu = nn.Parameter(torch.empty(E, H, D))
        self.wd = nn.Parameter(torch.empty(E, D, H))
        self.shared_up = nn.Parameter(torch.empty(config.num_shared_experts * H, D))  # the shared experts as one MLP (as in Kimi K3)
        self.shared_down = nn.Parameter(torch.empty(D, config.num_shared_experts * H))
        self.register_buffer("bias", torch.zeros(E))  # selection bias, set by Quantile Balancing
        self.register_buffer("load", torch.zeros(E), persistent=False)  # selections since the last GPT.pop_moe_stats()

    def forward(self, x):
        assert x.is_cuda and x.dtype == torch.bfloat16, "MoE runs Triton kernels: its input must be bfloat16 on CUDA"
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        s = torch.sigmoid(F.linear(x, self.router.to(x.dtype)).float())
        top = (s + self.bias).topk(self.k + 1, dim=1)
        sel = top.indices[:, :self.k]
        if self.training:  # balance for the next batch on this one's margins
            self.load.scatter_add_(0, sel.reshape(-1), torch.ones(sel.numel(), device=sel.device))
            # The cutoff: midway between the token's k-th and (k+1)-th largest biased scores. With few experts per token
            # the two are far apart, and at the (k+1)-th an expert the token did not select looks closer to getting in
            # than it is: loads then stay 10-30 % off an even split, against a few % at the midpoint.
            cutoff = top.values[:, self.k - 1:].mean(1)
            self.bias.copy_(quantile_balance(margin_counts(s.detach(), cutoff.detach(), self.bias), self.bias, self.k))
        g = s.gather(1, sel)
        p = g / g.sum(1, keepdim=True)
        y = routed_experts(x, self.wu, self.wd, sel, p)
        if self.shared_up.numel():
            y = y + F.linear(F.relu(F.linear(x, self.shared_up.to(x.dtype))).square(), self.shared_down.to(x.dtype))
        return y.view(shape)
