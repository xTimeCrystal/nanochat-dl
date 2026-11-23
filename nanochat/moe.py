"""Routed experts of a Mixture of Experts in the style of Kimi K3 (github.com/MoonshotAI/Kimi-K3), for the MLP of gpt.py:
routed(mlp, x). Triton kernels (bfloat16 on CUDA) and Kimi K3's Quantile Balancing."""

import torch
import torch.distributed as dist
from torch import Tensor
import triton
import triton.language as tl

ROWS = dict(BM=128, BN=64, BK=32, num_warps=4, num_stages=3)  # tile of _rows
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
    # W is read as it is stored and cast to A's dtype a tile at a time: float32 weights need no copy.
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
    n = triton.cdiv(D, 256)
    y = o.new_empty(T, D)
    dots = y if x is None else o.new_empty(T * K, n, dtype=torch.float32)
    _combine[(triton.cdiv(T, 32), n)](o, y, y if x is None else x, dots, T, D, n, K=K, DOT=x is not None, BT=32, BD=256, num_warps=4)
    return y if x is None else (y, dots.sum(1).view(T, K))


# forward and backward as custom ops, so torch.compile compiles the model around them; the forward also returns what the
# backward reuses (gated activations a, for silu the pre-activations z, the plan).
@torch.library.custom_op("moe::routed_experts", mutates_args=())
def _experts(x: Tensor, fc: Tensor, proj: Tensor, sel: Tensor, p: Tensor, silu: bool) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    (T, K), (E, H, D) = sel.shape, fc.shape
    x, p, fc, proj = x.contiguous(), p.contiguous(), fc.contiguous(), proj.contiguous()
    plan = make_plan(sel, E)
    a, z = x.new_empty(T * K, H), x.new_empty(T * K if silu else 0, H)
    _rows_gemm(x, fc, a, p, plan, ACT, gather=True, wt=True, silu=silu, hid=z if silu else None)  # z = x @ fc^T, a = p * act(z)
    o = x.new_empty(T * K, D)
    _rows_gemm(a, proj, o, p, plan, scatter=True, wt=True)  # o = a @ proj^T
    return _sum_slots(o, T, K), a, z, *plan


_SIDE = {}  # device -> its second stream


@torch.library.custom_op("moe::routed_experts_backward", mutates_args=())
def _experts_backward(dy: Tensor, x: Tensor, p: Tensor, a: Tensor, z: Tensor, fc: Tensor, proj: Tensor,
                      rowslot: Tensor, off: Tensor, tiles: Tensor, silu: bool) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    (T, K), D = p.shape, x.shape[1]
    x, p, dy, fc, proj = x.contiguous(), p.contiguous(), dy.contiguous().to(x.dtype), fc.contiguous(), proj.contiguous()
    plan = rowslot, off, tiles
    dz, dxs = torch.empty_like(a), x.new_empty(T * K, D)
    dfc, dproj = fc.new_empty(fc.shape), proj.new_empty(proj.shape)
    dots = p.new_empty(T * K, _block(a.shape[1], ROWS, "BN")[1]) if silu else None  # per slot, each column program's share of dL/dp
    # Two pairs that do not depend on each other, one per stream: with few rows a launch's last wave of tiles is part
    # empty, and the other stream's kernel fills it. Side stream: dproj, then dfc once dz is there.
    main = torch.cuda.current_stream()
    if OVERLAP and x.device not in _SIDE:
        _SIDE[x.device] = torch.cuda.Stream(x.device)
    side = _SIDE[x.device] if OVERLAP else main
    side.wait_stream(main)
    with torch.cuda.stream(side):
        _weight_grad(dy, a, dproj, plan, lgather=True)  # dy^T a
    _rows_gemm(dy, proj, dz, p, plan, DZ, gather=True, silu=silu, hid=z if silu else a, dots=dots)  # da = dy @ proj, dz = da * da/dz
    side.wait_stream(main)
    with torch.cuda.stream(side):
        _weight_grad(dz, x, dfc, plan, lgather=False)  # dz^T x
    _rows_gemm(dz, fc, dxs, p, plan, scatter=True)  # dx per slot = dz @ fc
    main.wait_stream(side)  # (before the caller's stream can free or reuse what the side stream reads)
    if silu:  # dL/dp = <da, silu(z)>: the dots of dz's epilogue
        return _sum_slots(dxs, T, K), dfc, dproj, dots.sum(1).view(T, K)
    # relu^2: a is of degree 2 in z, so dL/dp = <da, a> / p = <dz @ fc, x> / (2 p): the slot's dx dotted with x, in the slot sum
    dx, dots = _sum_slots(dxs, T, K, x)
    return dx, dfc, dproj, torch.where(p > 0, dots / (2 * p), 0.0)  # (p = 0: an underflowed score, which has no gradient)


@_experts.register_fake
def _(x, fc, proj, sel, p, silu):
    R, (E, H, D) = sel.numel(), fc.shape
    return (x.new_empty(x.shape), x.new_empty(R, H), x.new_empty(R if silu else 0, H),
            sel.new_empty(R), sel.new_empty(E + 1), sel.new_empty(triton.cdiv(R, ROWS["BM"]) + E, 3))


@_experts_backward.register_fake
def _(dy, x, p, a, z, fc, proj, rowslot, off, tiles, silu):
    return x.new_empty(x.shape), fc.new_empty(fc.shape), proj.new_empty(proj.shape), p.new_empty(p.shape)


def _setup(ctx, inputs, output):
    x, fc, proj, sel, p, silu = inputs
    ctx.save_for_backward(x, p, *output[1:3], fc, proj, *output[3:])
    ctx.silu = silu
    ctx.set_materialize_grads(False)  # (only y gets a gradient: no zeros the size of a and z for the other outputs)


def _backward(ctx, dy, *unused):
    dx, dfc, dproj, dp = _experts_backward(dy, *ctx.saved_tensors, ctx.silu)
    return dx, dfc, dproj, None, dp, None


_experts.register_autograd(_backward, setup_context=_setup)


def routed_experts(x, fc, proj, sel, p, activation):
    # y = sum_k p[:, k] * expert_{sel[:, k]}(x), expert_e(x) = act(x @ fc[e]^T) @ proj[e]^T, act "relu2" (relu squared) or "silu";
    # x (T, D) bf16; fc (E, H, D) and proj (E, D, H) in any float dtype, read as they are; sel (T, K), p (T, K) float32. Any D
    # and H, but the kernels load a row as vectors only if its length is a multiple of 16: at other widths the GEMMs run at
    # half speed.
    assert activation in ("relu2", "silu"), f"activation must be 'relu2' or 'silu', not {activation!r}"
    return _experts(x, fc, proj, sel, p, activation == "silu")[0]


# Quantile Balancing (Kimi K3 tech report, 2.3.3): per expert, the bias that would have given it exactly k/E of the tokens
# with every token's cutoff held fixed, read off a histogram of the routing margins (score + bias - the token's cutoff).
# Training forwards add their margins to mlp.margins; the layer's balance() sets its bias from them (base_train: after
# every micro-batch) and resets them. Histograms add, so DDP all-reduces them. A token's cutoff may
# lie anywhere between its k-th and (k+1)-th largest biased scores; the report takes the (k+1)-th, routed (below) their
# midpoint. The histogram covers every margin there can be, as the report's does: +-span, span = 1 + the range of the
# biases, as a score is in (0, 1) and a cutoff lies between two biased scores. (A fixed window caps a step at its
# half-width: an expert left without tokens recovers slowly.)
BINS = 1024  # (the report: 1000. With 512, the first layers of a trained d16 balance worse within a micro-batch)


def margin_counts(s, cutoff, bias, span):
    T, E = s.shape
    idx = (((s + bias) - cutoff[:, None] + span) * (BINS / (2 * span))).long().clamp_(0, BINS - 1)
    return torch.zeros(E, BINS, device=s.device).scatter_add_(1, idx.t(), torch.ones(E, T, device=s.device))


@torch.no_grad()
def quantile_balance(counts, bias, k):
    # The bias from a layer's margin histogram counts (E, BINS), recorded under this bias
    E = counts.shape[0]
    if dist.is_available() and dist.is_initialized():
        counts = counts.clone()
        dist.all_reduce(counts)
    span = 1 + bias.max() - bias.min()  # (the forwards' span: the bias has not changed since)
    above = counts.flip(1).cumsum(1)  # margins in each expert's top i + 1 bins
    target = counts.sum(1, keepdim=True) * k / E
    i = (above < target).sum(1, keepdim=True).clamp_(max=BINS - 1)  # the bin where the target is crossed
    below = above.gather(1, (i - 1).clamp_(min=0)) * (i > 0)
    frac = ((target - below) / (above.gather(1, i) - below).clamp_(min=1)).clamp_(0, 1)
    bias = bias - span * (1 - (i + frac).squeeze(1) * (2 / BINS))
    return bias - bias.mean()


def routed(mlp, x):
    # The routed experts of gpt.py's MLP: sum_k p_k expert_k(x) over each token's k = num_experts_per_token experts with the
    # largest s + bias, where s = sigmoid(router(x)) and p = s / (the sum of the k). The bias only steers the selection;
    # Quantile Balancing sets it between forwards. Every (token, expert) pair is computed: nothing is dropped.
    assert x.is_cuda and x.dtype == torch.bfloat16, "the routed experts' kernels take bfloat16 on CUDA"
    k, shape = mlp.num_experts_per_token, x.shape
    x = x.reshape(-1, shape[-1])
    s = torch.sigmoid(mlp.router(x).float())
    top = (s + mlp.bias).topk(min(k + 1, s.shape[1]), dim=1)  # the k selected and, if there is one, the next
    sel = top.indices[:, :k]
    if mlp.training:  # record the loads and the margins Quantile Balancing will use
        mlp.load.scatter_add_(0, sel.reshape(-1), torch.ones(sel.numel(), device=sel.device))
        # The cutoff: midway between the token's k-th and (k+1)-th largest biased scores (the k-th alone when every
        # expert is selected). What an expert has to beat is the k-th largest biased score of the others: the token's
        # (k+1)-th if it selects the expert, its k-th if not. The midpoint is half the gap off for both, which about
        # halves every step. The exact cutoff overshoots, as all biases move at once (2 experts would swap loads
        # forever); the report's (k+1)-th leaves loads 10-30 % off an even split with 2 experts per token.
        cutoff = top.values[:, k - 1:].mean(1)
        span = 1 + mlp.bias.max() - mlp.bias.min()
        mlp.margins.add_(margin_counts(s.detach(), cutoff.detach(), mlp.bias, span))
    g = s.gather(1, sel)
    return routed_experts(x, mlp.e_fc, mlp.e_proj, sel, g / g.sum(1, keepdim=True), mlp.activation).view(shape)
