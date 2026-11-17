"""A K3-style Mixture-of-Experts layer with fused, dropless expert kernels (Triton). One file, meant to be forked.

    s = sigmoid(x @ W_r)                            router scores, one per expert
    sel = top-k experts by s + bias                 the bias only steers the selection (Quantile Balancing sets it)
    p = s[sel] / sum(s[sel])                        mixture weights: the selected scores, renormalized
    y = shared(x) + sum_k p_k E_{sel_k}(x)          E_e(x) = relu(x @ wu[e].T)**2 @ wd[e].T, weights laid out as in nn.Linear

Dropless: every (token, expert) pair the router picks is computed, whatever the load per expert; nothing is padded.

How the routed experts run. The R = T*K pairs are sorted by expert ("rows") and cut into tiles of at most BM rows of
one expert. The tile table has a fixed length (an upper bound for any routing), so no shape depends on the routing and
the host never waits for it. A conventional implementation copies x into row order, runs two grouped GEMMs and copies
the result back; here each GEMM reads and writes through the permutation itself (`_rows`), so the routed branch keeps
two buffers only: the gated hidden activations a = p * relu(z)**2 (R x H, kept for the backward) and the per-slot
outputs (R x D). The backward is four more GEMMs of the same kind (`_rows` twice, `_wgrad` twice); the gradient of
the mixture weights falls out of the sum over slots. GEMMs use bf16 operands on tensor cores with fp32 accumulators;
weight gradients are fp32.

Not here: K3's latent projection around the routed experts and its SiTU-GLU activation (relu**2 is nanochat's)."""
import torch
import torch.distributed as dist
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl

# tile shapes (rows x output columns per program, chunk of the reduction dimension) and launch options, per kernel
ROWS = dict(BM=128, BN=128, BK=32, num_warps=4, num_stages=3)
WGRAD = dict(BM=128, BN=128, BK=32, num_warps=4, num_stages=3)
WGRAD_LONG = dict(BK=128, num_stages=2)  # WGRAD's loop step on a GPU with room for it (>= 160 KB of shared memory per block: A100, H100)
_WGRAD_DEFAULT = dict(WGRAD)
NONE, ACT, DZ = (tl.constexpr(i) for i in range(3))  # epilogues of _rows


@triton.jit
def _rows(A, W, OUT, P, HID, ROWSLOT, TILES, KD, ND, NP,
          K: tl.constexpr, GATHER: tl.constexpr, SCATTER: tl.constexpr, WT: tl.constexpr, EPI: tl.constexpr,
          BM: tl.constexpr, BN: tl.constexpr, BK: tl.constexpr):
    """OUT[o(rows)] = epilogue(A[a(rows)] @ W[e]) for one tile of rows of expert e and BN output columns.

    With slot = ROWSLOT[row] = token * K + k: A is read in token order (rows slot // K) if GATHER, else in row order;
    OUT is written in slot order if SCATTER, else in row order. W[e] is (KD, ND), or (ND, KD) used transposed if WT.
    Epilogues. ACT (forward): a = p[slot] * relu(.)**2, the hidden activation with its mixture weight folded in, so that
    everything downstream of it is already gated. DZ (backward, . = g = dy @ wd, a = HID[rows]): the gradient of the
    pre-activation, g * d a / d z = g * 2 sqrt(p a)."""
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
    """OUT[e] (MD, ND), float32 = sum over the rows of expert e of L[row]^T Rt[row]: a weight gradient. One operand is
    in row order, the other in token order (gathered): L if LGATHER (d wd = dy^T a), else Rt (d wu = dz^T x)."""
    pm, pn, e = tl.program_id(0), tl.program_id(1), tl.program_id(2)  # an expert's programs run together: its rows stay cached
    r1 = tl.load(OFF + e + 1).to(tl.int32)
    rm = pm * BM + tl.arange(0, BM)
    rn = pn * BN + tl.arange(0, BN)
    mm, mn = rm < MD, rn < ND
    acc = tl.zeros((BM, BN), dtype=tl.float32)
    for r in range(tl.load(OFF + e).to(tl.int32), r1, BK):
        rows = (r + tl.arange(0, BK)).to(tl.int64)  # (a 32-bit loop counter: with a 64-bit one the loop runs ~25% slower)
        mr = rows < r1
        tok = tl.load(ROWSLOT + rows, mask=mr, other=0) // K
        lrow = tok if LGATHER else rows
        rrow = rows if LGATHER else tok
        # (L^T indexed directly, not loaded as rows and transposed: on an A100 the transposition costs more)
        a = tl.load(L + lrow[None, :] * MD + rm[:, None], mask=mm[:, None] & mr[None, :], other=0.0)
        b = tl.load(Rt + rrow[:, None] * ND + rn[None, :], mask=mr[:, None] & mn[None, :], other=0.0)
        acc = tl.dot(a, b, acc)
    tl.store(OUT + (e.to(tl.int64) * MD + rm[:, None]) * ND + rn[None, :], acc, mask=mm[:, None] & mn[None, :])


@triton.jit
def _tiles(COUNTS, OFF, TILES, E, EP: tl.constexpr, BM: tl.constexpr):
    """The plan's tables from the experts' row counts, in one launch: OFF[e], the first row of expert e (OFF[E]: the number
    of rows), and TILES[t] = (expert, first row, row count) of row tile t, with count 0 for a tile past the last one.
    Program t writes OFF[t] and TILES[t]."""
    t = tl.program_id(0)
    ie = tl.arange(0, EP)
    c = tl.load(COUNTS + ie, mask=ie < E, other=0)
    if t <= E:
        tl.store(OFF + t, tl.sum(tl.where(ie < t, c, 0)))
    cum = tl.cumsum((c + BM - 1) // BM, 0)  # the tiles of experts 0..e
    done = (cum <= t) & (ie < E)            # the experts whose tiles all come before t
    e = tl.sum(done.to(tl.int64))
    used = e < E
    e = tl.minimum(e, E - 1)
    j = t - tl.max(tl.where(done, cum, 0))  # the tile's index within its expert
    tl.store(TILES + 3 * t, e)
    tl.store(TILES + 3 * t + 1, tl.sum(tl.where(done, c, 0)) + j * BM)
    tl.store(TILES + 3 * t + 2, tl.where(used, tl.minimum(tl.load(COUNTS + e) - j * BM, BM), 0))


@triton.jit
def _combine(O, Y, X, DOTS, T, D, NP, K: tl.constexpr, DOT: tl.constexpr, BT: tl.constexpr, BD: tl.constexpr):
    """Y[t] = sum_k O[t * K + k]: the K slots of a token, summed. If DOT, also each slot's <O[slot], X[t]>, of which a
    program stores its columns' share in DOTS[slot, program] (NP programs per token, summed by the caller)."""
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
    """The routing sel (T, K) as kernel tables, all of routing-independent shape: rowslot (R,), the slot of each row (the
    pairs sorted by expert, tokens in order within an expert: slot = token * K + k); off (E + 1,), the experts' first
    rows; tiles, the row tiles (see _tiles), as many as any routing can need."""
    T, K = sel.shape
    flat = sel.reshape(-1)
    keys, rowslot = torch.sort(flat, stable=True)
    counts = torch.diff(torch.searchsorted(keys, torch.arange(E + 1, device=flat.device)))  # (torch.bincount would read its maximum back to the host)
    off, tiles = counts.new_empty(E + 1), counts.new_empty(triton.cdiv(T * K, ROWS["BM"]) + E, 3)
    _tiles[(len(tiles),)](counts, off, tiles, E, EP=triton.next_power_of_2(E), BM=ROWS["BM"])
    return rowslot, off, tiles


def _block(n, cfg, key):
    """A tile size for a dimension of n (no larger than needed) and the number of tiles."""
    b = min(cfg[key], triton.next_power_of_2(n))
    return b, triton.cdiv(n, b)


def _rows_gemm(a, w, out, p, plan, epi=NONE, gather=False, scatter=False, wt=False, hid=None):
    rowslot, off, tiles = plan
    hid = out if hid is None else hid  # (only the DZ epilogue reads it)
    kd, nd = a.shape[1], out.shape[1]
    bn, n = _block(nd, ROWS, "BN")
    cfg = dict(ROWS, BN=bn, BK=min(ROWS["BK"], triton.next_power_of_2(kd)))
    _rows[(len(tiles) * n,)](a, w, out, p, hid, rowslot, tiles, kd, nd, n, K=p.shape[1],
                           GATHER=gather, SCATTER=scatter, WT=wt, EPI=epi, **cfg)


def _weight_grad(l, r, E, plan, lgather):
    rowslot, off, tiles = plan
    md, nd = l.shape[1], r.shape[1]
    out = torch.empty(E, md, nd, device=l.device, dtype=torch.float32)
    cfg = WGRAD  # (the long loop step is 5-16 % faster on an A100 at MoE shapes; unless WGRAD was edited, e.g. by a tile sweep)
    if cfg == _WGRAD_DEFAULT and torch.cuda.get_device_properties(l.device).shared_memory_per_block_optin >= 160 * 1024:
        cfg = dict(WGRAD, **WGRAD_LONG)
    bm, m = _block(md, cfg, "BM")
    bn, n = _block(nd, cfg, "BN")
    _wgrad[(m, n, E)](l, r, out, rowslot, off, md, nd, K=len(rowslot) // (l if lgather else r).shape[0], LGATHER=lgather, **dict(cfg, BM=bm, BN=bn))
    return out


def _sum_slots(o, T, K, x=None):
    """(T, D): the K slot rows of each token, summed. With x (T, D), also (T, K): each slot row's dot product with x."""
    D = o.shape[1]
    if K == 1 and x is None:
        return o
    n = triton.cdiv(D, 256)
    y = o.new_empty(T, D)
    dots = y if x is None else o.new_empty(T * K, n, dtype=torch.float32)
    _combine[(triton.cdiv(T, 32), n)](o, y, y if x is None else x, dots, T, D, n, K=K, DOT=x is not None, BT=32, BD=256, num_warps=4)
    return y if x is None else (y, dots.sum(1).view(T, K))


# The forward and the backward are custom operators: opaque single calls to autograd and to torch.compile, which would
# otherwise stop compiling the surrounding model at the first kernel launch. The forward also returns what the backward
# reuses: the gated hidden activations a (R, H) in row order, the bf16 weights and the plan.
@torch.library.custom_op("moe::routed_experts", mutates_args=())
def _experts(x: Tensor, wu: Tensor, wd: Tensor, sel: Tensor, p: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    (T, K), (E, H, D) = sel.shape, wu.shape
    x, p = x.contiguous(), p.contiguous()
    wu = wu.to(x.dtype, copy=True, memory_format=torch.contiguous_format)  # (copy: an operator's outputs may not be its inputs;
    wd = wd.to(x.dtype, copy=True, memory_format=torch.contiguous_format)  # contiguous: the kernels index (E, H, D) and (E, D, H) directly)
    plan = make_plan(sel, E)
    a = x.new_empty(T * K, H)
    _rows_gemm(x, wu, a, p, plan, ACT, gather=True, wt=True)    # a = p * relu(x @ wu^T)**2, in row order
    o = x.new_empty(T * K, D)
    _rows_gemm(a, wd, o, p, plan, scatter=True, wt=True)        # o = a @ wd^T, in slot order
    return _sum_slots(o, T, K), a, wu, wd, *plan


@torch.library.custom_op("moe::routed_experts_backward", mutates_args=())
def _experts_backward(dy: Tensor, x: Tensor, p: Tensor, a: Tensor, wu: Tensor, wd: Tensor,
                      rowslot: Tensor, off: Tensor, tiles: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The gradients of x, wu, wd and p, given dy."""
    (T, K), (E, H, D) = p.shape, wu.shape
    x, p, dy = x.contiguous(), p.contiguous(), dy.contiguous().to(x.dtype)
    plan = rowslot, off, tiles
    dz = torch.empty_like(a)
    _rows_gemm(dy, wd, dz, p, plan, DZ, gather=True, hid=a)     # dz = (dy @ wd) * da/dz, in row order
    dxs = x.new_empty(T * K, D)
    _rows_gemm(dz, wu, dxs, p, plan, scatter=True)              # dx per slot = dz @ wu
    dwu = _weight_grad(dz, x, E, plan, lgather=False)           # dz^T x
    dwd = _weight_grad(dy, a, E, plan, lgather=True)            # dy^T a
    # a = p relu(z)**2 is of degree 2 in z = x @ wu^T, so <da, a> = <dz, z> / 2 = <dz @ wu, x> / 2: the gradient of a
    # mixture weight, <da, a> / p, is its slot's dx dotted with x, which the slot sum reads anyway
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
    """y (T, D) = sum_k p[:, k] * E_{sel[:, k]}(x), with E_e(x) = relu(x @ wu[e].T)**2 @ wd[e].T.
    x: (T, D) bf16; wu: (E, H, D); wd: (E, D, H); sel: (T, K) int64 expert ids, distinct per token; p: (T, K) float32.
    Differentiable in x, wu, wd and p."""
    return _experts(x, wu, wd, sel, p)[0]


def routed_experts_reference(x, wu, wd, sel, p):
    """The same in plain PyTorch, one expert at a time: the definition the kernels are tested against."""
    y = x.new_zeros(x.shape[0], wd.shape[1])
    for e in range(wu.shape[0]):
        t, k = (sel == e).nonzero(as_tuple=True)
        y.index_add_(0, t, p[t, k, None].to(x.dtype) * (torch.relu(x[t] @ wu[e].to(x.dtype).t()).square() @ wd[e].to(x.dtype).t()))
    return y


BINS, SPAN = 512, 0.5  # Quantile Balancing's histogram: BINS bins over the margins in (-SPAN, SPAN)


def margin_counts(s, cutoff, bias):
    """The histogram per expert (E, BINS) of the margins m = s[t, j] + bias[j] - cutoff[t] of the tokens just routed.
    s: (T, E) router scores; cutoff: (T,) each token's (k + 1)-th largest biased score, the bar an expert has to clear
    to be picked: expert j got token t when m > 0. Margins beyond (-SPAN, SPAN) count in the end bins."""
    T, E = s.shape
    idx = (((s + bias) - cutoff[:, None] + SPAN) * (BINS / (2 * SPAN))).long().clamp_(0, BINS - 1)  # ((s + bias) as topk saw it: the cutoff expert's margin is exactly 0)
    return torch.zeros(E, BINS, device=s.device).scatter_add_(1, idx.t(), torch.ones(E, T, device=s.device))


@torch.library.custom_op("moe::quantile_balance", mutates_args=())  # (an operator for torch.compile's sake: the all-reduce)
def quantile_balance(counts: Tensor, bias: Tensor, k: int) -> Tensor:
    """K3's Quantile Balancing: the selection bias (E,) that gives every expert its target load, k / E of the tokens,
    read off the margins the current bias left (counts: margin_counts, summed over whatever was routed with this bias).
    The bias has to move by minus the (1 - k / E) quantile of the expert's margins, read from its histogram, linear
    within a bin: counts add up, so across GPUs one all-reduce gives the whole batch's. The histogram spans
    (-SPAN, SPAN) around the current bias, which is where the answer is once balancing has settled; before that a step
    is at most SPAN."""
    E = counts.shape[0]
    if dist.is_available() and dist.is_initialized():
        counts = counts.clone()
        dist.all_reduce(counts)
    above = counts.flip(1).cumsum(1)  # above[j, i]: how many margins of expert j fall in its top i + 1 bins
    target = counts.sum(1, keepdim=True) * k / E
    i = (above < target).sum(1, keepdim=True).clamp_(max=BINS - 1)  # the bin in which the target is crossed
    below = above.gather(1, (i - 1).clamp_(min=0)) * (i > 0)
    frac = ((target - below) / (above.gather(1, i) - below).clamp_(min=1)).clamp_(0, 1)
    bias = bias - (SPAN - (i + frac).squeeze(1) * (2 * SPAN / BINS))
    return bias - bias.mean()


@quantile_balance.register_fake
def _(counts, bias, k):
    return torch.empty_like(bias)


class K3MoE(nn.Module):
    """dim -> dim. num_experts routed experts of hidden width expert_hidden, top_k active per token, plus shared experts
    (dense, on every token) of total hidden width shared_hidden: N shared MLPs summed are one MLP N times as wide."""

    def __init__(self, dim, num_experts, top_k, expert_hidden, shared_hidden=0):
        super().__init__()
        self.top_k = top_k
        self.router = nn.Parameter(torch.empty(num_experts, dim).uniform_(-1, 1) * (3 / dim) ** 0.5)  # (all-equal scores would tie every token to the same experts)
        self.wu = nn.Parameter(torch.empty(num_experts, expert_hidden, dim).uniform_(-1, 1) * (3 / dim) ** 0.5)
        self.wd = nn.Parameter(torch.zeros(num_experts, dim, expert_hidden))
        self.shared_up = nn.Parameter(torch.empty(shared_hidden, dim).uniform_(-1, 1) * (3 / dim) ** 0.5)
        self.shared_down = nn.Parameter(torch.zeros(dim, shared_hidden))
        self.register_buffer("bias", torch.zeros(num_experts))  # selection bias, set by balance
        self.register_buffer("counts", torch.zeros(num_experts, BINS), persistent=False)  # margins since the last balance
        self.register_buffer("load", torch.zeros(num_experts), persistent=False)  # selections per expert since the last read (GPT.pop_moe_stats), to log

    def route(self, x, sel=None):
        """Expert ids sel (T, K) and mixture weights p (T, K) for the tokens x (T, D). Given sel, those experts are used
        (weights from the present scores): for a model that runs its layers on a sequence more than once and keeps the
        first pass's choice of experts."""
        k = self.top_k
        s = torch.sigmoid(F.linear(x, self.router.to(x.dtype)).float())
        if sel is None:
            top = (s + self.bias).topk(k + 1, dim=1)
            sel = top.indices[:, :k]
            if self.training:
                self.counts += margin_counts(s.detach(), top.values[:, k].detach(), self.bias)
                self.load.scatter_add_(0, sel.reshape(-1), torch.ones(sel.numel(), device=sel.device))
        g = s.gather(1, sel)
        return sel, g / g.sum(1, keepdim=True)

    def routed(self, x, sel=None):
        """The layer's output and the experts it used, sel (tokens, K). sel: see route. For a batch that goes through the
        layer more than once: the caller calls balance() after its last pass (forward does, after its one)."""
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        sel, p = self.route(x, sel)
        y = routed_experts(x, self.wu, self.wd, sel, p) if x.is_cuda and x.dtype == torch.bfloat16 else routed_experts_reference(x, self.wu, self.wd, sel, p)
        if self.shared_up.numel():
            y = y + F.linear(F.relu(F.linear(x, self.shared_up.to(x.dtype))).square(), self.shared_down.to(x.dtype))
        return y.view(shape), sel

    def balance(self):
        """The bias for the next batch, from everything routed since the last call: a batch is never routed with a bias
        derived from itself, and one that is routed more than once (routed) is balanced over all its passes."""
        self.bias.copy_(quantile_balance(self.counts, self.bias, self.top_k))
        self.counts.zero_()

    def forward(self, x):
        y = self.routed(x)[0]
        if self.training:
            self.balance()
        return y
