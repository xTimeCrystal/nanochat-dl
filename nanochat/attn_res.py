"""
Attention Residuals (Kimi Team 2026, github.com/MoonshotAI/Attention-Residuals): each block reads a softmax mixture over
depth of everything before it, instead of the running sum of a residual stream,

    h_l = sum_i softmax_i(s_li) v_i,    s_li = u_l . v_i / rms(v_i),

over the sources v_i (the backbone's input, then every block's output) with one learned pseudo-query u_l per block.
The block reads norm(h_l); the last mixture, after the last block, is the backbone's output (normed).

    backbone = AttnResBackbone(blocks, model_dim)   # blocks: nn.ModuleList, block(x, *args, **kwargs) -> its output
    y = backbone(x, *args, **kwargs)                # x: (..., model_dim), extra arguments passed on to every block

The input stays outside: a caller may build it however it likes (e.g. a Full Bandwidth Transformer's feedback input) and
may run the backbone any number of times per step, gradients flowing through every run. Backbones nest: a block can be
another AttnResBackbone, or an IdentityResidualBackbone (x = x + block(x)), which gives block AttnRes.

On CUDA with bf16 activations the mixtures run in Triton kernels that move little memory:
- every source is kept once and a mixture saves only its output (norm(h) in bf16 and its bf16 rounding residual, which
  hands the backward a 16-bit h), its logsumexp and rstd;
- forward: the pseudo-queries are known in advance, so one pass over the sources so far completes a mixture and leaves
  the online-softmax state (float32 sum, running max and sum) of the next BLOCK - 1; each of those then reads its state
  and only the sources added since, not every source;
- backward: the sources added between two block boundaries (every BLOCK mixtures) form a group; from the boundary on, one
  lead node stands in for the whole group in every later mixture, so its backward reads each later mixture's gradient
  once for all of them and hands each member a float32 partial; a member then reads only the gradients of the mixtures
  before the boundary.
Elsewhere a plain PyTorch implementation (the reference) runs instead.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError: # no Triton (CPU / MPS): the kernels below are never launched, the backbone runs its PyTorch path
    triton, tl, libdevice = SimpleNamespace(jit=lambda fn: fn), SimpleNamespace(constexpr=None), None

EPS = torch.finfo(torch.float32).eps
BLOCK = 4  # mixtures per forward block and per backward group (a power of 2)
TILE = dict(elems=1024, num_warps=1)  # elements and warps per program: _fwd, _bwd, _bwd_src
BLOCK_TILE = dict(elems=2048, num_warps=2)  # _partial (BLOCK accumulators per row)
LEAD_TILE = dict(elems=1024, num_warps=2)  # _bwd_lead (same)


@triton.jit
def _rstd(v, eps, C: tl.constexpr):
    return 1.0 / tl.sqrt(tl.sum(v * v, -1) / C + eps)


@triton.jit
def _fwd(SRC, U, Y, YLO, LSE, RSTD, ACC, MX, SM, M, K, eps, C: tl.constexpr, HAS_P: tl.constexpr,
         BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr):
    """One mixture over the K sources of the pointer table SRC, continued from a saved state (ACC, MX, SM) if HAS_P."""
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rc = tl.arange(0, BLOCK_C)
    mm, mc = rm < M, rc < C
    mask = mm[:, None] & mc[None, :]
    u = tl.load(U + rc, mask=mc, other=0.0)
    if HAS_P:
        m_i = tl.load(MX + rm, mask=mm, other=0.0)
        l_i = tl.load(SM + rm, mask=mm, other=1.0)
        acc = tl.load(ACC + rm[:, None] * C + rc[None, :], mask=mask, other=0.0)
    else:
        m_i = tl.full([BLOCK_M], float("-inf"), tl.float32)
        l_i = tl.zeros([BLOCK_M], tl.float32)
        acc = tl.zeros([BLOCK_M, BLOCK_C], tl.float32)
    for k in range(K):
        v = tl.load(tl.load(SRC + k).to(tl.pointer_type(tl.bfloat16)) + rm[:, None] * C + rc[None, :], mask=mask, other=0.0).to(tl.float32)
        s = libdevice.mul_rn(tl.sum(v * u[None, :], 1), _rstd(v, eps, C))  # (a rounded product, no FMA: the backward recomputes it bitwise)
        m_new = tl.maximum(m_i, s)
        a, b = tl.exp(m_i - m_new), tl.exp(s - m_new)
        acc = acc * a[:, None] + b[:, None] * v
        l_i = l_i * a + b
        m_i = m_new
    h = acc / l_i[:, None]
    rstd = _rstd(h, eps, C)
    tl.store(LSE + rm, m_i + tl.log(l_i), mask=mm)
    tl.store(RSTD + rm, rstd, mask=mm)
    y = h * rstd[:, None]
    yb = y.to(tl.bfloat16)
    tl.store(Y + rm[:, None] * C + rc[None, :], yb, mask=mask)
    tl.store(YLO + rm[:, None] * C + rc[None, :], (y - yb.to(tl.float32)).to(tl.bfloat16), mask=mask)


@triton.jit
def _partial(SRC, U, Y, YLO, LSE, RSTD, ACC, MX, SM, M, K, NB, eps, C: tl.constexpr, NBP: tl.constexpr,
             BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr):
    """The NB mixtures U[0..NB-1] over the K sources of SRC in one pass: mixture 0 completed as in _fwd, the online-softmax
    state of mixtures 1..NB-1 stored (ACC[b-1] = sum_k exp(s_bk - MX[b-1]) v_k, MX, SM = the running max and sum)."""
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rc = tl.arange(0, BLOCK_C)
    rb = tl.arange(0, NBP)
    mm, mc, mb = rm < M, rc < C, rb < NB
    mask = mm[:, None] & mc[None, :]
    u = tl.load(U + rb[:, None] * C + rc[None, :], mask=mb[:, None] & mc[None, :], other=0.0)
    m_i = tl.full([NBP, BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([NBP, BLOCK_M], tl.float32)
    acc = tl.zeros([NBP, BLOCK_M, BLOCK_C], tl.float32)
    for k in range(K):
        v = tl.load(tl.load(SRC + k).to(tl.pointer_type(tl.bfloat16)) + rm[:, None] * C + rc[None, :], mask=mask, other=0.0).to(tl.float32)
        s = libdevice.mul_rn(tl.sum(v[None, :, :] * u[:, None, :], 2), _rstd(v, eps, C)[None, :])
        m_new = tl.maximum(m_i, s)
        a, b = tl.exp(m_i - m_new), tl.exp(s - m_new)
        acc = acc * a[:, :, None] + b[:, :, None] * v[None, :, :]
        l_i = l_i * a + b
        m_i = m_new
    first = (rb == 0)[:, None]
    m0, l0 = tl.sum(tl.where(first, m_i, 0.0), 0), tl.sum(tl.where(first, l_i, 0.0), 0)
    h = tl.sum(tl.where(first[:, :, None], acc, 0.0), 0) / l0[:, None]
    rstd = _rstd(h, eps, C)
    tl.store(LSE + rm, m0 + tl.log(l0), mask=mm)
    tl.store(RSTD + rm, rstd, mask=mm)
    y = h * rstd[:, None]
    yb = y.to(tl.bfloat16)
    tl.store(Y + rm[:, None] * C + rc[None, :], yb, mask=mask)
    tl.store(YLO + rm[:, None] * C + rc[None, :], (y - yb.to(tl.float32)).to(tl.bfloat16), mask=mask)
    rest = (rb >= 1) & mb
    ob = tl.maximum(rb - 1, 0).to(tl.int64)
    tl.store(MX + ob[:, None] * M + rm[None, :], m_i, mask=rest[:, None] & mm[None, :])
    tl.store(SM + ob[:, None] * M + rm[None, :], l_i, mask=rest[:, None] & mm[None, :])
    tl.store(ACC + (ob[:, None, None] * M + rm[None, :, None]) * C + rc[None, None, :], acc, mask=rest[:, None, None] & mask[None, :, :])


@triton.jit
def _bwd(DY, Y, YLO, RSTD, LSE, G, P, M, C: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr):
    """A mixture's norm backward from its output as y + y_lo (no source read): G = dh (bf16), P = (logsumexp, G . h)."""
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rc = tl.arange(0, BLOCK_C)
    mm, mc = rm < M, rc < C
    mask = mm[:, None] & mc[None, :]
    rstd = tl.load(RSTD + rm, mask=mm, other=1.0)
    dy = tl.load(DY + rm[:, None] * C + rc[None, :], mask=mask, other=0.0).to(tl.float32)
    y = (tl.load(Y + rm[:, None] * C + rc[None, :], mask=mask, other=0.0).to(tl.float32)
         + tl.load(YLO + rm[:, None] * C + rc[None, :], mask=mask, other=0.0).to(tl.float32))
    dh = (rstd[:, None] * (dy - y * (tl.sum(dy * y, 1) / C)[:, None])).to(tl.bfloat16)
    tl.store(G + rm[:, None] * C + rc[None, :], dh, mask=mask)
    tl.store(P + rm * 2, tl.load(LSE + rm, mask=mm, other=0.0), mask=mm)
    tl.store(P + rm * 2 + 1, tl.sum(dh.to(tl.float32) * y, 1) / rstd, mask=mm)


@triton.jit
def _bwd_src(V, GT, PT, U, DV, W, PART, M, N, eps, C: tl.constexpr, HAS_P: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr):
    """dv of a source over its N consumers (G, P from the tables GT, PT; their queries U), added to PART if HAS_P;
    W[m] = g r per consumer and token, for du_m = W_m @ v."""
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rc = tl.arange(0, BLOCK_C)
    mm, mc = rm < M, rc < C
    mask = mm[:, None] & mc[None, :]
    v = tl.load(V + rm[:, None] * C + rc[None, :], mask=mask, other=0.0).to(tl.float32)
    r = _rstd(v, eps, C)
    if HAS_P:
        acc = tl.load(PART + rm[:, None] * C + rc[None, :], mask=mask, other=0.0)
    else:
        acc = tl.zeros([BLOCK_M, BLOCK_C], tl.float32)
    for m in range(N):
        dh = tl.load(tl.load(GT + m).to(tl.pointer_type(tl.bfloat16)) + rm[:, None] * C + rc[None, :], mask=mask, other=0.0).to(tl.float32)
        pp = tl.load(PT + m).to(tl.pointer_type(tl.float32))
        lse, c = tl.load(pp + rm * 2, mask=mm, other=0.0), tl.load(pp + rm * 2 + 1, mask=mm, other=0.0)
        u = tl.load(U + m * C + rc, mask=mc, other=0.0)
        uv = tl.sum(v * u[None, :], 1)
        alpha = tl.exp(libdevice.mul_rn(uv, r) - lse)  # the forward's softmax weight
        g = alpha * (tl.sum(dh * v, 1) - c)
        acc += alpha[:, None] * dh + g[:, None] * (r[:, None] * u[None, :] - (r * r * r * uv / C)[:, None] * v)
        tl.store(W + m * M + rm, (g * r).to(tl.bfloat16), mask=mm)
    tl.store(DV + rm[:, None] * C + rc[None, :], acc.to(tl.bfloat16), mask=mask)


@triton.jit
def _bwd_lead(VT, GT, PT, U, PART, W, M, N, NB, eps, C: tl.constexpr, NBP: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr):
    """_bwd_src for the NB sources of VT at once over N consumers, each consumer's G read once: PART[b] (float32 partial
    dv of member b), W[b, m]."""
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rc = tl.arange(0, BLOCK_C)
    rb = tl.arange(0, NBP)
    mm, mc, mb = rm < M, rc < C, rb < NB
    mask = mm[:, None] & mc[None, :]
    vp = tl.load(VT + rb, mask=mb, other=0).to(tl.pointer_type(tl.bfloat16))
    v = tl.load(vp[:, None, None] + (rm[None, :, None] * C + rc[None, None, :]), mask=mb[:, None, None] & mask[None, :, :], other=0.0).to(tl.float32)
    r = _rstd(v, eps, C)  # (NBP, BLOCK_M)
    acc = tl.zeros([NBP, BLOCK_M, BLOCK_C], tl.float32)
    gn = tl.load(tl.load(GT).to(tl.pointer_type(tl.bfloat16)) + rm[:, None] * C + rc[None, :], mask=mask & (N > 0), other=0.0)
    for m in range(N):
        dh = gn.to(tl.float32)
        gn = tl.load(tl.load(GT + tl.minimum(m + 1, N - 1)).to(tl.pointer_type(tl.bfloat16)) + rm[:, None] * C + rc[None, :],
                     mask=mask & (m + 1 < N), other=0.0)  # (the next G, loaded while this one is used)
        pp = tl.load(PT + m).to(tl.pointer_type(tl.float32))
        lse, c = tl.load(pp + rm * 2, mask=mm, other=0.0), tl.load(pp + rm * 2 + 1, mask=mm, other=0.0)
        u = tl.load(U + m * C + rc, mask=mc, other=0.0)
        uv = tl.sum(v * u[None, None, :], 2)
        alpha = tl.exp(libdevice.mul_rn(uv, r) - lse[None, :])
        g = alpha * (tl.sum(dh[None, :, :] * v, 2) - c[None, :])
        acc += alpha[:, :, None] * dh[None, :, :] + g[:, :, None] * (r[:, :, None] * u[None, None, :] - (r * r * r * uv / C)[:, :, None] * v)
        tl.store(W + (rb[:, None].to(tl.int64) * N + m) * M + rm[None, :], (g * r).to(tl.bfloat16), mask=mb[:, None] & mm[None, :])
    tl.store(PART + (rb[:, None, None].to(tl.int64) * M + rm[None, :, None]) * C + rc[None, None, :], acc, mask=mb[:, None, None] & mask[None, :, :])


def _grid(M, C, tile):
    bc = triton.next_power_of_2(C)
    bm = max(1, min(64, tile["elems"] // bc))
    return (triton.cdiv(M, bm),), dict(BLOCK_C=bc, BLOCK_M=bm, num_warps=tile["num_warps"])


def _table(ts):
    """A device table of data pointers, copied from pinned memory without blocking the host."""
    return torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64).pin_memory().to(ts[0].device, non_blocking=True)


# ---- custom ops: opaque to torch.compile, every tensor they touch is an argument ----

@torch.library.custom_op("attn_res::fwd", mutates_args=(), device_types="cuda")
def _fwd_op(vs: list[Tensor], u: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """norm(h) (bf16), its bf16 rounding residual, logsumexp and rstd(h) of the mixture of vs ((M, C) bf16) under u ((C,) f32)."""
    M, C = vs[0].shape
    y, y_lo = torch.empty_like(vs[0]), torch.empty_like(vs[0])
    lse, rstd = vs[0].new_empty(M, dtype=torch.float32), vs[0].new_empty(M, dtype=torch.float32)
    grid, cfg = _grid(M, C, TILE)
    _fwd[grid](_table(vs), u, y, y_lo, lse, rstd, y, lse, lse, M, len(vs), EPS, C=C, HAS_P=False, **cfg)
    return y, y_lo, lse, rstd


@torch.library.custom_op("attn_res::partial", mutates_args=(), device_types="cuda")
def _partial_op(vs: list[Tensor], U: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The mixtures U ((nb, C), nb >= 2) over vs: mixture 0 as _fwd_op, the others' online-softmax state (nb-1 rows)."""
    M, C = vs[0].shape
    nb = U.shape[0]
    y, y_lo = torch.empty_like(vs[0]), torch.empty_like(vs[0])
    lse, rstd = vs[0].new_empty(M, dtype=torch.float32), vs[0].new_empty(M, dtype=torch.float32)
    acc = vs[0].new_empty(nb - 1, M, C, dtype=torch.float32)
    mx, sm = vs[0].new_empty(nb - 1, M, dtype=torch.float32), vs[0].new_empty(nb - 1, M, dtype=torch.float32)
    grid, cfg = _grid(M, C, BLOCK_TILE)
    _partial[grid](_table(vs), U, y, y_lo, lse, rstd, acc, mx, sm, M, len(vs), nb, EPS, C=C, NBP=triton.next_power_of_2(nb), **cfg)
    return y, y_lo, lse, rstd, acc, mx, sm


@torch.library.custom_op("attn_res::fwd_from", mutates_args=(), device_types="cuda")
def _fwd_from_op(vs: list[Tensor], u: Tensor, acc: Tensor, mx: Tensor, sm: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """_fwd_op's outputs for a mixture continued from its state (acc (M, C), mx, sm (M,) f32) over the newer sources vs."""
    M, C = acc.shape
    y, y_lo = acc.new_empty(M, C, dtype=torch.bfloat16), acc.new_empty(M, C, dtype=torch.bfloat16)
    lse, rstd = mx.new_empty(M), mx.new_empty(M)
    grid, cfg = _grid(M, C, TILE)
    _fwd[grid](_table(vs) if vs else mx, u, y, y_lo, lse, rstd, acc, mx, sm, M, len(vs), EPS, C=C, HAS_P=True, **cfg)
    return y, y_lo, lse, rstd


@torch.library.custom_op("attn_res::bwd", mutates_args=(), device_types="cuda")
def _bwd_op(dy: Tensor, y: Tensor, y_lo: Tensor, rstd: Tensor, lse: Tensor) -> tuple[Tensor, Tensor]:
    """G = dh ((M, C) bf16) and P = (logsumexp, G . h) ((M, 2) f32) of a mixture, from dy and its saved outputs."""
    M, C = y.shape
    G, P = torch.empty_like(y), y.new_empty(M, 2, dtype=torch.float32)
    grid, cfg = _grid(M, C, TILE)
    _bwd[grid](dy, y, y_lo, rstd, lse, G, P, M, C=C, **cfg)
    return G, P


@torch.library.custom_op("attn_res::bwd_src", mutates_args=(), device_types="cuda")
def _bwd_src_op(v: Tensor, Gs: list[Tensor], Ps: list[Tensor], U: Tensor, part: Tensor | None) -> tuple[Tensor, Tensor]:
    """dv of the source v over its consumers (their G, P and queries U), plus part (a float32 partial) if given; and the
    source's share of each consumer's du ((N, C) f32)."""
    M, C = v.shape
    dv, W = torch.empty_like(v), v.new_empty(len(Gs), M)
    tables = (_table(Gs), _table(Ps)) if Gs else (W, W)
    grid, cfg = _grid(M, C, TILE)
    _bwd_src[grid](v, *tables, U, dv, W, dv if part is None else part, M, len(Gs), EPS, C=C, HAS_P=part is not None, **cfg)
    return dv, torch.mm(W, v, out_dtype=torch.float32)


@torch.library.custom_op("attn_res::bwd_lead", mutates_args=(), device_types="cuda")
def _bwd_lead_op(vs: list[Tensor], Gs: list[Tensor], Ps: list[Tensor], U: Tensor) -> tuple[Tensor, Tensor]:
    """The float32 partial dv of every source of vs ((nb, M, C)) over the consumers (Gs, Ps, U), and their share of du."""
    M, C = vs[0].shape
    part, W = vs[0].new_empty(len(vs), M, C, dtype=torch.float32), vs[0].new_empty(len(vs), len(Gs), M)
    grid, cfg = _grid(M, C, LEAD_TILE)
    _bwd_lead[grid](_table(vs), _table(Gs), _table(Ps), U, part, W, M, len(Gs), len(vs), EPS, C=C, NBP=triton.next_power_of_2(len(vs)), **cfg)
    du = torch.mm(W[0], vs[0], out_dtype=torch.float32)
    for b in range(1, len(vs)):
        du += torch.mm(W[b], vs[b], out_dtype=torch.float32)
    return part, du


@torch.library.custom_op("attn_res::source", mutates_args=())
def _source_op(v: Tensor) -> Tensor:
    """A contiguous bf16 copy of a source that is neither."""
    return torch.empty_like(v, dtype=torch.bfloat16, memory_format=torch.contiguous_format).copy_(v)


def _f32(t, *shape):
    return t.new_empty(*shape, dtype=torch.float32)


_fwd_op.register_fake(lambda vs, u: (torch.empty_like(vs[0]), torch.empty_like(vs[0]), _f32(vs[0], vs[0].shape[0]), _f32(vs[0], vs[0].shape[0])))
_partial_op.register_fake(lambda vs, U: (torch.empty_like(vs[0]), torch.empty_like(vs[0]), _f32(vs[0], vs[0].shape[0]), _f32(vs[0], vs[0].shape[0]),
                                         _f32(vs[0], U.shape[0] - 1, *vs[0].shape), _f32(vs[0], U.shape[0] - 1, vs[0].shape[0]),
                                         _f32(vs[0], U.shape[0] - 1, vs[0].shape[0])))
_fwd_from_op.register_fake(lambda vs, u, acc, mx, sm: (acc.new_empty(acc.shape, dtype=torch.bfloat16), acc.new_empty(acc.shape, dtype=torch.bfloat16),
                                                      torch.empty_like(mx), torch.empty_like(mx)))
_bwd_op.register_fake(lambda dy, y, y_lo, rstd, lse: (torch.empty_like(y), _f32(y, y.shape[0], 2)))
_bwd_src_op.register_fake(lambda v, Gs, Ps, U, part: (torch.empty_like(v), _f32(v, U.shape[0], v.shape[1])))
_bwd_lead_op.register_fake(lambda vs, Gs, Ps, U: (_f32(vs[0], len(vs), *vs[0].shape), _f32(vs[0], U.shape[0], vs[0].shape[1])))
_source_op.register_fake(lambda v: torch.empty_like(v, dtype=torch.bfloat16, memory_format=torch.contiguous_format))
_source_op.register_autograd(lambda ctx, g: g.to(ctx.dtype), setup_context=lambda ctx, inputs, output: setattr(ctx, "dtype", inputs[0].dtype))


# ---- autograd. A source returns one zero-memory token pair per consumer ((M, C) bf16 and (M, 2) f32 expanded zeros); a
# mixture takes its sources' tokens and returns its G and P as their gradients, so a source's backward runs once every
# consumer's has, with all their G and P in hand. Nothing is mutated in the backward: AOTAutograd can trace it. ----

def _tokens(v, U):
    zb, zf = v.new_zeros(()), U.new_zeros(())
    n, (M, C) = U.shape[0], v.shape
    return (*(zb.expand(M, C) for _ in range(n)), *(zf.expand(M, 2) for _ in range(n)))


def _live(grads, n):
    """The consumers whose mixture ran (a contiguous range a..b): their G and P."""
    live = [m for m in range(n) if grads[m] is not None]
    if not live:
        return None
    a, b = live[0], live[-1] + 1
    assert b - a == len(live)
    return a, b, [grads[m].contiguous() for m in live], [grads[n + m].contiguous() for m in live]


class _Source(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, U, member):
        """v: (M, C) bf16; U: the queries of its consumers. A group member has one more output, a float32 token that the
        group's lead node takes, whose gradient is v's partial dv over the mixtures after the group's boundary."""
        ctx.save_for_backward(v, U)
        ctx.member = member
        ctx.set_materialize_grads(False)
        return _tokens(v, U) + ((U.new_zeros(()).expand(v.shape),) if member else ())

    @staticmethod
    def backward(ctx, *grads):
        v, U = ctx.saved_tensors
        n = U.shape[0]
        part = grads[2 * n] if ctx.member else None
        live = _live(grads, n)
        if live is None:
            return (None if part is None else part.to(v.dtype)), None, None
        a, b, Gs, Ps = live
        dv, du = torch.ops.attn_res.bwd_src(v, Gs, Ps, U[a:b], None if part is None else part.contiguous())
        return dv, F.pad(du, (0, 0, a, n - b)), None


class _Lead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, n_members, *args):
        """A group's stand-in in the mixtures after its boundary (queries U): args = the members' sources, then their
        float32 tokens. One token pair per mixture, as a source's."""
        ctx.save_for_backward(U, *args[:n_members])
        ctx.n_members = n_members
        ctx.set_materialize_grads(False)
        return _tokens(args[0], U)

    @staticmethod
    def backward(ctx, *grads):
        U, *vs = ctx.saved_tensors
        n = U.shape[0]
        live = _live(grads, n)
        if live is None:
            return (None,) * (2 + 2 * ctx.n_members)
        a, b, Gs, Ps = live
        part, du = torch.ops.attn_res.bwd_lead(vs, Gs, Ps, U[a:b])
        return F.pad(du, (0, 0, a, n - b)), None, *((None,) * ctx.n_members), *part.unbind(0)


class _Mix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, n_src, n_new, acc, mx, sm, *args):
        """args: the n_src sources, then this mixture's tokens. acc None: U (C,) mixes every source, or U (nb, C) mixes
        under U[0] and also returns the state of the next nb - 1 mixtures (non-differentiable); else continued from a saved
        state (acc, mx, sm) over the n_new newest sources."""
        vs = list(args[:n_src])
        if acc is not None:
            y, y_lo, lse, rstd = torch.ops.attn_res.fwd_from(vs[n_src - n_new:], U, acc, mx, sm)
            out = y
        elif U.dim() == 2:
            y, y_lo, lse, rstd, acc, mx, sm = torch.ops.attn_res.partial(vs, U)
            ctx.mark_non_differentiable(acc, mx, sm)
            out = y, acc, mx, sm
        else:
            y, y_lo, lse, rstd = torch.ops.attn_res.fwd(vs, U)
            out = y
        ctx.save_for_backward(lse, rstd, y, y_lo)
        ctx.n_src, ctx.n_tok = n_src, (len(args) - n_src) // 2
        return out

    @staticmethod
    def backward(ctx, dy, *unused):
        lse, rstd, y, y_lo = ctx.saved_tensors
        G, P = torch.ops.attn_res.bwd(dy.contiguous(), y, y_lo, rstd, lse)
        return (None,) * (6 + ctx.n_src) + (G,) * ctx.n_tok + (P,) * ctx.n_tok


class _Stream:
    """One run of a backbone: its sources and mixtures, through the kernels (bf16 on CUDA)."""

    def __init__(self, x, U):
        self.U, self.shape, self.i = U, x.shape, 0  # U (n, C) float32: one query per mixture; i: mixtures so far
        self.grad = torch.is_grad_enabled()
        self.v = []  # the sources, (M, C) bf16, detached
        self.tg, self.ts = [[] for _ in range(len(U))], [[] for _ in range(len(U))]  # per mixture: its tokens
        self.group = {}  # group boundary (a mixture index) -> its members: (source index, float32 token)
        self.state = None  # (first mixture, nb, sources it covers, acc, mx, sm): the saved state of mixtures first+1 .. first+nb-1
        self.append(x)

    def append(self, v):
        v = v.reshape(-1, v.shape[-1])
        if v.dtype != torch.bfloat16 or not v.is_contiguous():
            v = _source_op(v)
        self.v.append(v.detach())
        if not self.grad:
            return
        end = (self.i // BLOCK + 1) * BLOCK  # this source's group boundary
        member = BLOCK > 1 and end < len(self.U)
        U = self.U[self.i:end] if member else self.U[self.i:]
        toks = _Source.apply(v, U, member)
        n = U.shape[0]
        if member:
            self.group.setdefault(end, []).append((len(self.v) - 1, toks[2 * n]))
        for m in range(n):
            self.tg[self.i + m].append(toks[m])
            self.ts[self.i + m].append(toks[n + m])

    def read(self):
        i, self.i = self.i, self.i + 1
        n_src = len(self.v)
        if self.grad and i in self.group:  # a group boundary: from here on the lead node stands in for the group
            members = self.group.pop(i)
            toks = _Lead.apply(self.U[i:], len(members), *(self.v[j] for j, _ in members), *(t for _, t in members))
            n = len(self.U) - i
            for m in range(n):
                self.tg[i + m].append(toks[m])
                self.ts[i + m].append(toks[n + m])
        s = self.state
        if s is not None and 1 <= i - s[0] < s[1]:  # inside a forward block: its saved state, then the sources added since
            r = i - s[0] - 1
            n_new, state, U = n_src - s[2], (s[3][r], s[4][r], s[5][r]), self.U[i]
        else:  # a block's first mixture: one pass over every source, for the whole block
            n_new, state, U = 0, (None, None, None), self.U[i:i + min(BLOCK, len(self.U) - i)]
            U = U[0] if len(U) == 1 else U
        if self.grad:
            out = _Mix.apply(U, n_src, n_new, *state, *self.v, *self.tg[i], *self.ts[i])
        elif state[0] is not None:
            out = torch.ops.attn_res.fwd_from(self.v[n_src - n_new:], U, *state)[0]
        elif U.dim() == 2:
            out = torch.ops.attn_res.partial(self.v, U)
            out = (out[0], *out[4:])
        else:
            out = torch.ops.attn_res.fwd(self.v, U)[0]
        if isinstance(out, tuple):
            out, *saved = out
            self.state = (i, U.shape[0], n_src, *saved)
        return out.view(self.shape)


def reference_mix(vs, u):
    """norm(sum_i softmax_i(u . v_i / rms(v_i)) v_i) over the sources vs, in float32 (the definition the kernels follow)."""
    V = torch.stack([v.float() for v in vs])
    s = (V * u.float()).sum(-1) * (V.square().mean(-1) + EPS).rsqrt()
    h = (s.softmax(0).unsqueeze(-1) * V).sum(0)
    return F.rms_norm(h, h.shape[-1:], eps=EPS).to(vs[0].dtype)


class _ReferenceStream:
    """The same in plain PyTorch, for other devices and dtypes (every source stacked at every mixture: O(L^2) memory)."""

    def __init__(self, x, U):
        self.U, self.v = U, [x]

    def append(self, v):
        self.v.append(v)

    def read(self):
        return reference_mix(self.v, self.U[len(self.v) - 2])


class AttnResBackbone(nn.Module):
    """Attention Residuals over blocks (an nn.ModuleList): block i reads the mixture of the input and the outputs of
    blocks 0..i-1 (block 0 reads the input itself) and the output is the mixture after the last block. One pseudo-query
    per block, zero-initialised: every mixture starts uniform."""

    def __init__(self, blocks: nn.ModuleList, model_dim: int):
        super().__init__()
        self.blocks = blocks
        self.pseudo_queries = nn.Parameter(torch.zeros(len(blocks), model_dim))

    def forward(self, x, *args, **kwargs):
        stream = (_Stream if x.is_cuda and x.dtype == torch.bfloat16 else _ReferenceStream)(x, self.pseudo_queries.float())
        for block in self.blocks:
            stream.append(block(x, *args, **kwargs))
            x = stream.read()
        return x


class IdentityResidualBackbone(nn.Module):
    """The plain residual stream over blocks: x = x + block(x). As a block of an AttnResBackbone: block AttnRes."""

    def __init__(self, blocks: nn.ModuleList, model_dim: int = None):
        super().__init__()
        self.blocks = blocks

    def forward(self, x, *args, **kwargs):
        for block in self.blocks:
            x = x + block(x, *args, **kwargs)
        return x
