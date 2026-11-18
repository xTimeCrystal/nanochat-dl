"""
Attention Residuals (Kimi Team 2026, github.com/MoonshotAI/Attention-Residuals) as a residual backbone.

A residual stream gives every block norm(x), x the running sum of the input and the block outputs so far. Here the
running sum can be ended before any block: what was summed is kept as a source, the block's output starts a new sum,
and every block reads a softmax mixture over the kept sources and the current sum, with its own pseudo-query u:

    h = sum_i softmax_i(s_i) v_i,    s_i = u . v_i / rms(v_i);    the block reads norm(h).

    backbone = AttnResBackbone(blocks, model_dim, starts=None)   # blocks: nn.ModuleList, block(x, *args, **kwargs) -> its output
    y = backbone(x, *args, **kwargs)                             # the first block reads x itself; y: the mixture after the last

starts: the indices of the blocks whose output starts a new source.
- None, every block: the input and every block output are sources, (full) Attention Residuals.
- range(0, len(blocks), S): the paper's Block AttnRes with S sublayers per block (the input is a source, then each
  block's sublayers sum into one).
- (), never: one sum, the input included, so every mixture has a single source and is its norm: the pre-norm residual
  stream x = x + block(norm(x)), at that stream's cost (the kernels then are an RMSNorm and its backward).
The input stays outside: a caller builds it as it likes (a Full Bandwidth Transformer's feedback input) and may run the
backbone any number of times per step, gradients flowing through every run. A block can itself be a backbone.

On CUDA with bf16 activations the mixtures run in Triton kernels (plain PyTorch elsewhere). What they cost grows with
the number of kept sources, not of blocks:
- memory: every source is kept once (the block's output itself, or the running sum). Eager, a mixture keeps its output
  (the next block's input anyway) and, if it read kept sources, two floats per token; compiled, it is a checkpoint and
  keeps nothing: its output is rebuilt from the sources in the backward.
- forward: the pseudo-queries are known in advance, so one pass over the kept sources completes a mixture and leaves the
  online-softmax state (bf16 sum, running max and sum) of the next BLOCK - 1; each of those then reads its state and the
  sources added since.
- backward: a source that one mixture alone reads (a running sum that goes on) gets its gradient from that mixture
  directly. A kept source collects it from the mixtures that read it: BLOCK consecutive ones form a group, and from the
  next mixture on one lead node stands in for the group, reading each later mixture's gradient once for all of them.
"""

from types import SimpleNamespace

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from torch import Tensor
try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
except ImportError: # no Triton (CPU / MPS): the kernels below are never launched, the backbone runs its PyTorch path
    triton, tl, libdevice = SimpleNamespace(jit=lambda fn: fn), SimpleNamespace(constexpr=None), None

EPS = torch.finfo(torch.float32).eps
BLOCK = 4  # mixtures per forward pass over the kept sources, and kept sources per backward group (2: a few (M, C) tensors less in flight, 1-2 % slower)
TILE = dict(elems=1024, num_warps=1)  # elements and warps per program: _mix, _fwd, _bwd, _bwd_src (bandwidth-bound at any tile)
LEAD_TILE = dict(elems=1024, num_warps=4)  # _bwd_lead (an accumulator per source and row)


@triton.jit
def _rstd(v, eps, C: tl.constexpr):
    return 1.0 / tl.sqrt(tl.sum(v * v, -1) / C + eps)


@triton.jit
def _mix(SRC, U, Y, S, ACC, MX, SM, M, K, KS, NB, eps, C: tl.constexpr, NBP: tl.constexpr, HAS_S: tl.constexpr,
         BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr):
    """One pass over the K sources of the pointer table SRC for NB mixtures (queries U[0..NB-1]). Mixture 0, over all of
    them: Y = norm(h) and, if HAS_S (KS > 0: it reads kept sources), S = (logsumexp, rstd(h)) per token. Mixtures 1..NB-1: their online-softmax
    state over the first KS sources (ACC[b-1] = sum_k exp(s_bk - MX[b-1]) v_k in bf16; MX, SM: the running max and sum)."""
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
        s = libdevice.mul_rn(tl.sum(v[None, :, :] * u[:, None, :], 2), _rstd(v, eps, C)[None, :])  # (a rounded product, no FMA: the backward recomputes it bitwise)
        m_new = tl.maximum(m_i, s)
        a, b = tl.exp(m_i - m_new), tl.exp(s - m_new)
        upd = ((rb == 0) | (k < KS))[:, None]
        acc = tl.where(upd[:, :, None], acc * a[:, :, None] + b[:, :, None] * v[None, :, :], acc)
        l_i = tl.where(upd, l_i * a + b, l_i)
        m_i = tl.where(upd, m_new, m_i)
    first = (rb == 0)[:, None]
    m0, l0 = tl.sum(tl.where(first, m_i, 0.0), 0), tl.sum(tl.where(first, l_i, 0.0), 0)
    h = tl.sum(tl.where(first[:, :, None], acc, 0.0), 0) / l0[:, None]
    rstd = _rstd(h, eps, C)
    if HAS_S:
        tl.store(S + rm * 2, m0 + tl.log(l0), mask=mm)
        tl.store(S + rm * 2 + 1, rstd, mask=mm)
    tl.store(Y + rm[:, None] * C + rc[None, :], (h * rstd[:, None]).to(tl.bfloat16), mask=mask)
    rest = (rb >= 1) & mb
    ob = tl.maximum(rb - 1, 0).to(tl.int64)
    tl.store(MX + ob[:, None] * M + rm[None, :], m_i, mask=rest[:, None] & mm[None, :])
    tl.store(SM + ob[:, None] * M + rm[None, :], l_i, mask=rest[:, None] & mm[None, :])
    tl.store(ACC + (ob[:, None, None] * M + rm[None, :, None]) * C + rc[None, None, :], acc.to(tl.bfloat16), mask=rest[:, None, None] & mask[None, :, :])


@triton.jit
def _fwd(SRC, U, Y, S, ACC, MX, SM, M, K, eps, C: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr):
    """A mixture continued from its state (ACC, MX, SM: see _mix) over the K sources added since: Y and S as _mix's."""
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rc = tl.arange(0, BLOCK_C)
    mm, mc = rm < M, rc < C
    mask = mm[:, None] & mc[None, :]
    u = tl.load(U + rc, mask=mc, other=0.0)
    m_i = tl.load(MX + rm, mask=mm, other=0.0)
    l_i = tl.load(SM + rm, mask=mm, other=1.0)
    acc = tl.load(ACC + rm[:, None] * C + rc[None, :], mask=mask, other=0.0).to(tl.float32)
    for k in range(K):
        v = tl.load(tl.load(SRC + k).to(tl.pointer_type(tl.bfloat16)) + rm[:, None] * C + rc[None, :], mask=mask, other=0.0).to(tl.float32)
        s = libdevice.mul_rn(tl.sum(v * u[None, :], 1), _rstd(v, eps, C))
        m_new = tl.maximum(m_i, s)
        a, b = tl.exp(m_i - m_new), tl.exp(s - m_new)
        acc = acc * a[:, None] + b[:, None] * v
        l_i = l_i * a + b
        m_i = m_new
    h = acc / l_i[:, None]
    rstd = _rstd(h, eps, C)
    tl.store(S + rm * 2, m_i + tl.log(l_i), mask=mm)
    tl.store(S + rm * 2 + 1, rstd, mask=mm)
    tl.store(Y + rm[:, None] * C + rc[None, :], (h * rstd[:, None]).to(tl.bfloat16), mask=mask)


@triton.jit
def _bwd(DY, Y, S, D, U, G, P, DD, W, M, eps, C: tl.constexpr, HAS_D: tl.constexpr, HAS_G: tl.constexpr,
         BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr):
    """A mixture's backward from its output y: dh = rstd (dy - y (dy . y) / C); no kept source is read.
    HAS_G (it read kept sources): G = dh (bf16) and P = (logsumexp, G . h), for their backward (_bwd_src, _bwd_lead).
    HAS_D (its newest source D has no other reader): that source's gradient DD and, with kept sources, W = g r
    (du = W @ d). Without kept sources h = d: DD is the RMSNorm backward, and S (nothing was saved) is not read."""
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rc = tl.arange(0, BLOCK_C)
    mm, mc = rm < M, rc < C
    mask = mm[:, None] & mc[None, :]
    dy = tl.load(DY + rm[:, None] * C + rc[None, :], mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y + rm[:, None] * C + rc[None, :], mask=mask, other=0.0).to(tl.float32)
    if HAS_D:
        d = tl.load(D + rm[:, None] * C + rc[None, :], mask=mask, other=0.0).to(tl.float32)
    else:
        d = y
    u = tl.load(U + rc, mask=mc, other=0.0)
    r = _rstd(d, eps, C)
    uv = tl.sum(d * u[None, :], 1)
    if HAS_G:
        lse = tl.load(S + rm * 2, mask=mm, other=0.0)
        rstd = tl.load(S + rm * 2 + 1, mask=mm, other=1.0)
    else:  # h = d: its logit is the logsumexp, its rstd h's
        lse = libdevice.mul_rn(uv, r)
        rstd = r
    dh = (rstd[:, None] * (dy - y * (tl.sum(dy * y, 1) / C)[:, None])).to(tl.bfloat16)
    dhf = dh.to(tl.float32)
    if HAS_G:
        c = tl.sum(dhf * y, 1) / rstd
        tl.store(G + rm[:, None] * C + rc[None, :], dh, mask=mask)
        tl.store(P + rm * 2, lse, mask=mm)
        tl.store(P + rm * 2 + 1, c, mask=mm)
    else:
        c = tl.sum(dhf * d, 1)
    if HAS_D:  # (alone, alpha = 1 and g = 0 exactly: dd = dh)
        alpha = tl.exp(libdevice.mul_rn(uv, r) - lse)  # the forward's softmax weight
        g = alpha * (tl.sum(dhf * d, 1) - c)
        dd = alpha[:, None] * dhf + g[:, None] * (r[:, None] * u[None, :] - (r * r * r * uv / C)[:, None] * d)
        tl.store(DD + rm[:, None] * C + rc[None, :], dd.to(tl.bfloat16), mask=mask)
        if HAS_G:
            tl.store(W + rm, (g * r).to(tl.bfloat16), mask=mm)


@triton.jit
def _bwd_src(V, GT, PT, U, DV, W, PART, M, N, eps, C: tl.constexpr, HAS_P: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr):
    """dv of a kept source over N of its readers (their G, P through the tables GT, PT; their queries U), added to PART
    (its group's lead's share) if HAS_P; W[m] = g r per reader and token, for du_m = W_m @ v."""
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rc = tl.arange(0, BLOCK_C)
    mm, mc = rm < M, rc < C
    mask = mm[:, None] & mc[None, :]
    v = tl.load(V + rm[:, None] * C + rc[None, :], mask=mask, other=0.0).to(tl.float32)
    r = _rstd(v, eps, C)
    if HAS_P:
        acc = tl.load(PART + rm[:, None] * C + rc[None, :], mask=mask, other=0.0).to(tl.float32)
    else:
        acc = tl.zeros([BLOCK_M, BLOCK_C], tl.float32)
    for m in range(N):
        dh = tl.load(tl.load(GT + m).to(tl.pointer_type(tl.bfloat16)) + rm[:, None] * C + rc[None, :], mask=mask, other=0.0).to(tl.float32)
        pp = tl.load(PT + m).to(tl.pointer_type(tl.float32))
        lse, c = tl.load(pp + rm * 2, mask=mm, other=0.0), tl.load(pp + rm * 2 + 1, mask=mm, other=0.0)
        u = tl.load(U + m * C + rc, mask=mc, other=0.0)
        uv = tl.sum(v * u[None, :], 1)
        alpha = tl.exp(libdevice.mul_rn(uv, r) - lse)
        g = alpha * (tl.sum(dh * v, 1) - c)
        acc += alpha[:, None] * dh + g[:, None] * (r[:, None] * u[None, :] - (r * r * r * uv / C)[:, None] * v)
        tl.store(W + m * M + rm, (g * r).to(tl.bfloat16), mask=mm)
    tl.store(DV + rm[:, None] * C + rc[None, :], acc.to(tl.bfloat16), mask=mask)


@triton.jit
def _bwd_lead(VT, GT, PT, U, PART, W, M, N, NB, eps, C: tl.constexpr, NBP: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr):
    """_bwd_src for the NB kept sources of VT at once over N readers, each reader's G read once: PART[b] (bf16, source
    b's share of dv from these readers), W[b, m]."""
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rc = tl.arange(0, BLOCK_C)
    rb = tl.arange(0, NBP)
    mm, mc, mb = rm < M, rc < C, rb < NB
    mask = mm[:, None] & mc[None, :]
    vp = tl.load(VT + rb, mask=mb, other=0).to(tl.pointer_type(tl.bfloat16))
    v = tl.load(vp[:, None, None] + (rm[None, :, None] * C + rc[None, None, :]), mask=mb[:, None, None] & mask[None, :, :], other=0.0).to(tl.float32)
    r = _rstd(v, eps, C)
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
    tl.store(PART + (rb[:, None, None].to(tl.int64) * M + rm[None, :, None]) * C + rc[None, None, :], acc.to(tl.bfloat16), mask=mb[:, None, None] & mask[None, :, :])


def _grid(M, C, tile):
    bc = triton.next_power_of_2(C)
    bm = max(1, min(64, tile["elems"] // bc))
    return (triton.cdiv(M, bm),), dict(BLOCK_C=bc, BLOCK_M=bm, num_warps=tile["num_warps"])


def _table(ts):
    """A device table of data pointers, copied from pinned memory without blocking the host."""
    return torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64).pin_memory().to(ts[0].device, non_blocking=True)


# ---- custom ops: opaque to torch.compile, every tensor they touch is an argument ----

@torch.library.custom_op("attn_res::mix", mutates_args=(), device_types="cuda")
def _mix_op(vs: list[Tensor], U: Tensor, ks: int) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """vs: (M, C) bf16 sources; U: (nb, C) float32. y = norm(h) of the mixture of vs under U[0], S = its (logsumexp,
    rstd(h)) per token (float32; empty if ks = 0), and the state (acc bf16, mx, sm; nb - 1 rows) of the mixtures of
    vs[:ks], the kept sources, under U[1:]."""
    (M, C), nb, f32 = vs[0].shape, U.shape[0], torch.float32
    y, S = torch.empty_like(vs[0]), vs[0].new_empty(M * (ks > 0), 2, dtype=f32)
    acc, mx, sm = vs[0].new_empty(nb - 1, M, C), vs[0].new_empty(nb - 1, M, dtype=f32), vs[0].new_empty(nb - 1, M, dtype=f32)
    grid, cfg = _grid(M, C, TILE)
    _mix[grid](_table(vs), U, y, S, acc, mx, sm, M, len(vs), ks, nb, EPS, C=C, NBP=triton.next_power_of_2(nb), HAS_S=ks > 0, **cfg)
    return y, S, acc, mx, sm


@torch.library.custom_op("attn_res::mix_from", mutates_args=(), device_types="cuda")
def _mix_from_op(vs: list[Tensor], u: Tensor, acc: Tensor, mx: Tensor, sm: Tensor) -> tuple[Tensor, Tensor]:
    """y and S as mix's, for a mixture continued from its state (acc (M, C) bf16, mx, sm (M,)) over the sources vs added since."""
    M, C = acc.shape
    y, S = torch.empty_like(acc), mx.new_empty(M, 2)
    grid, cfg = _grid(M, C, TILE)
    _fwd[grid](_table(vs), u, y, S, acc, mx, sm, M, len(vs), EPS, C=C, **cfg)
    return y, S


@torch.library.custom_op("attn_res::bwd", mutates_args=(), device_types="cuda")
def _bwd_op(dy: Tensor, y: Tensor, S: Tensor, ds: list[Tensor], U: Tensor, i: int) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The backward of mixture i (see _bwd). S: empty if it read no kept source; ds: the source it alone read, or none.
    Returns G, P (for its kept sources; empty without), dd (that source's gradient; empty without) and that source's
    share of dU ((n, C) float32, row i)."""
    M, C = y.shape
    has_g, has_d = S.numel() > 0, len(ds) > 0
    G, P = y.new_empty(M * has_g, C), y.new_empty(M * has_g, 2, dtype=torch.float32)
    dd, W, dU = y.new_empty(M * has_d, C), y.new_empty(int(has_d and has_g), M), torch.zeros_like(U)
    grid, cfg = _grid(M, C, TILE)
    _bwd[grid](dy, y, S, (*ds, y)[0], U[i], G, P, dd, W, M, EPS, C=C, HAS_D=has_d, HAS_G=has_g, **cfg)
    dU[i:i + len(W)] = torch.mm(W, (*ds, y)[0], out_dtype=torch.float32)
    return G, P, dd, dU


@torch.library.custom_op("attn_res::bwd_src", mutates_args=(), device_types="cuda")
def _bwd_src_op(v: Tensor, Gs: list[Tensor], Ps: list[Tensor], U: Tensor, first: int, part: Tensor | None) -> tuple[Tensor, Tensor]:
    """dv of the kept source v over its readers, the mixtures first .. first + len(Gs) - 1 (their G, P; all the queries
    U), plus part (its lead's share) if given; and v's share of dU ((n, C) float32, those rows)."""
    M, C = v.shape
    dv, W, dU = torch.empty_like(v), v.new_empty(len(Gs), M), torch.zeros_like(U)
    grid, cfg = _grid(M, C, TILE)
    _bwd_src[grid](v, _table(Gs), _table(Ps), U[first:], dv, W, dv if part is None else part, M, len(Gs), EPS, C=C, HAS_P=part is not None, **cfg)
    dU[first:first + len(Gs)] = torch.mm(W, v, out_dtype=torch.float32)
    return dv, dU


@torch.library.custom_op("attn_res::bwd_lead", mutates_args=(), device_types="cuda")
def _bwd_lead_op(vs: list[Tensor], Gs: list[Tensor], Ps: list[Tensor], U: Tensor, first: int) -> tuple[Tensor, Tensor]:
    """Each kept source of vs's share of dv from the readers first .. first + len(Gs) - 1 (their G, P; all the queries
    U) ((len(vs), M, C) bf16), and their share of dU ((n, C) float32, those rows)."""
    M, C = vs[0].shape
    part, W, dU = vs[0].new_empty(len(vs), M, C), vs[0].new_empty(len(vs), len(Gs), M), torch.zeros_like(U)
    grid, cfg = _grid(M, C, LEAD_TILE)
    _bwd_lead[grid](_table(vs), _table(Gs), _table(Ps), U[first:], part, W, M, len(Gs), len(vs), EPS, C=C, NBP=triton.next_power_of_2(len(vs)), **cfg)
    for b in range(len(vs)):
        dU[first:first + len(Gs)] += torch.mm(W[b], vs[b], out_dtype=torch.float32)
    return part, dU


@torch.library.custom_op("attn_res::total", mutates_args=())
def _total_op(dUs: list[Tensor]) -> Tensor:
    """The sum of the nodes' shares of dU."""
    return torch.stack(dUs).sum(0)


def _f32(t, *shape):
    return t.new_empty(*shape, dtype=torch.float32)


_mix_op.register_fake(lambda vs, U, ks: (torch.empty_like(vs[0]), _f32(vs[0], vs[0].shape[0] * (ks > 0), 2), vs[0].new_empty(U.shape[0] - 1, *vs[0].shape),
                                         _f32(vs[0], U.shape[0] - 1, vs[0].shape[0]), _f32(vs[0], U.shape[0] - 1, vs[0].shape[0])))
_mix_from_op.register_fake(lambda vs, u, acc, mx, sm: (torch.empty_like(acc), mx.new_empty(mx.shape[0], 2)))
_bwd_op.register_fake(lambda dy, y, S, ds, U, i: (y.new_empty(S.shape[0], y.shape[1]), _f32(y, S.shape[0], 2), y.new_empty(y.shape[0] * len(ds), y.shape[1]),
                                                  torch.empty_like(U)))
_bwd_src_op.register_fake(lambda v, Gs, Ps, U, first, part: (torch.empty_like(v), torch.empty_like(U)))
_bwd_lead_op.register_fake(lambda vs, Gs, Ps, U, first: (vs[0].new_empty(len(vs), *vs[0].shape), torch.empty_like(U)))
_total_op.register_fake(lambda dUs: torch.empty_like(dUs[0]))


# ---- autograd. A kept source returns one zero-memory token pair per reader ((M, C) bf16 and (M, 2) f32 expanded zeros);
# a mixture takes its kept sources' tokens and returns its G and P as their gradients, so a kept source's backward runs
# once every reader's has, with all their G and P in hand. Nothing is mutated in the backward: AOTAutograd can trace it. ----

def _tokens(v, U, n):
    zb, zf = v.new_zeros(()), U.new_zeros(())
    M, C = v.shape
    return (*(zb.expand(M, C) for _ in range(n)), *(zf.expand(M, 2) for _ in range(n)))


def _live(grads, n):
    """The readers whose mixture ran (a contiguous range a..b): their G and P."""
    live = [m for m in range(n) if grads[m] is not None]
    if not live:
        return None
    a, b = live[0], live[-1] + 1
    assert b - a == len(live)
    return a, b, [grads[m].contiguous() for m in live], [grads[n + m].contiguous() for m in live]


class _Queries(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, n):
        """n zero-memory tokens shaped as the pseudo-queries U, one per node of a run: the gradient a node returns for its
        token is its share of dU, and the shares are summed here, in one operator. (Left to autograd the sum is a chain of
        small additions through the whole backward, around which torch.compile schedules the recomputation of the blocks:
        it then runs their weight gradients last and keeps every block's activations alive until then.)"""
        ctx.set_materialize_grads(False)
        z = U.new_zeros(())
        return tuple(z.expand(U.shape) for _ in range(n))

    @staticmethod
    def backward(ctx, *grads):
        return torch.ops.attn_res.total([g for g in grads if g is not None]), None


class _Source(torch.autograd.Function):
    @staticmethod
    def forward(ctx, v, U, q, first, n, member):
        """A kept source v ((M, C) bf16), read by the mixtures first .. first + n - 1 (up to its group's boundary, or all the
        rest). U: all the pseudo-queries (values); q: this node's token (see _Queries). Returns its n token pairs and, for a
        group member, one more token that the group's lead node takes, whose gradient is v's share of dv from the mixtures
        after the boundary."""
        ctx.save_for_backward(v, U)
        ctx.first, ctx.n, ctx.member = first, n, member
        ctx.set_materialize_grads(False)
        return _tokens(v, U, n) + (v.new_zeros(()).expand(v.shape),) * member

    @staticmethod
    def backward(ctx, *grads):
        v, U = ctx.saved_tensors
        part = grads[2 * ctx.n] if ctx.member else None
        live = _live(grads, ctx.n)
        if live is None:
            return part, None, None, None, None, None
        a, b, Gs, Ps = live
        dv, dU = torch.ops.attn_res.bwd_src(v, Gs, Ps, U, ctx.first + a, None if part is None else part.contiguous())
        return dv, None, dU, None, None, None


class _Lead(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, q, first, n_members, *args):
        """A group's stand-in in the mixtures from first on (U, q: as _Source's). args: the members' sources, then their
        tokens. Returns one token pair per mixture, as a kept source's."""
        ctx.save_for_backward(U, *args[:n_members])
        ctx.first, ctx.n_members = first, n_members
        ctx.set_materialize_grads(False)
        return _tokens(args[0], U, U.shape[0] - first)

    @staticmethod
    def backward(ctx, *grads):
        U, *vs = ctx.saved_tensors
        live = _live(grads, U.shape[0] - ctx.first)
        if live is None:
            return (None,) * (4 + 2 * ctx.n_members)
        a, b, Gs, Ps = live
        part, dU = torch.ops.attn_res.bwd_lead(vs, Gs, Ps, U, ctx.first + a)
        return None, dU, None, None, *((None,) * ctx.n_members), *part.unbind(0)


def _mix_ops(U, i, nb, acc, mx, sm, n_dir, vs):
    """y, S (and the next mixtures' state) of mixture i: one pass over its sources vs (the last n_dir not kept), also for
    the states of the nb - 1 next mixtures; or continued from its state (acc, mx, sm) over the sources vs added since."""
    if acc is None:
        return torch.ops.attn_res.mix(vs, U[i:i + nb], len(vs) - n_dir)
    return torch.ops.attn_res.mix_from(vs, U[i], acc, mx, sm)


class _Mix(torch.autograd.Function):
    @staticmethod
    def forward(ctx, U, q, i, nb, acc, mx, sm, n_src, n_dir, *args):
        """Mixture i (see _mix_ops; U, q: as _Source's). args: the n_src sources it reads now, the kept ones detached (their
        gradients go through their tokens) and the last n_dir (0 or 1) the newest source if this mixture alone reads it
        (differentiated here); then this mixture's tokens, one pair per kept source or lead node. Returns y and the state, if any."""
        y, S, *state = _mix_ops(U, i, nb, acc, mx, sm, n_dir, list(args[:n_src]))
        ctx.mark_non_differentiable(*state)
        ctx.save_for_backward(U, y, S, *args[n_src - n_dir:n_src])
        ctx.i, ctx.n = i, (n_src, n_dir, (len(args) - n_src) // 2)
        return (y, *state)

    @staticmethod
    def backward(ctx, dy, *unused):
        U, y, S, *ds = ctx.saved_tensors
        n_src, n_dir, n_tok = ctx.n
        G, P, dd, dU = torch.ops.attn_res.bwd(dy.contiguous(), y, S, ds, U, ctx.i)
        return (None, dU, *(None,) * 7, *(None,) * (n_src - n_dir), *(dd,) * n_dir, *(G,) * n_tok, *(P,) * n_tok)


class _Stream:
    """One run of a backbone through the kernels: the kept sources, which every later mixture reads, and the newest one."""

    def __init__(self, x, U, starts):
        self.U, self.starts, self.shape, self.i = U.detach(), starts, x.shape, 0  # U (n, C) float32: a query per mixture; i: mixtures so far
        self.grad = torch.is_grad_enabled()
        self.q = iter(_Queries.apply(U, 2 * len(U) + len(U) // BLOCK + 2) if self.grad else ())  # a token per node, for its share of dU
        self.kept, self.direct = [], []  # the kept sources ((M, C) bf16, detached); the newest source, if the next mixture alone reads it
        self.tg, self.ts = [[] for _ in U], [[] for _ in U]  # per mixture: its tokens, a pair per kept source or lead node
        self.group = {}  # a group's boundary (a mixture) -> its members: (source, token)
        self.first, self.seen, self.state = 0, 0, [U.new_empty(0)] * 3  # the last pass: its mixture, the kept sources it covered, the next mixtures' states
        self._new(x.reshape(-1, x.shape[-1]), starts[0])

    def _new(self, v, stays):
        """The newest source: kept, so every later mixture reads it too (stays), or read by the next one only (a sum that goes on)."""
        self.open = v = v.to(torch.bfloat16).contiguous()
        self.direct = [v] * (not stays)
        if stays:
            self._keep(v)

    def _keep(self, v):
        """A kept source: its tokens for the mixtures up to its group's boundary (from there on the group's lead node stands in)."""
        self.kept.append(v.detach())
        if self.grad:
            end = min((self.i // BLOCK + 1) * BLOCK, len(self.U))
            member = end < len(self.U)
            toks = _Source.apply(v, self.U, next(self.q), self.i, end - self.i, member)
            self.group.setdefault(end, []).extend([(self.kept[-1], toks[-1])] * member)
            self._hand(toks)

    def _hand(self, toks):
        """A node's token pairs, to the mixtures from the next one on."""
        n = len(toks) // 2
        for m in range(n):
            self.tg[self.i + m].append(toks[m])
            self.ts[self.i + m].append(toks[n + m])

    def add(self, v):
        """A block's output: a new source if the block starts one, else added to the running sum."""
        v, nxt = v.reshape(-1, v.shape[-1]), self.i + 1
        self._new(v if self.starts[self.i] else self.open + v, nxt < len(self.U) and self.starts[nxt])

    def read(self):
        """norm(h) of the next mixture: the kept sources and the newest one under the next pseudo-query."""
        i, n = self.i, len(self.U)
        members = self.group.pop(i, [])
        if members:  # a group's boundary: from this mixture on its lead node stands in for its sources
            self._hand(_Lead.apply(self.U, next(self.q), i, len(members), *(v for v, _ in members), *(t for _, t in members)))
        r = i - self.first - 1  # this mixture's row in the last pass's states, if it has one
        if 0 <= r < len(self.state[0]):  # continued: its state, then the sources added since
            nb, state, vs = 1, [s[r] for s in self.state], self.kept[self.seen:] + self.direct
        else:  # one pass over every source: this mixture, and the states of the next ones (none without kept sources: nothing to share)
            nb, state, vs = (min(BLOCK, n - i) if self.kept else 1), [None] * 3, self.kept + self.direct
        if self.grad:  # compiled, a checkpoint: norm(h) is rebuilt from the sources in the backward instead of saved
            args = (self.U, next(self.q), i, nb, *state, len(vs), len(self.direct), *vs, *self.tg[i], *self.ts[i])
            y, *state = torch.utils.checkpoint.checkpoint(_Mix.apply, *args, use_reentrant=False) if torch.compiler.is_compiling() else _Mix.apply(*args)
        else:
            y, _, *state = _mix_ops(self.U, i, nb, *state, len(self.direct), vs)
        if state:
            self.first, self.seen, self.state = i, len(self.kept), state
        self.i = i + 1
        return y.view(self.shape)


def reference_mix(vs, u):
    """norm(sum_i softmax_i(u . v_i / rms(v_i)) v_i) over the sources vs, in float32: the definition the kernels follow."""
    V = torch.stack([v.float() for v in vs])
    s = (V * u.float()).sum(-1) * (V.square().mean(-1) + EPS).rsqrt()
    h = (s.softmax(0).unsqueeze(-1) * V).sum(0)
    return F.rms_norm(h, h.shape[-1:], eps=EPS).to(vs[0].dtype)


class _ReferenceStream:
    """The same in plain PyTorch, for other devices and dtypes. Each mixture is a checkpoint: it keeps references to its
    sources and stacks them again in the backward."""

    def __init__(self, x, U, starts):
        self.U, self.starts, self.i, self.v = U, starts, 0, [x]

    def add(self, v):
        if self.starts[self.i]:
            self.v.append(v)
        else:
            self.v[-1] = self.v[-1] + v

    def read(self):
        u, self.i = self.U[self.i], self.i + 1
        if not torch.is_grad_enabled():
            return reference_mix(self.v, u)
        return torch.utils.checkpoint.checkpoint(reference_mix, list(self.v), u, use_reentrant=False)


class AttnResBackbone(nn.Module):
    """A residual backbone over blocks (an nn.ModuleList) whose running sum is ended before the blocks in starts (their
    indices; None: every block), see the module docstring. Block i reads the mixture of the kept sources and the current
    sum (block 0: the input itself), and the output is the mixture after the last block. One pseudo-query per mixture,
    zero-initialised (uniform mixtures); a mixture of a single source is its norm, whatever its query."""

    def __init__(self, blocks: nn.ModuleList, model_dim: int, starts=None):
        super().__init__()
        self.blocks = blocks
        self.starts = [starts is None or i in starts for i in range(len(blocks))]
        self.pseudo_queries = nn.Parameter(torch.zeros(len(blocks), model_dim))

    def forward(self, x, *args, **kwargs):
        stream = (_Stream if x.is_cuda and x.dtype == torch.bfloat16 else _ReferenceStream)(x, self.pseudo_queries.float(), self.starts)
        for block in self.blocks:
            stream.add(block(x, *args, **kwargs))
            x = stream.read()
        return x


class IdentityResidualBackbone(AttnResBackbone):
    """The backbone whose sum never ends: the pre-norm residual stream x = x + block(norm(x)), its output normed."""

    def __init__(self, blocks: nn.ModuleList, model_dim: int):
        super().__init__(blocks, model_dim, starts=())
