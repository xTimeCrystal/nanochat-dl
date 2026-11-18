"""Attention Residuals (Kimi Team 2026, github.com/MoonshotAI/Attention-Residuals) as a residual backbone: AttnResBackbone.
Triton kernels: the activations must be bfloat16 on CUDA."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
import triton
import triton.language as tl
from torch import Tensor
from triton.language.extra import libdevice

EPS = torch.finfo(torch.float32).eps
BLOCK = 4  # mixtures per forward pass over the kept sources; kept sources per backward group
TILE = dict(per_warp=1024, min_warps=1)  # row elements per warp, fewest warps: _mix, _fwd, _bwd, _bwd_src
LEAD_TILE = dict(per_warp=512, min_warps=2)  # the same for _bwd_lead


# ---- kernels. A program takes whole rows of (M, C) bf16 tensors. h = sum_k softmax_k(s_k) v_k with logits
# s_k = u . v_k / rms(v_k) (NORM_KEYS) or u . v_k; the output is h / rms(h) (NORM_OUT) or h. ----

@triton.jit
def _rstd(v, eps, C: tl.constexpr, on: tl.constexpr):
    """1 / rms(v) per row; 1 if the norm is off."""
    if on:
        r = 1.0 / tl.sqrt(tl.sum(v * v, -1) / C + eps)
    else:
        r = tl.full(v.shape[:-1], 1.0, tl.float32)
    return r


@triton.jit
def _mix(SRC, U, Y, S, ACC, MX, SM, M, K, KS, NB, eps, C: tl.constexpr, NBP: tl.constexpr, NORM_KEYS: tl.constexpr, NORM_OUT: tl.constexpr,
         BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr):
    """One pass over the K sources of the pointer table SRC, for NB mixtures (queries U[0..NB-1]).
    Mixture 0 reads all K: its output Y and S = (logsumexp, 1 / rms(h)) per token.
    Mixtures 1..NB-1 read the first KS, the kept ones: their state ACC (bf16 running sum), MX and SM (running max and sum)."""
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rc = tl.arange(0, BLOCK_C)
    rb = tl.arange(0, NBP)
    mm, mc, mb = rm < M, rc < C, rb < NB
    mask = mm[:, None] & mc[None, :]
    m_i = tl.full([NBP, BLOCK_M], float("-inf"), tl.float32)
    l_i = tl.zeros([NBP, BLOCK_M], tl.float32)
    acc = tl.zeros([NBP, BLOCK_M, BLOCK_C], tl.float32)
    for k in range(K):
        v = tl.load(tl.load(SRC + k).to(tl.pointer_type(tl.bfloat16)) + rm[:, None] * C + rc[None, :], mask=mask, other=0.0).to(tl.float32)
        uv = tl.zeros([NBP, BLOCK_M], tl.float32)
        for b in tl.static_range(NBP):  # one query at a time: an (NBP, C) product at once does not fit a thread's registers
            u = tl.load(U + b * C + rc, mask=mc & (b < NB), other=0.0)
            uv = tl.where((rb == b)[:, None], tl.sum(v * u[None, :], 1)[None, :], uv)
        s = libdevice.mul_rn(uv, _rstd(v, eps, C, NORM_KEYS)[None, :])  # a rounded product (no FMA): the backward recomputes it bitwise
        upd = ((rb == 0) | (k < KS))[:, None]  # mixtures 1.. skip the sources after the kept ones
        m_new = tl.where(upd, tl.maximum(m_i, s), m_i)
        a, b = tl.where(upd, tl.exp(m_i - m_new), 1.0), tl.where(upd, tl.exp(s - m_new), 0.0)
        acc = acc * a[:, :, None] + b[:, :, None] * v[None, :, :]
        l_i = l_i * a + b
        m_i = m_new
    first = (rb == 0)[:, None]
    m0, l0 = tl.sum(tl.where(first, m_i, 0.0), 0), tl.sum(tl.where(first, l_i, 0.0), 0)
    h = tl.sum(tl.where(first[:, :, None], acc, 0.0), 0) / l0[:, None]
    rstd = _rstd(h, eps, C, NORM_OUT)
    tl.store(S + rm * 2, m0 + tl.log(l0), mask=mm)
    tl.store(S + rm * 2 + 1, rstd, mask=mm)
    tl.store(Y + rm[:, None] * C + rc[None, :], (h * rstd[:, None]).to(tl.bfloat16), mask=mask)
    rest = (rb >= 1) & mb
    ob = tl.maximum(rb - 1, 0).to(tl.int64)
    tl.store(MX + ob[:, None] * M + rm[None, :], m_i, mask=rest[:, None] & mm[None, :])
    tl.store(SM + ob[:, None] * M + rm[None, :], l_i, mask=rest[:, None] & mm[None, :])
    tl.store(ACC + (ob[:, None, None] * M + rm[None, :, None]) * C + rc[None, None, :], acc.to(tl.bfloat16), mask=rest[:, None, None] & mask[None, :, :])


@triton.jit
def _fwd(SRC, U, Y, S, ACC, MX, SM, M, K, eps, C: tl.constexpr, NORM_KEYS: tl.constexpr, NORM_OUT: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr):
    """A mixture continued from its state (ACC, MX, SM) over the K sources added since: Y and S as in _mix."""
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
        s = libdevice.mul_rn(tl.sum(v * u[None, :], 1), _rstd(v, eps, C, NORM_KEYS))
        m_new = tl.maximum(m_i, s)
        a, b = tl.exp(m_i - m_new), tl.exp(s - m_new)
        acc = acc * a[:, None] + b[:, None] * v
        l_i = l_i * a + b
        m_i = m_new
    h = acc / l_i[:, None]
    rstd = _rstd(h, eps, C, NORM_OUT)
    tl.store(S + rm * 2, m_i + tl.log(l_i), mask=mm)
    tl.store(S + rm * 2 + 1, rstd, mask=mm)
    tl.store(Y + rm[:, None] * C + rc[None, :], (h * rstd[:, None]).to(tl.bfloat16), mask=mask)


@triton.jit
def _bwd(DY, Y, S, D, U, G, P, DD, W, M, eps, C: tl.constexpr, HAS_D: tl.constexpr, NORM_KEYS: tl.constexpr, NORM_OUT: tl.constexpr,
         BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr):
    """A mixture's backward from its output Y and that output's gradient DY: G = dL/dh (bf16) and P = (logsumexp, G . h)
    per token, which the backward of its kept sources reads. For a mixture of one source, G is that source's gradient.
    HAS_D: also DD, the gradient of its own source D (the running sum, read beside kept sources), and W, for dU = W @ D."""
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rc = tl.arange(0, BLOCK_C)
    mm, mc = rm < M, rc < C
    mask = mm[:, None] & mc[None, :]
    dy = tl.load(DY + rm[:, None] * C + rc[None, :], mask=mask, other=0.0).to(tl.float32)
    y = tl.load(Y + rm[:, None] * C + rc[None, :], mask=mask, other=0.0).to(tl.float32)
    lse = tl.load(S + rm * 2, mask=mm, other=0.0)
    rstd = tl.load(S + rm * 2 + 1, mask=mm, other=1.0)
    if NORM_OUT:
        dh = (rstd[:, None] * (dy - y * (tl.sum(dy * y, 1) / C)[:, None])).to(tl.bfloat16)
    else:
        dh = dy.to(tl.bfloat16)
    dhf = dh.to(tl.float32)
    c = tl.sum(dhf * y, 1) / rstd
    tl.store(G + rm[:, None] * C + rc[None, :], dh, mask=mask)
    tl.store(P + rm * 2, lse, mask=mm)
    tl.store(P + rm * 2 + 1, c, mask=mm)
    if HAS_D:
        d = tl.load(D + rm[:, None] * C + rc[None, :], mask=mask, other=0.0).to(tl.float32)
        u = tl.load(U + rc, mask=mc, other=0.0)
        r = _rstd(d, eps, C, NORM_KEYS)
        uv = tl.sum(d * u[None, :], 1)
        alpha = tl.exp(libdevice.mul_rn(uv, r) - lse)  # the forward's softmax weight
        g = alpha * (tl.sum(dhf * d, 1) - c)  # dL/dlogit
        if NORM_KEYS:  # the logit u . d / rms(d) also depends on d through rms(d)
            dd = alpha[:, None] * dhf + g[:, None] * (r[:, None] * u[None, :] - (r * r * r * uv / C)[:, None] * d)
        else:
            dd = alpha[:, None] * dhf + g[:, None] * u[None, :]
        tl.store(DD + rm[:, None] * C + rc[None, :], dd.to(tl.bfloat16), mask=mask)
        tl.store(W + rm, (g * r).to(tl.bfloat16), mask=mm)


@triton.jit
def _bwd_src(V, GT, PT, U, DV, W, PART, M, N, eps, C: tl.constexpr, HAS_P: tl.constexpr, NORM_KEYS: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr):
    """The gradient DV of a kept source V from N readers (their G and P in the pointer tables GT, PT; their queries U),
    plus PART (its lead's share) if HAS_P. W[m]: per reader, for dU[m] = W[m] @ V."""
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rc = tl.arange(0, BLOCK_C)
    mm, mc = rm < M, rc < C
    mask = mm[:, None] & mc[None, :]
    v = tl.load(V + rm[:, None] * C + rc[None, :], mask=mask, other=0.0).to(tl.float32)
    r = _rstd(v, eps, C, NORM_KEYS)
    if HAS_P:
        acc = tl.load(PART + rm[:, None] * C + rc[None, :], mask=mask, other=0.0).to(tl.float32)
    else:
        acc = tl.zeros([BLOCK_M, BLOCK_C], tl.float32)
    along = tl.zeros([BLOCK_M], tl.float32)  # with key norm, the logits also depend on v through rms(v): the coefficient of v, applied once
    for m in range(N):
        dh = tl.load(tl.load(GT + m).to(tl.pointer_type(tl.bfloat16)) + rm[:, None] * C + rc[None, :], mask=mask, other=0.0).to(tl.float32)
        pp = tl.load(PT + m).to(tl.pointer_type(tl.float32))
        lse, c = tl.load(pp + rm * 2, mask=mm, other=0.0), tl.load(pp + rm * 2 + 1, mask=mm, other=0.0)
        u = tl.load(U + m * C + rc, mask=mc, other=0.0)
        uv = tl.sum(v * u[None, :], 1)
        alpha = tl.exp(libdevice.mul_rn(uv, r) - lse)
        g = alpha * (tl.sum(dh * v, 1) - c)
        acc += alpha[:, None] * dh + (g * r)[:, None] * u[None, :]
        if NORM_KEYS:
            along += g * r * r * r * uv / C
        tl.store(W + m * M + rm, (g * r).to(tl.bfloat16), mask=mm)
    tl.store(DV + rm[:, None] * C + rc[None, :], (acc - along[:, None] * v).to(tl.bfloat16), mask=mask)


@triton.jit
def _bwd_lead(VT, GT, PT, U, PART, W, M, N, NB, eps, C: tl.constexpr, NBP: tl.constexpr, NORM_KEYS: tl.constexpr, BLOCK_C: tl.constexpr, BLOCK_M: tl.constexpr):
    """_bwd_src for the NB kept sources of the pointer table VT at once, over N readers, each reader's G read once:
    PART[b] (bf16) is source b's gradient from these readers, W[b, m] as _bwd_src's."""
    rm = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    rc = tl.arange(0, BLOCK_C)
    rb = tl.arange(0, NBP)
    mm, mc, mb = rm < M, rc < C, rb < NB
    mask = mm[:, None] & mc[None, :]
    vp = tl.load(VT + rb, mask=mb, other=0).to(tl.pointer_type(tl.bfloat16))
    v = tl.load(vp[:, None, None] + (rm[None, :, None] * C + rc[None, None, :]), mask=mb[:, None, None] & mask[None, :, :], other=0.0).to(tl.float32)
    r = _rstd(v, eps, C, NORM_KEYS)
    acc = tl.zeros([NBP, BLOCK_M, BLOCK_C], tl.float32)
    along = tl.zeros([NBP, BLOCK_M], tl.float32)
    gn = tl.load(tl.load(GT).to(tl.pointer_type(tl.bfloat16)) + rm[:, None] * C + rc[None, :], mask=mask & (N > 0), other=0.0)
    for m in range(N):
        dh = gn.to(tl.float32)
        gn = tl.load(tl.load(GT + tl.minimum(m + 1, N - 1)).to(tl.pointer_type(tl.bfloat16)) + rm[:, None] * C + rc[None, :],
                     mask=mask & (m + 1 < N), other=0.0)  # the next G, loaded while this one is used
        pp = tl.load(PT + m).to(tl.pointer_type(tl.float32))
        lse, c = tl.load(pp + rm * 2, mask=mm, other=0.0), tl.load(pp + rm * 2 + 1, mask=mm, other=0.0)
        u = tl.load(U + m * C + rc, mask=mc, other=0.0)
        uv = tl.sum(v * u[None, None, :], 2)
        alpha = tl.exp(libdevice.mul_rn(uv, r) - lse[None, :])
        g = alpha * (tl.sum(dh[None, :, :] * v, 2) - c[None, :])
        acc += alpha[:, :, None] * dh[None, :, :] + (g * r)[:, :, None] * u[None, None, :]
        if NORM_KEYS:
            along += g * r * r * r * uv / C
        tl.store(W + (rb[:, None].to(tl.int64) * N + m) * M + rm[None, :], (g * r).to(tl.bfloat16), mask=mb[:, None] & mm[None, :])
    tl.store(PART + (rb[:, None, None].to(tl.int64) * M + rm[None, :, None]) * C + rc[None, None, :], (acc - along[:, :, None] * v).to(tl.bfloat16),
             mask=mb[:, None, None] & mask[None, :, :])


def _grid(M, C, tile):
    """Whole rows per program (1024 elements, or one row if wider), with warps by the row's width."""
    bc = triton.next_power_of_2(C)
    bm = max(1, 1024 // bc)
    return (triton.cdiv(M, bm),), dict(BLOCK_C=bc, BLOCK_M=bm, num_warps=max(tile["min_warps"], bc // tile["per_warp"]))


def _table(ts):
    """A device table of the tensors' data pointers."""
    return torch.tensor([t.data_ptr() for t in ts], dtype=torch.int64).pin_memory().to(ts[0].device, non_blocking=True)


# ---- custom ops: opaque to torch.compile. Sources are (M, C) bf16; U is (n, C) float32, a query per mixture. ----

@torch.library.custom_op("attn_res::mix", mutates_args=(), device_types="cuda")
def _mix_op(vs: list[Tensor], U: Tensor, ks: int, norm_keys: bool, norm_out: bool) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """The mixture of vs under U[0]: its output y and S (see _mix); and the states (acc, mx, sm) of the mixtures of
    vs[:ks], the kept sources, under U[1:]."""
    (M, C), nb, f32 = vs[0].shape, U.shape[0], torch.float32
    y, S = torch.empty_like(vs[0]), vs[0].new_empty(M, 2, dtype=f32)
    acc, mx, sm = vs[0].new_empty(nb - 1, M, C), vs[0].new_empty(nb - 1, M, dtype=f32), vs[0].new_empty(nb - 1, M, dtype=f32)
    grid, cfg = _grid(M, C, TILE)
    _mix[grid](_table(vs), U, y, S, acc, mx, sm, M, len(vs), ks, nb, EPS, C=C, NBP=triton.next_power_of_2(nb), NORM_KEYS=norm_keys, NORM_OUT=norm_out, **cfg)
    return y, S, acc, mx, sm


@torch.library.custom_op("attn_res::mix_from", mutates_args=(), device_types="cuda")
def _mix_from_op(vs: list[Tensor], u: Tensor, acc: Tensor, mx: Tensor, sm: Tensor, norm_keys: bool, norm_out: bool) -> tuple[Tensor, Tensor]:
    """y and S of a mixture continued from its state (acc, mx, sm) over the sources vs added since."""
    M, C = acc.shape
    y, S = torch.empty_like(acc), mx.new_empty(M, 2)
    grid, cfg = _grid(M, C, TILE)
    _fwd[grid](_table(vs), u, y, S, acc, mx, sm, M, len(vs), EPS, C=C, NORM_KEYS=norm_keys, NORM_OUT=norm_out, **cfg)
    return y, S


@torch.library.custom_op("attn_res::bwd", mutates_args=(), device_types="cuda")
def _bwd_op(dy: Tensor, y: Tensor, S: Tensor, ds: list[Tensor], U: Tensor, i: int, norm_keys: bool, norm_out: bool) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """The backward of mixture i (see _bwd): G, P, and for its own source ds[0] (if given) its gradient dd and its
    share of dU ((n, C) float32, row i)."""
    M, C = y.shape
    G, P, dU = torch.empty_like(y), y.new_empty(M, 2, dtype=torch.float32), torch.zeros_like(U)
    if len(ds) == 1:
        d, dd, W = ds[0], torch.empty_like(y), y.new_empty(1, M)
    else:  # no own source: d is a placeholder that is not read, dd and W are empty
        d, dd, W = y, y.new_empty(0, C), y.new_empty(0, M)
    grid, cfg = _grid(M, C, TILE)
    _bwd[grid](dy, y, S, d, U[i], G, P, dd, W, M, EPS, C=C, HAS_D=len(ds) == 1, NORM_KEYS=norm_keys, NORM_OUT=norm_out, **cfg)
    dU[i:i + len(ds)] = torch.mm(W, d, out_dtype=torch.float32)
    return G, P, dd, dU


@torch.library.custom_op("attn_res::bwd_src", mutates_args=(), device_types="cuda")
def _bwd_src_op(v: Tensor, Gs: list[Tensor], Ps: list[Tensor], U: Tensor, first: int, part: Tensor | None, norm_keys: bool) -> tuple[Tensor, Tensor]:
    """The gradient dv of the kept source v from the mixtures first .. first + len(Gs) - 1 (their G and P), plus part
    (its lead's share) if given; and its share of dU ((n, C) float32, those rows)."""
    M, C = v.shape
    dv, W, dU = torch.empty_like(v), v.new_empty(len(Gs), M), torch.zeros_like(U)
    grid, cfg = _grid(M, C, TILE)
    _bwd_src[grid](v, _table(Gs), _table(Ps), U[first:], dv, W, dv if part is None else part, M, len(Gs), EPS, C=C, HAS_P=part is not None,
                   NORM_KEYS=norm_keys, **cfg)
    dU[first:first + len(Gs)] = torch.mm(W, v, out_dtype=torch.float32)
    return dv, dU


@torch.library.custom_op("attn_res::bwd_lead", mutates_args=(), device_types="cuda")
def _bwd_lead_op(vs: list[Tensor], Gs: list[Tensor], Ps: list[Tensor], U: Tensor, first: int, norm_keys: bool) -> tuple[Tensor, Tensor]:
    """For each kept source of vs, its gradient from the mixtures first .. first + len(Gs) - 1 ((len(vs), M, C) bf16);
    and their share of dU ((n, C) float32, those rows)."""
    M, C = vs[0].shape
    part, W, dU = vs[0].new_empty(len(vs), M, C), vs[0].new_empty(len(vs), len(Gs), M), torch.zeros_like(U)
    grid, cfg = _grid(M, C, LEAD_TILE)
    _bwd_lead[grid](_table(vs), _table(Gs), _table(Ps), U[first:], part, W, M, len(Gs), len(vs), EPS, C=C, NBP=triton.next_power_of_2(len(vs)),
                    NORM_KEYS=norm_keys, **cfg)
    for b in range(len(vs)):
        dU[first:first + len(Gs)] += torch.mm(W[b], vs[b], out_dtype=torch.float32)
    return part, dU


@torch.library.custom_op("attn_res::total", mutates_args=())
def _total_op(dUs: list[Tensor]) -> Tensor:
    """The sum of the nodes' shares of dU."""
    return torch.stack(dUs).sum(0)


def _f32(t, *shape):
    return t.new_empty(*shape, dtype=torch.float32)


_mix_op.register_fake(lambda vs, U, ks, norm_keys, norm_out: (torch.empty_like(vs[0]), _f32(vs[0], vs[0].shape[0], 2), vs[0].new_empty(U.shape[0] - 1, *vs[0].shape),
                                                              _f32(vs[0], U.shape[0] - 1, vs[0].shape[0]), _f32(vs[0], U.shape[0] - 1, vs[0].shape[0])))
_mix_from_op.register_fake(lambda vs, u, acc, mx, sm, norm_keys, norm_out: (torch.empty_like(acc), mx.new_empty(mx.shape[0], 2)))
_bwd_op.register_fake(lambda dy, y, S, ds, U, i, norm_keys, norm_out: (torch.empty_like(y), _f32(y, y.shape[0], 2), y.new_empty(y.shape[0] * len(ds), y.shape[1]),
                                                                        torch.empty_like(U)))
_bwd_src_op.register_fake(lambda v, Gs, Ps, U, first, part, norm_keys: (torch.empty_like(v), torch.empty_like(U)))
_bwd_lead_op.register_fake(lambda vs, Gs, Ps, U, first, norm_keys: (vs[0].new_empty(len(vs), *vs[0].shape), torch.empty_like(U)))
_total_op.register_fake(lambda dUs: torch.empty_like(dUs[0]))


# ---- autograd. A kept source hands every mixture that reads it a token pair (zero-memory expanded zeros, shaped as G
# and P). The mixture returns its G and P as the tokens' gradients, so the source's backward runs after all its readers',
# with their G and P in hand. Nothing is mutated in the backward: AOTAutograd can trace it. ----

def _tokens(v, U, n):
    zb, zf = v.new_zeros(()), U.new_zeros(())
    M, C = v.shape
    return (*(zb.expand(M, C) for _ in range(n)), *(zf.expand(M, 2) for _ in range(n)))


def _live(grads, n):
    """Of n readers' token gradients (the Gs, then the Ps), those of the mixtures that ran (a contiguous range):
    the first one's index, the Gs and the Ps. None if none ran."""
    live = [m for m in range(n) if grads[m] is not None]
    if not live:
        return None
    assert live[-1] - live[0] + 1 == len(live)
    return live[0], [grads[m].contiguous() for m in live], [grads[n + m].contiguous() for m in live]


class _Queries(torch.autograd.Function):
    """n tokens shaped as the pseudo-queries U, one per node of a run. A node returns its share of dU as its token's
    gradient and the shares are summed here in one op. (Summed by autograd, they form a chain of small additions through
    the whole backward; torch.compile then schedules the blocks' weight gradients last and keeps their activations alive.)"""

    @staticmethod
    def forward(ctx, U, n):
        ctx.set_materialize_grads(False)
        z = U.new_zeros(())
        return tuple(z.expand(U.shape) for _ in range(n))

    @staticmethod
    def backward(ctx, *grads):
        return torch.ops.attn_res.total([g for g in grads if g is not None]), None


class _Source(torch.autograd.Function):
    """A kept source v, read by the mixtures first .. first + n - 1 (those before its group's lead takes over).
    Returns their n token pairs and, if its group is led, one more token for the lead: its gradient is v's gradient
    from the later mixtures. (No token is made that nobody takes: torch.compile would hand back zeros for it.)"""

    @staticmethod
    def forward(ctx, v, U, q, first, n, led, norm_keys):
        ctx.save_for_backward(v, U)
        ctx.first, ctx.n, ctx.led, ctx.norm_keys = first, n, led, norm_keys
        ctx.set_materialize_grads(False)
        tokens = _tokens(v, U, n)
        if led:
            tokens += (v.new_zeros(()).expand(v.shape),)
        return tokens

    @staticmethod
    def backward(ctx, *grads):
        v, U = ctx.saved_tensors
        part = grads[-1] if ctx.led else None
        live = _live(grads, ctx.n)
        if live is None:  # no reader of its own ran: the lead's share is its whole gradient
            return part, None, None, None, None, None, None
        a, Gs, Ps = live
        dv, dU = torch.ops.attn_res.bwd_src(v, Gs, Ps, U, ctx.first + a, None if part is None else part.contiguous(), ctx.norm_keys)
        return dv, None, dU, None, None, None, None


class _Lead(torch.autograd.Function):
    """Stands in for a group of kept sources in the mixtures from first on. args: the members' sources, then their
    lead tokens. Returns one token pair per mixture."""

    @staticmethod
    def forward(ctx, U, q, first, n_members, norm_keys, *args):
        ctx.save_for_backward(U, *args[:n_members])
        ctx.first, ctx.n_members, ctx.norm_keys = first, n_members, norm_keys
        ctx.set_materialize_grads(False)
        return _tokens(args[0], U, U.shape[0] - first)

    @staticmethod
    def backward(ctx, *grads):
        U, *vs = ctx.saved_tensors
        live = _live(grads, U.shape[0] - ctx.first)
        if live is None:
            return (None,) * (5 + 2 * ctx.n_members)
        a, Gs, Ps = live
        part, dU = torch.ops.attn_res.bwd_lead(vs, Gs, Ps, U, ctx.first + a, ctx.norm_keys)
        return None, dU, None, None, None, *((None,) * ctx.n_members), *part.unbind(0)


class _Mix(torch.autograd.Function):
    """Mixture i. Without a state (acc, mx, sm): one pass over its sources, which also leaves the states of the next
    nb - 1 mixtures over the kept ones. With a state: continued from it over the sources added since.
    args: the n_src sources read now (the kept ones detached: their gradients go through their tokens; the last n_dir,
    0 or 1, is the running sum, differentiated here), then this mixture's tokens. Returns its output and the states."""

    @staticmethod
    def forward(ctx, U, q, i, nb, acc, mx, sm, n_src, n_dir, norm_keys, norm_out, *args):
        vs, states = list(args[:n_src]), []
        if acc is None:
            y, S, *states = torch.ops.attn_res.mix(vs, U[i:i + nb], n_src - n_dir, norm_keys, norm_out)
        else:
            y, S = torch.ops.attn_res.mix_from(vs, U[i], acc, mx, sm, norm_keys, norm_out)
        n_tok = (len(args) - n_src) // 2
        ctx.mark_non_differentiable(*states)
        # Without tokens there is no kept source: the running sum is the only source, its gradient is G, and it is not needed.
        ctx.save_for_backward(U, y, S, *(args[n_src - n_dir:n_src] if n_tok else ()))
        ctx.i, ctx.n, ctx.norms = i, (n_src, n_dir, n_tok), (norm_keys, norm_out)
        return (y, *states)

    @staticmethod
    def backward(ctx, dy, *unused):
        U, y, S, *ds = ctx.saved_tensors
        n_src, n_dir, n_tok = ctx.n
        G, P, dd, dU = torch.ops.attn_res.bwd(dy.contiguous(), y, S, ds, U, ctx.i, *ctx.norms)
        own = dd if n_tok else G  # the running sum's gradient
        return (None, dU, *(None,) * 9, *(None,) * (n_src - n_dir), *(own,) * n_dir, *(G,) * n_tok, *(P,) * n_tok)


class _Stream:
    """One run of a backbone. Sources: the kept ones, which every later mixture reads, and the running sum, which only
    the next mixture reads (until it is kept). Mixture i is the one block i reads; the last one is the output."""

    def __init__(self, x, U, starts, norm_input, norm_keys, norm_output):
        assert x.is_cuda and x.dtype == torch.bfloat16, "AttnResBackbone runs Triton kernels: its input must be bfloat16 on CUDA"
        n = len(U)
        self.U, self.shape, self.i = U.detach(), x.shape, 0  # U (n, C) float32: a query per mixture; i: mixtures read so far
        self.starts = [*starts, False]  # after the last block nothing starts: its sum is not kept
        self.norm_keys, self.norm_out = norm_keys, [norm_input] + [norm_output] * (n - 1)  # norm_out: per mixture
        self.grad = torch.is_grad_enabled()
        self.q = iter(_Queries.apply(U, 3 * n) if self.grad else ())  # a token per kept source, lead and mixture, for its share of dU
        self.kept = []  # the kept sources ((M, C) bf16, detached)
        self.g_tokens, self.p_tokens = [[] for _ in U], [[] for _ in U]  # per mixture: its tokens, a pair per kept source or lead
        self.ahead, self.seen = [], 0  # the states the last pass left for the next mixtures; the kept sources they cover
        # Backward groups: BLOCK consecutive kept sources. A full group is led if at least BLOCK mixtures read all its members:
        # from the first of them on (the group's boundary), one lead node stands in for the members.
        kept_at = [i for i in range(n) if self.starts[i]]  # a sum kept before block i is first read by mixture i
        self.ends = []  # per kept source: its group's boundary, or n if the group is not led
        for g in range(0, len(kept_at), BLOCK):
            group = kept_at[g:g + BLOCK]
            led = len(group) == BLOCK and group[-1] + BLOCK <= n
            self.ends += [group[-1] if led else n] * len(group)
        self.groups = {}  # boundary -> the (source, lead token) pairs of the group's members
        self._new(x.reshape(-1, x.shape[-1]))

    def _new(self, v):
        """The running sum after a block (first: the input). It is kept if the next block starts a new source."""
        self.sum = v = v.to(torch.bfloat16).contiguous()
        self.direct = [v]  # read by the next mixture only
        if self.starts[self.i]:
            self.direct = []
            self.kept.append(v.detach())
            if self.grad:  # its tokens: for the mixtures up to its group's boundary and, if the group is led, one for the lead
                first = self.i + 1 if self._alone() else self.i  # the first mixture that reads it through a kernel
                end = self.ends[len(self.kept) - 1]
                led = end < len(self.U)
                tokens = _Source.apply(v, self.U, next(self.q), first, end - first, led, self.norm_keys)
                if led:
                    self.groups.setdefault(end, []).append((self.kept[-1], tokens[-1]))
                self._hand(tokens[:2 * (end - first)], first)

    def _alone(self):
        """The next mixture has one source and no norm: it is that source, and no kernel runs."""
        return len(self.kept) + len(self.direct) == 1 and not self.norm_out[self.i]

    def _hand(self, tokens, first):
        """A node's token pairs go to the mixtures first, first + 1, ..."""
        n = len(tokens) // 2
        for m in range(n):
            self.g_tokens[first + m].append(tokens[m])
            self.p_tokens[first + m].append(tokens[n + m])

    def add(self, v):
        """The output of the block that read the last mixture: a new source if that block starts one, else added to the sum."""
        v = v.reshape(-1, v.shape[-1])
        self._new(v if self.starts[self.i - 1] else self.sum + v)

    def read(self):
        """The next mixture's output."""
        i = self.i
        if i in self.groups:  # a group's boundary: from this mixture on its lead stands in for its sources
            sources, tokens = zip(*self.groups.pop(i))
            self._hand(_Lead.apply(self.U, next(self.q), i, len(sources), self.norm_keys, *sources, *tokens), i)
        if self._alone():
            y = self.sum
        else:
            if self.ahead:  # continued from the state the last pass left for it, over the sources added since
                nb, state, vs = 1, self.ahead.pop(0), self.kept[self.seen:] + self.direct
            else:  # a pass over every source; it also leaves the states of the next mixtures over the kept ones
                nb, state, vs = (min(BLOCK, len(self.U) - i) if self.kept else 1), (None, None, None), self.kept + self.direct
            args = (self.U, next(self.q, None), i, nb, *state, len(vs), len(self.direct), self.norm_keys, self.norm_out[i],
                    *vs, *self.g_tokens[i], *self.p_tokens[i])
            # A checkpoint: the output is rebuilt from the sources in the backward instead of being saved.
            y, *states = torch.utils.checkpoint.checkpoint(_Mix.apply, *args, use_reentrant=False)
            if states:  # after a pass: one state per next mixture (row selects: the compiler saves unbind's rows)
                self.ahead, self.seen = [[s[r] for s in states] for r in range(len(states[0]))], len(self.kept)
        self.i = i + 1
        return y.view(self.shape)


class AttnResBackbone(nn.Module):
    """A residual backbone. Every block reads a mixture of sources, h = sum_k softmax_k(u . key_k) v_k, with its own
    pseudo-query u; a source is the input or a sum of block outputs.

    blocks       nn.ModuleList. block(x, *args, **kwargs) returns the block's output only (not x + output).
    model_dim    channels of the input and of every block's output.
    starts       indices of the blocks whose output starts a new source. The other blocks add to the running sum.
                 () (default): one sum, input included: a plain residual stream.
                 range(len(blocks)): every output is a source: full AttnRes.
                 range(0, len(blocks), S): Block AttnRes, S sublayers per block.
    norm_input   True: the first block reads norm(x). False: it reads x.
    norm_keys    True: key_k = v_k / rms(v_k). False: key_k = v_k.
    norm_output  True: every later block reads norm(h), and the backbone returns norm(h). False: h itself.

    norm(t) = t / rms(t) over the channels, without parameters. The input x is bfloat16 on CUDA (Triton kernels).
    pseudo_queries: (len(blocks), model_dim), zero-initialised; row i is the query of the mixture after block i.
    A mixture of a single source does not depend on its query.
    """

    def __init__(self, blocks: nn.ModuleList, model_dim: int, starts=(), norm_input: bool = True, norm_keys: bool = True, norm_output: bool = True):
        super().__init__()
        assert all(i in range(len(blocks)) for i in starts), "starts: indices of blocks"
        self.blocks = blocks
        self.starts = [i in starts for i in range(len(blocks))]
        self.norm_input, self.norm_keys, self.norm_output = norm_input, norm_keys, norm_output
        self.pseudo_queries = nn.Parameter(torch.zeros(len(blocks), model_dim))

    def forward(self, x, *args, **kwargs):
        U = F.pad(self.pseudo_queries.float(), (0, 0, 1, 0))  # a row for the first mixture, which has one source and no query
        stream = _Stream(x, U, self.starts, self.norm_input, self.norm_keys, self.norm_output)
        for block in self.blocks:
            stream.add(block(stream.read(), *args, **kwargs))
        return stream.read()
