"""
FBT and thinking tokens (GPTConfig.fbt_passes, think_*): plain FBT against dl_pro's formula, the thinking passes against a
position-by-position decoder of the stream recurrence, causality, and a training step.

Run: python -m pytest tests/test_thinking.py -v
"""
import pytest
import torch

import nanochat.gpt as gpt
from nanochat.gpt import GPT, GPTConfig, norm

V = 96


@pytest.fixture(autouse=True)
def float32(monkeypatch):
    """fp32 throughout (CPU defaults to bf16), so the exactness checks compare the recurrence, not rounding."""
    monkeypatch.setattr(gpt, "COMPUTE_DTYPE", torch.float32)


def make(**kw):
    """A tiny random model with every path live (the zero-init projections and the fed-back state made nonzero)."""
    c = dict(sequence_len=64, vocab_size=V, n_layer=2, n_head=2, n_kv_head=2, n_embd=32, window_pattern="SL",
             fbt_passes=4, think_latents=2)
    c.update(kw)
    torch.manual_seed(0)
    m = GPT(GPTConfig(**c))
    m.init_weights()
    with torch.no_grad():
        for block in m.transformer.h:
            block.attn.c_proj.weight.normal_(0, 0.2)
            block.mlp.c_proj.weight.normal_(0, 0.2)
        m.lm_head.weight.normal_(0, 0.2)
        m.smear_lambda.fill_(0.5)
        m.fbt_h.weight.normal_(0, 0.05)  # (spectral norm ~0.6: a contractive chain, so rounding does not grow along it)
    return m.eval()


def tokens(B=2, T=6, seed=1):
    g = torch.Generator().manual_seed(seed)
    return torch.randint(0, V, (B, T), generator=g)


def embed(m, idx):
    """The real tokens' normed, smeared embedding, as GPT.forward computes it."""
    x = norm(m.transformer.wte(idx))
    gate = m.smear_lambda * torch.sigmoid(m.smear_gate(x[:, 1:, :24]))
    return torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)


@pytest.mark.parametrize("training", [False, True])
def test_fbt_is_dl_pro_fbt(training):
    """No latents: every pass on norm(W_t e + W_h norm(h_{t-1})), h = 0 in pass 1 and at t = 0; the mean of the passes'
    losses in training, the last pass otherwise."""
    m = make(think_latents=0).train(training)
    idx, y = tokens(), tokens(seed=2)
    T = idx.shape[1]
    x, h, losses = embed(m, idx), None, []
    for _ in range(m.config.fbt_passes):
        h_prev = torch.zeros_like(x) if h is None else torch.cat([torch.zeros_like(h[:, :1]), h[:, :-1]], dim=1)
        h = m._trunk(norm(m.fbt_t(x) + m.fbt_h(norm(h_prev))), idx, (m.cos[:, :T], m.sin[:, :T]), None, m.window_sizes)
        losses.append(m._head(h, y, "mean"))
    want = torch.stack(losses).mean() if training else losses[-1]
    torch.testing.assert_close(m(idx, y), want, rtol=1e-6, atol=1e-6)


@torch.no_grad()
def sequential(m, idx):
    """Inference's exact recurrence, one stream position at a time: position i is fed norm(W_t e_i + W_h norm(h_p)),
    h_p the final state of the position before it (zero for token 0), and its state is the trunk over the inputs so far."""
    B, T = idx.shape
    L = m.config.think_latents
    owner, kind = torch.arange(T).repeat_interleave(L + 1), torch.arange(L + 1).repeat(T)
    x, x_think = embed(m, idx), norm(m.transformer.wte.weight[V])
    ids = torch.where(kind == 0, idx[:, owner], V)
    windows = [(w * (L + 1), r) for w, r in m.window_sizes]
    inputs, h_fin = [], torch.zeros(B, owner.numel(), m.config.n_embd)
    for i in range(owner.numel()):
        o, k = int(owner[i]), int(kind[i])
        e = x[:, o] if k == 0 else x_think.expand(B, -1)
        p = i - 1 if k > 0 else (o * (L + 1) - 1 if o > 0 else None)  # (a real token's: the previous token's last latent)
        hp = torch.zeros(B, m.config.n_embd) if p is None else h_fin[:, p]
        inputs.append(norm(m.fbt_t(e) + m.fbt_h(norm(hp))))
        ow = owner[:i + 1]
        h_fin[:, i] = m._trunk(torch.stack(inputs, 1), ids[:, :i + 1], (m.cos[:, ow], m.sin[:, ow]), None, windows)[:, i]
    return m._head(h_fin[:, L::L + 1], None, "mean")


@pytest.mark.parametrize("schedule", ["grown", "stream"])
@torch.no_grad()
def test_passes_reach_the_sequential_recurrence(schedule):
    """With as many passes as stream positions the passes are exact (position i is exact after i + 1 of them, by
    causality): the evaluation equals the position-by-position decoder."""
    T, L = 5, 2
    m = make(think_schedule=schedule, think_eval_passes=T * (L + 1))
    idx = tokens(T=T)
    torch.testing.assert_close(m(idx), sequential(m, idx), rtol=1e-4, atol=1e-4)


@pytest.mark.parametrize("schedule", ["grown", "stream"])
def test_training_passes(schedule):
    """The training loss against the passes written out position by position: pass s holds d_s latents per token
    (grown: min(s - 1, L); stream: L), each position fed the previous pass's state of the position before it (in that
    pass's layout), every position scored on its token's next real token; the mean over the passes."""
    m = make(think_schedule=schedule).train()
    idx, y = tokens(), tokens(seed=2)
    (B, T), L, C = idx.shape, m.config.think_latents, m.config.n_embd
    x, x_think = embed(m, idx), norm(m.transformer.wte.weight[V])
    prev, losses = None, []
    for s in range(1, m.config.fbt_passes + 1):
        d = min(s - 1, L) if schedule == "grown" else L
        pos = [(t, k) for t in range(T) for k in range(d + 1)]
        e = torch.stack([x[:, t] if k == 0 else x_think.expand(B, -1) for t, k in pos], 1)
        hp = torch.zeros(B, len(pos), C)
        for i, (t, k) in enumerate(pos):
            if prev is not None and (k > 0 or t > 0):
                h_prev, d_prev = prev
                hp[:, i] = h_prev[:, t * (d_prev + 1) + k - 1 if k > 0 else (t - 1) * (d_prev + 1) + d_prev]
        owner = torch.tensor([t for t, _ in pos])
        ids = torch.stack([idx[:, t] if k == 0 else torch.full((B,), V) for t, k in pos], 1)
        h = m._trunk(norm(m.fbt_t(e) + m.fbt_h(norm(hp))), ids, (m.cos[:, owner], m.sin[:, owner]), None,
                     [(w * (d + 1), r) for w, r in m.window_sizes])
        losses.append(m._head(h, y[:, owner], "mean"))
        prev = (h, d)
    torch.testing.assert_close(m(idx, y), torch.stack(losses).mean(), rtol=1e-6, atol=1e-6)


@pytest.mark.parametrize("schedule", ["grown", "stream"])
@torch.no_grad()
def test_no_later_token_reads(schedule):
    """Changing token j leaves the predictions at every earlier position unchanged."""
    m = make(think_schedule=schedule, think_eval_passes=6)
    idx = tokens()
    idx2 = idx.clone()
    idx2[:, 3] = (idx2[:, 3] + 1) % V
    a, b = m(idx), m(idx2)
    torch.testing.assert_close(a[:, :3], b[:, :3], rtol=0, atol=1e-6)
    assert not torch.allclose(a[:, 3:], b[:, 3:])


@pytest.mark.parametrize("schedule", ["grown", "stream"])
def test_training_step(schedule):
    """A finite training loss, gradients reaching the fed-back state and the latents' token, every parameter grouped."""
    m = make(think_schedule=schedule).train()
    loss = m(tokens(), tokens(seed=2))
    loss.backward()
    assert torch.isfinite(loss)
    assert m.fbt_h.weight.grad.abs().sum() > 0
    assert m.transformer.wte.weight.grad[V].abs().sum() > 0
    m.setup_optimizer()


def test_random_thought_counts():
    """think_random (stream): each token's count from 0..L (a shuffled balanced multiset, one draw for both passes); the
    training loss == the passes written out position by position over that irregular stream, the position before a latent
    the previous one of its group, before a real token the previous token's last position; evaluation uses L everywhere."""
    m = make(think_schedule="stream", think_random=True, fbt_passes=2, think_latents=3).train()
    idx, y = tokens(T=8), tokens(T=8, seed=2)
    (B, T), L, C = idx.shape, 3, m.config.n_embd
    torch.manual_seed(5)
    loss = m(idx, y)
    torch.manual_seed(5)
    cnt = (torch.arange(T) % (L + 1))[torch.randperm(T)].tolist()
    assert sorted(cnt) == sorted(i % (L + 1) for i in range(T))
    pos = [(t, k) for t in range(T) for k in range(cnt[t] + 1)]
    where = {p: i for i, p in enumerate(pos)}
    x, x_think = embed(m, idx), norm(m.transformer.wte.weight[V])
    e = torch.stack([x[:, t] if k == 0 else x_think.expand(B, -1) for t, k in pos], 1)
    owner = torch.tensor([t for t, _ in pos])
    ids = torch.stack([idx[:, t] if k == 0 else torch.full((B,), V) for t, k in pos], 1)
    windows = [(w * len(pos) // T, r) for w, r in m.window_sizes]
    h, losses = None, []
    for s in range(2):
        hp = torch.zeros(B, len(pos), C)
        for i, (t, k) in enumerate(pos):
            if h is not None and (k > 0 or t > 0):
                hp[:, i] = h[:, where[(t, k - 1)] if k > 0 else where[(t - 1, cnt[t - 1])]]
        h = m._trunk(norm(m.fbt_t(e) + m.fbt_h(norm(hp))), ids, (m.cos[:, owner], m.sin[:, owner]), None, windows)
        losses.append(m._head(h, y[:, owner], "mean"))
    torch.testing.assert_close(loss, torch.stack(losses).mean(), rtol=1e-6, atol=1e-6)
    m.eval()  # (evaluation: L latents after every token, i.e. the uniform stream)
    u = make(think_schedule="stream", fbt_passes=2, think_latents=3)
    u.load_state_dict(m.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(m(idx), u(idx))
