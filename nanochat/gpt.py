"""
GPT model (rewrite, a lot simpler)
Notable features:
- rotary embeddings (and no positional embeddings)
- QK norm
- untied weights for token embedding and lm_head
- relu^2 activation in MLP
- norm after token embedding
- no learnable params in rmsnorm
- no bias in linear layers
- Group-Query Attention (GQA) support for more efficient inference
- Flash Attention 3 integration
"""

from functools import partial
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

from nanochat.common import get_dist_info, print0, COMPUTE_DTYPE
from nanochat.optim import MuonAdamW

# Our custom Flash Attention module that automatically uses FA3 when compatible and SDPA fallback otherwise
from nanochat.flash_attention import flash_attn

@dataclass
class GPTConfig:
    sequence_len: int = 2048
    vocab_size: int = 32768
    n_layer: int = 12
    n_head: int = 6 # number of query heads
    n_kv_head: int = 6 # number of key/value heads (GQA)
    n_embd: int = 768
    # Sliding window attention pattern string, tiled across layers. Final layer always L.
    # Characters: L=long (full context), S=short (quarter context)
    # Examples: "L"=all full context, "SL"=alternating, "SSL"=two short then one long
    window_pattern: str = "SSSL"
    # Attention Residuals (nanochat/attn_res.py): every attention and MLP reads a softmax-over-depth mixture of the
    # embedding and all earlier sublayer outputs instead of a residual stream (no resid/x0 lambdas, no backout)
    attn_res: bool = False
    # The MLP is a Mixture of Experts (Kimi K3 style): a shared part, which every token goes through, plus routed experts
    # (nanochat/moe.py), of which a token goes through num_experts_per_token. shared_intermediate_size is the shared part's
    # whole hidden width (0 = none), moe_intermediate_size one routed expert's. The dense MLP: shared 4 * n_embd, no experts.
    shared_intermediate_size: int = 3072 # (the dense MLP at the default n_embd; set it together with n_embd)
    num_experts: int = 0
    num_experts_per_token: int = 0
    moe_intermediate_size: int = 0
    activation: str = "relu2" # of every expert's hidden units: "relu2" (relu squared) or "silu"


def norm(x):
    return F.rms_norm(x, (x.size(-1),)) # note that this will run in bf16, seems ok

class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


def has_ve(layer_idx, n_layer):
    """Returns True if GPT layer should have Value Embedding (alternating, last layer always included)."""
    return layer_idx % 2 == (n_layer - 1) % 2

def apply_rotary_emb(x, cos, sin):
    # note: this rotates by -theta, the transpose of the textbook convention. Functionally
    # equivalent (only the relative q/k rotation matters), kept for checkpoint compatibility.
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:] # split up last dim into two halves
    y1 = x1 * cos + x2 * sin # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    return torch.cat([y1, y2], 3)

class CausalSelfAttention(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.layer_idx = layer_idx
        self.n_head = config.n_head
        self.n_kv_head = config.n_kv_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        assert self.n_kv_head <= self.n_head and self.n_head % self.n_kv_head == 0
        self.c_q = Linear(self.n_embd, self.n_head * self.head_dim, bias=False)
        self.c_k = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_v = Linear(self.n_embd, self.n_kv_head * self.head_dim, bias=False)
        self.c_proj = Linear(self.n_embd, self.n_embd, bias=False)
        self.ve_gate_channels = 12
        self.ve_gate = Linear(self.ve_gate_channels, self.n_kv_head, bias=False) if has_ve(layer_idx, config.n_layer) else None

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        # Shape: (B, T, H, D) - FA3's native layout, no transpose needed!
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_kv_head, self.head_dim)

        # Value residual (ResFormer): mix in value embedding with input-dependent gate per head
        if ve is not None:
            ve = ve.view(B, T, self.n_kv_head, self.head_dim)
            gate = 3 * torch.sigmoid(self.ve_gate(x[..., :self.ve_gate_channels]))  # (B, T, n_kv_head), range (0, 3)
            v = v + gate.unsqueeze(-1) * ve

        # Apply Rotary Embeddings to queries and keys to get relative positional encoding
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = norm(q), norm(k) # QK norm
        q = q * 1.2  # sharper attention (split scale between Q and K), TODO think through better
        k = k * 1.2

        # Flash Attention (FA3 or SDPA fallback)
        # window_size is (left, right) tuple: (N, 0) for causal, (-1, 0) for full context
        if kv_cache is None:
            # Training: causal attention with optional sliding window
            y = flash_attn.flash_attn_func(q, k, v, causal=True, window_size=window_size)
        else:
            # Inference: use flash_attn_with_kvcache which handles cache management
            k_cache, v_cache = kv_cache.get_layer_cache(self.layer_idx)
            y = flash_attn.flash_attn_with_kvcache(
                q, k_cache, v_cache,
                k=k, v=v,
                cache_seqlens=kv_cache.cache_seqlens,
                causal=True,
                window_size=window_size,
            )
            # Advance position after last layer processes
            if self.layer_idx == kv_cache.n_layers - 1:
                kv_cache.advance(T)

        # Re-assemble the heads and project back to residual stream
        y = y.contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    """y = shared(x) + routed(x), every expert x -> act(x @ fc^T) @ proj^T.
    shared(x): the sum of the shared experts, for every token. Side by side they are one MLP, c_fc and c_proj.
    routed(x): the sum over the token's num_experts_per_token routed experts of p_k expert_k(x) (nanochat/moe.py)."""
    def __init__(self, config):
        super().__init__()
        self.num_experts = config.num_experts
        self.num_experts_per_token = config.num_experts_per_token
        self.activation = config.activation
        assert self.activation in ("relu2", "silu")
        hidden = config.shared_intermediate_size
        assert hidden > 0 or self.num_experts > 0 # (the dense MLP: shared_intermediate_size=4 * n_embd)
        self.c_fc = Linear(config.n_embd, hidden, bias=False)
        self.c_proj = Linear(hidden, config.n_embd, bias=False)
        if self.num_experts > 0: # routed experts: a router, and stacks (num_experts, out, in) of their c_fc and c_proj
            assert 0 < self.num_experts_per_token <= self.num_experts and config.moe_intermediate_size > 0
            # (top-1 of several experts: p = s / s = 1, so the router would get no gradient)
            assert self.num_experts_per_token > 1 or self.num_experts == 1
            assert config.n_embd % 16 == 0 and config.moe_intermediate_size % 16 == 0 # (else the kernels run at half speed)
            self.router = Linear(config.n_embd, self.num_experts, bias=False)
            self.e_fc = nn.Parameter(torch.empty(self.num_experts, config.moe_intermediate_size, config.n_embd))
            self.e_proj = nn.Parameter(torch.empty(self.num_experts, config.n_embd, config.moe_intermediate_size))
            self.register_buffer("bias", torch.zeros(self.num_experts)) # selection bias, set by Quantile Balancing
            self.register_buffer("load", torch.zeros(self.num_experts), persistent=False) # selections per expert since they were last read
            from nanochat.moe import BINS # (Triton kernels: imported only in this mode)
            self.register_buffer("margins", torch.zeros(self.num_experts, BINS), persistent=False) # routing margins since the last balance

    def forward(self, x):
        y = self.c_fc(x)
        y = F.silu(y) if self.activation == "silu" else F.relu(y).square()
        y = self.c_proj(y)
        if self.num_experts > 0:
            from nanochat.moe import routed # (Triton kernels: imported only in this mode)
            y = y + routed(self, x)
        return y

    def balance(self):
        # Quantile Balancing: this layer's routing bias from the margins its training forwards recorded since the last
        # call (an all-reduce: every rank calls it). The bias stays fixed between calls. The dense MLP has none.
        if self.num_experts > 0:
            from nanochat.moe import quantile_balance
            self.bias.copy_(quantile_balance(self.margins, self.bias, self.num_experts_per_token))
            self.margins.zero_()


class Block(nn.Module):
    def __init__(self, config, layer_idx):
        super().__init__()
        self.attn = CausalSelfAttention(config, layer_idx)
        self.mlp = MLP(config)

    def forward(self, x, ve, cos_sin, window_size, kv_cache):
        x = x + self.attn(norm(x), ve, cos_sin, window_size, kv_cache)
        x = x + self.mlp(norm(x))
        return x


class AttnResAttention(CausalSelfAttention):
    """A layer's attention as an AttnRes block: reads the mixture (already normed), returns its output."""
    def __init__(self, config, layer_idx, window_size):
        super().__init__(config, layer_idx)
        self.window_size = window_size

    def forward(self, x, ves, cos_sin, kv_cache):
        return super().forward(x, ves.get(self.layer_idx), cos_sin, self.window_size, kv_cache)


class AttnResMLP(MLP):
    """A layer's MLP as an AttnRes block."""
    def forward(self, x, ves, cos_sin, kv_cache):
        return super().forward(x)


class GPT(nn.Module):
    def __init__(self, config, pad_vocab_size_to=64):
        """
        NOTE a major footgun: this __init__ function runs in meta device context (!!)
        Therefore, any calculations inside here are shapes and dtypes only, no actual data.
        => We actually initialize all data (parameters, buffers, etc.) in init_weights() instead.
        """
        super().__init__()
        self.config = config
        # Compute per-layer window sizes for sliding window attention
        # window_size is (left, right) tuple: (-1, 0) for full context, (N, 0) for sliding window
        self.window_sizes = self._compute_window_sizes(config)
        # Pad vocab for efficiency (DDP, tensor cores). This is just an optimization - outputs are cropped in forward().
        # https://huggingface.co/docs/transformers/main_classes/model#transformers.PreTrainedModel.resize_token_embeddings
        padded_vocab_size = ((config.vocab_size + pad_vocab_size_to - 1) // pad_vocab_size_to) * pad_vocab_size_to
        if padded_vocab_size != config.vocab_size:
            print0(f"Padding vocab_size from {config.vocab_size} to {padded_vocab_size} for efficiency")
        if config.attn_res: # every sublayer is a block of the backbone, the embedding its input
            from nanochat.attn_res import AttnResBackbone # (Triton kernels: imported only in this mode)
            h = AttnResBackbone(nn.ModuleList([m for i in range(config.n_layer) for m in (AttnResAttention(config, i, self.window_sizes[i]), AttnResMLP(config))]),
                                config.n_embd, starts=range(2 * config.n_layer), norm_input=True, norm_keys=True, norm_output=True)
        else:
            h = nn.ModuleList([Block(config, layer_idx) for layer_idx in range(config.n_layer)])
        self.transformer = nn.ModuleDict({
            "wte": nn.Embedding(padded_vocab_size, config.n_embd),
            "h": h,
        })
        self.lm_head = Linear(config.n_embd, padded_vocab_size, bias=False)
        if not config.attn_res:
            # Per-layer learnable scalars (inspired by modded-nanogpt)
            # resid_lambdas: scales the residual stream at each layer (init 1.0 = neutral)
            # x0_lambdas: blends initial embedding back in at each layer (init 0.0 = disabled)
            # Separate parameters so they can have different optimizer treatment
            self.resid_lambdas = nn.Parameter(torch.ones(config.n_layer))   # fake init, real init in init_weights()
            self.x0_lambdas = nn.Parameter(torch.zeros(config.n_layer))     # fake init, real init in init_weights()
            # Backout: subtract cached mid-layer residual before final norm to remove low-level features
            self.backout_lambda = nn.Parameter(0.2 * torch.ones(1))
        # Smear: mix previous token's embedding into current token (cheap bigram-like info)
        self.smear_gate = Linear(24, 1, bias=False)
        self.smear_lambda = nn.Parameter(torch.zeros(1))
        # Value embeddings (ResFormer-style): alternating layers, last layer always included
        head_dim = config.n_embd // config.n_head
        kv_dim = config.n_kv_head * head_dim
        self.value_embeds = nn.ModuleDict({str(i): nn.Embedding(padded_vocab_size, kv_dim) for i in range(config.n_layer) if has_ve(i, config.n_layer)})
        # To support meta device initialization, we init the rotary embeddings here, but it's just "fake" meta tensors only.
        # As for rotary_seq_len, these rotary embeddings are pretty small/cheap in memory,
        # so let's just over-compute them by 10X, but assert fail if we ever reach that amount.
        # In the future we can dynamically grow the cache, for now it's fine.
        self.rotary_seq_len = config.sequence_len * 10 # 10X over-compute should be enough, TODO make nicer?
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False) # persistent=False means it's not saved to the checkpoint
        self.register_buffer("sin", sin, persistent=False)

    @torch.no_grad()
    def init_weights(self):
        """
        Initialize the full model in this one function for maximum clarity.

        wte (embedding):     normal, std=1.0
        lm_head:             normal, std=0.001
        for each block:
            attn.c_q:        uniform, std=1/sqrt(n_embd)
            attn.c_k:        uniform, std=1/sqrt(n_embd)
            attn.c_v:        uniform, std=1/sqrt(n_embd)
            attn.c_proj:     zeros
            mlp.c_fc:        uniform, std=1/sqrt(n_embd)
            mlp.c_proj:      zeros
        """

        # Embedding and unembedding
        torch.nn.init.normal_(self.transformer.wte.weight, mean=0.0, std=0.8)
        torch.nn.init.normal_(self.lm_head.weight, mean=0.0, std=0.001)

        # Transformer blocks: uniform init with bound = sqrt(3) * std (same standard deviation as normal)
        n_embd = self.config.n_embd
        s = 3**0.5 * n_embd**-0.5 # sqrt(3) multiplier makes sure Uniform achieves the same std as Normal
        for attn, mlp in self.layers():
            torch.nn.init.uniform_(attn.c_q.weight, -s, s) # weights use Uniform to avoid outliers
            torch.nn.init.uniform_(attn.c_k.weight, -s, s)
            torch.nn.init.uniform_(attn.c_v.weight, -s, s)
            torch.nn.init.zeros_(attn.c_proj.weight) # projections are zero
            torch.nn.init.uniform_(mlp.c_fc.weight, -s * 0.4, s * 0.4)  # 0.4x init scale for c_fc
            torch.nn.init.zeros_(mlp.c_proj.weight)
            if self.config.num_experts > 0: # routed experts like c_fc and c_proj, their router like c_q
                torch.nn.init.uniform_(mlp.router.weight, -s, s)
                torch.nn.init.uniform_(mlp.e_fc, -s * 0.4, s * 0.4)
                torch.nn.init.zeros_(mlp.e_proj)
                torch.nn.init.zeros_(mlp.bias)
                torch.nn.init.zeros_(mlp.load)
                torch.nn.init.zeros_(mlp.margins)

        # Per-layer scalars
        n_layer = self.config.n_layer
        if self.config.attn_res:
            torch.nn.init.zeros_(self.transformer.h.pseudo_queries) # uniform mixtures at the start
        else:
            # Per-layer resid init: stronger residual at early layers, weaker at deep layers
            for i in range(n_layer):
                self.resid_lambdas.data[i] = 1.15 - (0.10 * i / max(n_layer - 1, 1))
            # Decaying x0 init: earlier layers get more input embedding blending
            for i in range(n_layer):
                self.x0_lambdas.data[i] = 0.20 - (0.15 * i / max(n_layer - 1, 1))
            torch.nn.init.constant_(self.backout_lambda, 0.2)

        # Smear scalar and gate must be explicitly initialized
        torch.nn.init.zeros_(self.smear_lambda)
        torch.nn.init.uniform_(self.smear_gate.weight, 0.0, 0.02)

        # Value embeddings (init like c_v: uniform with same std)
        for ve in self.value_embeds.values():
            torch.nn.init.uniform_(ve.weight, -s, s)

        # Gate weights init with small positive values so gates start slightly above neutral
        for attn, _ in self.layers():
            if attn.ve_gate is not None:
                torch.nn.init.uniform_(attn.ve_gate.weight, 0.0, 0.02)

        # Rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

        # Cast embeddings to COMPUTE_DTYPE: optimizer can tolerate reduced-precision
        # embeddings and it saves memory. Exception: fp16 requires fp32 embeddings
        # because GradScaler cannot unscale fp16 gradients.
        if COMPUTE_DTYPE != torch.float16:
            self.transformer.wte.to(dtype=COMPUTE_DTYPE)
            for ve in self.value_embeds.values():
                ve.to(dtype=COMPUTE_DTYPE)

    def layers(self):
        """(attention, MLP) of every layer."""
        if self.config.attn_res:
            b = self.transformer.h.blocks
            return [(b[2 * i], b[2 * i + 1]) for i in range(self.config.n_layer)]
        return [(block.attn, block.mlp) for block in self.transformer.h]

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=100000, device=None):
        # TODO: bump base theta more? e.g. 100K is more common more recently
        # autodetect the device from model embeddings
        if device is None:
            device = self.transformer.wte.weight.device
        # stride the channels
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        # stride the time steps
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        # calculate the rotation frequencies at each (time, channel) pair
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = cos.to(COMPUTE_DTYPE), sin.to(COMPUTE_DTYPE)
        cos, sin = cos[None, :, None, :], sin[None, :, None, :] # add batch and head dims for later broadcasting
        return cos, sin

    def _compute_window_sizes(self, config):
        """
        Compute per-layer window sizes for sliding window attention.

        Returns list of (left, right) tuples for FA3's window_size parameter:
        - left: how many tokens before current position to attend to (-1 = unlimited)
        - right: how many tokens after current position to attend to (0 for causal)

        Pattern string is tiled across layers. Final layer always gets L (full context).
        Characters: L=long (full context), S=short (quarter context)
        """
        pattern = config.window_pattern.upper()
        assert all(c in "SL" for c in pattern), f"Invalid window_pattern: {pattern}. Use only S and L."
        # Map characters to window sizes
        long_window = config.sequence_len
        short_window = -(-long_window // 4 // 128) * 128  # ceil to FA3 tile size (2048 -> 768)
        char_to_window = {
            "L": (long_window, 0),
            "S": (short_window, 0),
        }
        # Tile pattern across layers
        window_sizes = []
        for layer_idx in range(config.n_layer):
            char = pattern[layer_idx % len(pattern)]
            window_sizes.append(char_to_window[char])
        # Final layer always gets full context
        window_sizes[-1] = (long_window, 0)
        return window_sizes

    def get_device(self):
        return self.transformer.wte.weight.device

    def estimate_flops(self):
        """
        Return the estimated FLOPs per token for the model (forward + backward).
        Each matmul weight parameter contributes 2 FLOPs (multiply *, accumulate +) in forward, and 2X that in backward => 2+4=6.
        Cleanest explanation of this: https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4
        On top of that, 12 * h * q * effective_seq_len accounts for key @ query matmul flops inside attention.
        With sliding windows, effective_seq_len varies per layer (capped by window size).
        Ref: https://arxiv.org/abs/2204.02311 (PaLM paper).
        This is ~1% off from the exact formulas of Chinchilla paper, the difference is:
        - Chinchilla counts the embedding layer as flops (? weird, it's just a lookup => we ignore)
        - Chinchilla counts exp/sum/divide in attention softmax as flops (a little sus and very tiny => we ignore)
        """
        h, q, t = self.config.n_head, self.config.n_embd // self.config.n_head, self.config.sequence_len
        # Sum attention FLOPs per layer, accounting for sliding window
        attn_flops = 0
        for window_size in self.window_sizes:
            window = window_size[0]  # (left, right) tuple, we use left
            effective_seq = t if window < 0 else min(window, t)
            attn_flops += 12 * h * q * effective_seq
        num_flops_per_token = 6 * self.num_matmul_params() + attn_flops
        return num_flops_per_token

    def num_matmul_params(self):
        """
        The number of parameters that participate in matmuls with the token stream,
        i.e. contribute 2 FLOPs/param to the forward pass. Counted structurally: every
        matmul in this model goes through the Linear class, while non-matmul params
        (embeddings = lookups, per-layer scalars) are nn.Embedding or raw Parameters.
        """
        matmul_params = sum(m.weight.numel() for m in self.modules() if isinstance(m, Linear))
        c = self.config # routed experts are not Linear modules: a token goes through num_experts_per_token per layer
        matmul_params += c.n_layer * c.num_experts_per_token * 2 * c.moe_intermediate_size * c.n_embd
        return matmul_params

    def pop_moe_stats(self):
        # the tokens routed to every expert since the last call, (layers, experts), summed over micro-batches and
        # ranks (an all-reduce: every rank calls it), then reset
        mlps = [mlp for _, mlp in self.layers()]
        counts = torch.stack([mlp.load for mlp in mlps])
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(counts)
        for mlp in mlps:
            mlp.load.zero_()
        return counts

    def balance_routers(self):
        # Quantile Balancing of every layer (MLP.balance): after a micro-batch's forward, on every rank
        for _, mlp in self.layers():
            mlp.balance()

    def estimate_decode_flops(self, context_len):
        """
        Forward FLOPs to decode one token at a given context length during inference:
        2 FLOPs per matmul param, plus attention over min(context, window) per layer.
        """
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = sum(4 * h * q * min(context_len, window) for window, _ in self.window_sizes)
        decode_flops = 2 * self.num_matmul_params() + attn_flops
        return decode_flops

    def estimate_prefill_flops(self, num_tokens):
        """Forward FLOPs to prefill a prompt: causal, so token t attends to min(t, window)."""
        h = self.config.n_head
        q = self.config.n_embd // self.config.n_head
        attn_flops = 0
        for window, _ in self.window_sizes:
            w = min(window, num_tokens)
            attended_tokens = w * (w + 1) // 2 + (num_tokens - w) * w # ramp up to w, then flat
            attn_flops += 4 * h * q * attended_tokens
        prefill_flops = 2 * self.num_matmul_params() * num_tokens + attn_flops
        return prefill_flops

    def kv_bytes_per_token(self):
        """Bytes to *store* one token of KV cache during inference, per row (all layers)."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize # the KV cache is kept in the compute dtype
        return self.config.n_layer * 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes

    def kv_read_bytes(self, context_len):
        """Bytes of KV cache *read* by one decode step at a given context length, per row.
        Sliding window layers only attend to (and read) the last `window` tokens."""
        head_dim = self.config.n_embd // self.config.n_head
        kv_dtype_bytes = COMPUTE_DTYPE.itemsize
        total = 0
        for window, _ in self.window_sizes:
            total += 2 * self.config.n_kv_head * head_dim * kv_dtype_bytes * min(context_len, window)
        return total

    def num_scaling_params(self):
        """
        Return detailed parameter counts for scaling law analysis.
        Different papers use different conventions:
        - Kaplan et al. excluded embedding parameters
        - Chinchilla included all parameters
        Ref: https://arxiv.org/abs/2203.15556 (Chinchilla paper)
        Ref: https://arxiv.org/abs/2001.08361 (Kaplan et al. original scaling laws paper)

        Returns a dict with counts for each parameter group, so downstream analysis
        can experiment with which combination gives the cleanest scaling laws.
        """
        # Count each group separately (mirrors the grouping in setup_optimizers)
        wte = sum(p.numel() for p in self.transformer.wte.parameters())
        value_embeds = sum(p.numel() for p in self.value_embeds.parameters())
        lm_head = sum(p.numel() for p in self.lm_head.parameters())
        transformer_matrices = sum(p.numel() for p in self.transformer.h.parameters()) - (self.transformer.h.pseudo_queries.numel() if self.config.attn_res else 0)
        scalars = sum(p.numel() for p in self.scalar_params())
        total = wte + value_embeds + lm_head + transformer_matrices + scalars
        c = self.config # the matrices a token goes through: all but the routers and the routed experts it is not sent to
        idle = c.n_layer * c.n_embd * (c.num_experts + 2 * (c.num_experts - c.num_experts_per_token) * c.moe_intermediate_size)
        assert total == sum(p.numel() for p in self.parameters()), "Parameter count mismatch"
        return {
            'wte': wte,
            'value_embeds': value_embeds,
            'lm_head': lm_head,
            'transformer_matrices': transformer_matrices,
            'active_matrices': transformer_matrices - idle,
            'scalars': scalars,
            'total': total,
        }

    def scalar_params(self):
        """The small parameters outside the matrices and embeddings."""
        if self.config.attn_res:
            return [self.transformer.h.pseudo_queries, self.smear_gate.weight, self.smear_lambda]
        return [self.resid_lambdas, self.x0_lambdas, self.smear_gate.weight, self.smear_lambda, self.backout_lambda]

    def setup_optimizer(self, unembedding_lr=0.004, embedding_lr=0.2, matrix_lr=0.02, weight_decay=0.0, scalar_lr=0.5, attn_res_lr=1e-3):
        model_dim = self.config.n_embd

        # Separate out all parameters into groups
        attn_res = self.config.attn_res
        # (a router of routed experts is a matrix too: under AdamW its scores saturate; without shared experts c_fc and c_proj are empty)
        matrix_params = [p for p in self.transformer.h.parameters() if p.numel() > 0 and (not attn_res or p is not self.transformer.h.pseudo_queries)]
        value_embeds_params = list(self.value_embeds.parameters())
        embedding_params = list(self.transformer.wte.parameters())
        lm_head_params = list(self.lm_head.parameters())
        resid_params = [] if attn_res else [self.resid_lambdas]
        x0_params = [] if attn_res else [self.x0_lambdas]
        smear_params = [self.smear_gate.weight, self.smear_lambda] + ([] if attn_res else [self.backout_lambda])
        attn_res_params = [self.transformer.h.pseudo_queries] if attn_res else []
        assert sum(p.numel() > 0 for p in self.parameters()) == len(matrix_params) + len(embedding_params) + len(lm_head_params) + len(value_embeds_params) + len(resid_params) + len(x0_params) + len(smear_params) + len(attn_res_params)

        # Scale the LR for the AdamW parameters by ∝1/√dmodel (tuned for 768 dim model)
        dmodel_lr_scale = (model_dim / 768) ** -0.5
        print0(f"Scaling the LR for the AdamW parameters ∝1/√({model_dim}/768) = {dmodel_lr_scale:.6f}")

        # Build param_groups with all required fields explicit
        param_groups = [
            # AdamW groups (embeddings, lm_head, scalars)
            dict(kind='adamw', name='lm_head', params=lm_head_params, lr=unembedding_lr * dmodel_lr_scale, betas=(0.8, 0.96), eps=1e-10, weight_decay=0.01),
            dict(kind='adamw', name='embedding', params=embedding_params, lr=embedding_lr * dmodel_lr_scale, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.001),
            dict(kind='adamw', name='value_embeds', params=value_embeds_params, lr=embedding_lr * dmodel_lr_scale * 0.5, betas=(0.8, 0.995), eps=1e-10, weight_decay=0.01),
        ]
        if not attn_res: # (the dense groups in their original order: optimizer checkpoints load by group position)
            param_groups += [
                dict(kind='adamw', name='resid', params=resid_params, lr=scalar_lr * 0.01, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.05),
                dict(kind='adamw', name='x0', params=x0_params, lr=scalar_lr, betas=(0.96, 0.95), eps=1e-10, weight_decay=0.0),  # higher beta1 for x0
            ]
        param_groups.append(dict(kind='adamw', name='smear', params=smear_params, lr=0.2, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
        if attn_res: # pseudo-queries: Adam at attn_res_lr as given (no dmodel scaling), no weight decay
            param_groups.append(dict(kind='adamw', name='attn_res', params=attn_res_params, lr=attn_res_lr, betas=(0.8, 0.95), eps=1e-10, weight_decay=0.0))
        # Muon groups (matrix params, grouped by shape for stacking; an expert stack (E, out, in) is E matrices)
        for shape in sorted({p.shape for p in matrix_params}):
            group_params = [p for p in matrix_params if p.shape == shape]
            param_groups.append(dict(
                kind='muon', name='matrix', params=group_params, lr=matrix_lr,
                momentum=0.95, ns_steps=5, beta2=0.9, weight_decay=weight_decay,
            ))

        optimizer = MuonAdamW(param_groups)
        for group in optimizer.param_groups:
            group["initial_lr"] = group["lr"]
        return optimizer

    def forward(self, idx, targets=None, kv_cache=None, loss_reduction='mean', logit_positions=None):
        # logit_positions: compute the logits (and the loss) only at these sequence positions, e.g. a slice or an index tensor.
        # The logits are B x T x vocab in fp32, so an eval that scores a few tokens per row can skip the rest. None = all.
        B, T = idx.size()

        # Grab the rotary embeddings for the current sequence length (they are of shape (1, seq_len, 1, head_dim/2))
        assert T <= self.cos.size(1), f"Sequence length grew beyond the rotary embeddings cache: {T} > {self.cos.size(1)}"
        assert idx.device == self.cos.device, f"Rotary embeddings and idx are on different devices: {idx.device} != {self.cos.device}"
        assert self.cos.dtype == COMPUTE_DTYPE, f"Rotary embeddings must be in {COMPUTE_DTYPE}, got {self.cos.dtype}"
        # if kv cache exists, we need to offset the rotary embeddings to the current position in the cache
        T0 = 0 if kv_cache is None else kv_cache.get_pos()
        cos_sin = self.cos[:, T0:T0+T], self.sin[:, T0:T0+T] # truncate cache to current sequence length

        # Embed the tokens
        x = self.transformer.wte(idx) # embed current token
        x = x.to(COMPUTE_DTYPE) # ensure activations are in compute dtype (no-op usually, but active for fp16 code path)
        x = norm(x)

        # Smear: mix previous token's embedding into current position (cheap bigram info)
        if kv_cache is None:
            # Training / naive generate: full sequence available, use fast slice
            assert T > 1, "Training forward pass should have T > 1"
            gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
            x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
        else:
            # KV cache inference: read prev embedding from cache, store current for next step
            x_pre_smear = kv_cache.prev_embedding
            kv_cache.prev_embedding = x[:, -1:, :]
            if T > 1:
                # Prefill: apply smear to positions 1+, same as training
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, 1:, :24]))
                x = torch.cat([x[:, :1], x[:, 1:] + gate * x[:, :-1]], dim=1)
            elif x_pre_smear is not None:
                # Decode: single token, use cached prev embedding
                gate = self.smear_lambda.to(x.dtype) * torch.sigmoid(self.smear_gate(x[:, :, :24]))
                x = x + gate * x_pre_smear

        # Forward the trunk of the Transformer
        if self.config.attn_res:
            ves = {int(i): ve(idx).to(x.dtype) for i, ve in self.value_embeds.items()}
            x = self.transformer.h(x, ves, cos_sin, kv_cache) # (every sublayer reads a normed mixture; the output is normed)
        else:
            x0 = x  # save initial normalized embedding for x0 residual
            n_layer = self.config.n_layer
            backout_layer = n_layer // 2  # cache at halfway point
            x_backout = None
            for i, block in enumerate(self.transformer.h):
                x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
                ve = self.value_embeds[str(i)](idx).to(x.dtype) if str(i) in self.value_embeds else None
                x = block(x, ve, cos_sin, self.window_sizes[i], kv_cache)
                if i == backout_layer:
                    x_backout = x
            # Subtract mid-layer residual to remove low-level features before logit projection
            if x_backout is not None:
                x = x - self.backout_lambda.to(x.dtype) * x_backout
            x = norm(x)
        if logit_positions is not None:
            x = x[:, logit_positions]
            targets = None if targets is None else targets[:, logit_positions]

        # Forward the lm_head (compute logits)
        softcap = 15 # smoothly cap the logits to the range [-softcap, softcap]
        logits = self.lm_head(x) # (B, T, padded_vocab_size) <- very big tensor, large amount of memory
        logits = logits[..., :self.config.vocab_size] # slice to remove padding
        logits = logits.float() # switch to fp32 for logit softcap and loss computation
        logits = softcap * torch.tanh(logits / softcap) # squash the logits

        if targets is not None:
            # training: given the targets, compute and return the loss
            # TODO experiment with chunked cross-entropy?
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1), ignore_index=-1, reduction=loss_reduction)
            return loss
        else:
            # inference: just return the logits directly
            return logits

    @torch.inference_mode()
    def generate(self, tokens, max_tokens, temperature=1.0, top_k=None, seed=42):
        """
        Naive autoregressive streaming inference.
        To make it super simple, let's assume:
        - batch size is 1
        - ids and the yielded tokens are simple Python lists and ints
        """
        assert isinstance(tokens, list)
        device = self.get_device()
        rng = None
        if temperature > 0:
            rng = torch.Generator(device=device)
            rng.manual_seed(seed)
        ids = torch.tensor([tokens], dtype=torch.long, device=device) # add batch dim
        for _ in range(max_tokens):
            logits = self.forward(ids) # (B, T, vocab_size)
            logits = logits[:, -1, :] # (B, vocab_size)
            if top_k is not None and top_k > 0:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            if temperature > 0:
                logits = logits / temperature
                probs = F.softmax(logits, dim=-1)
                next_ids = torch.multinomial(probs, num_samples=1, generator=rng)
            else:
                next_ids = torch.argmax(logits, dim=-1, keepdim=True)
            ids = torch.cat((ids, next_ids), dim=1)
            token = next_ids.item()
            yield token
