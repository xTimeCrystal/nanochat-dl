"""Mixture of Experts in the style of Kimi K3 (github.com/MoonshotAI/Kimi-K3) as the MLP sublayer: MoE.
Its shared experts alone are a standard MLP; its routed experts run on Triton kernels (nanochat/moe_kernels.py)."""

import torch
import torch.nn as nn
import torch.nn.functional as F


def relu2(h):
    return F.relu(h).square()


ACTIVATIONS = {"relu2": relu2, "silu": F.silu}  # of an expert's hidden units: relu squared (nanochat's MLP) or silu


class Linear(nn.Linear):
    """nn.Linear that casts weights to match input dtype in forward.
    Replaces autocast: master weights stay fp32 for optimizer precision,
    but matmuls run in the activation dtype (typically bf16 from embeddings)."""
    def forward(self, x):
        return F.linear(x, self.weight.to(dtype=x.dtype))


class MoE(nn.Module):
    """An MLP sublayer, model_dim -> model_dim: y = shared(x) + routed(x).

    shared(x) = sum over the shared experts of expert(x)                    every token goes through them
    routed(x) = sum over the token's k routed experts of p_k expert_k(x)
    expert(x) = act(x @ up^T) @ down^T
    s = sigmoid(x @ router^T)                    router scores, one per routed expert
    the token's experts: top-k of s + bias       the bias only steers the selection; Quantile Balancing sets it in training
    p = s[experts] / sum(s[experts])             mixture weights

    model_dim                  width of x and y
    num_experts                routed experts (0: none)
    num_experts_per_token      k, the routed experts of a token, at most num_experts (0 without routed experts)
    moe_intermediate_size      hidden width of one routed expert (0 without routed experts)
    num_shared_experts         shared experts (0: none)
    shared_intermediate_size   hidden width of one shared expert (0 without shared experts)
    activation                 "relu2" (relu squared) or "silu"

    Without routed experts this is a standard MLP on any device and dtype: num_shared_experts = 1 and
    shared_intermediate_size = 4 * model_dim with "relu2" is nanochat's, c_fc and c_proj its two linear layers.
    With routed experts the input is bfloat16 on CUDA (Triton kernels), model_dim and moe_intermediate_size are multiples
    of 16, and every routed (token, expert) pair is computed: nothing is dropped. With k = num_experts every token goes
    through every routed expert, and one routed expert alone (p = 1) is the standard MLP on the kernels. In training, each
    forward also sets the bias for the next batch from this one's routing, and the kernels read their own bfloat16 copies
    of the routed weights, made again when the weights change: after an optimizer step or an in-place write, not after a
    write through .data.
    """

    def __init__(self, model_dim: int, num_experts: int, num_experts_per_token: int, moe_intermediate_size: int,
                 num_shared_experts: int, shared_intermediate_size: int, activation: str):
        super().__init__()
        D, E, K, H, S, W = model_dim, num_experts, num_experts_per_token, moe_intermediate_size, num_shared_experts, shared_intermediate_size
        assert E + S > 0, "MoE: no experts at all"
        assert activation in ACTIVATIONS, f"MoE: activation must be one of {list(ACTIVATIONS)}, not {activation!r}"
        self.model_dim, self.num_experts, self.num_experts_per_token, self.num_shared_experts, self.activation = D, E, K, S, activation
        self.act = ACTIVATIONS[activation]
        if S > 0:  # the shared experts' hidden units side by side: one MLP S * W wide
            assert W > 0, "MoE: with shared experts, shared_intermediate_size > 0"
            self.c_fc = Linear(D, S * W, bias=False)
            self.c_proj = Linear(S * W, D, bias=False)
        else:
            assert W == 0, "MoE: without shared experts, shared_intermediate_size is 0"
        if E > 0:
            assert 0 < K <= E, "MoE: with routed experts, 0 < num_experts_per_token <= num_experts"
            assert H > 0 and D % 16 == 0 and H % 16 == 0, \
                "MoE: with routed experts, model_dim and moe_intermediate_size must be multiples of 16 (at other widths the kernels' GEMMs run at half speed)"
            import nanochat.moe_kernels  # (Triton: needed only with routed experts)
            self.router = nn.Parameter(torch.empty(E, D))
            self.wu = nn.Parameter(torch.empty(E, H, D))
            self.wd = nn.Parameter(torch.empty(E, D, H))
            self.register_buffer("bias", torch.zeros(E))  # selection bias, set by Quantile Balancing
            self.register_buffer("load", torch.zeros(E), persistent=False)  # selections per expert since they were last read and zeroed
        else:
            assert K == 0 and H == 0, "MoE: without routed experts, num_experts_per_token and moe_intermediate_size are 0"

    def init_weights(self):
        """nanochat's MLP init for every expert: up-projections uniform with std 0.4 / sqrt(model_dim), down-projections zero.
        The router uniform with std 1 / sqrt(model_dim); the balancing bias and the load counts zero."""
        s = 3 ** 0.5 * self.model_dim ** -0.5  # (uniform on (-s, s) has std 1 / sqrt(model_dim))
        if self.num_shared_experts > 0:
            torch.nn.init.uniform_(self.c_fc.weight, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(self.c_proj.weight)
        if self.num_experts > 0:
            torch.nn.init.uniform_(self.router, -s, s)
            torch.nn.init.uniform_(self.wu, -s * 0.4, s * 0.4)
            torch.nn.init.zeros_(self.wd)
            self.bias.zero_()
            self.load.zero_()

    def num_active_params(self):
        """The expert parameters a token goes through: the shared experts and k of the routed ones (the router aside)."""
        n = 0
        if self.num_shared_experts > 0:
            n += self.c_fc.weight.numel() + self.c_proj.weight.numel()
        if self.num_experts > 0:
            n += (self.wu.numel() + self.wd.numel()) * self.num_experts_per_token // self.num_experts
        return n

    def shared(self, x):
        return self.c_proj(self.act(self.c_fc(x)))

    def routed(self, x):
        from nanochat.moe_kernels import margin_counts, quantile_balance, routed_experts
        assert x.is_cuda and x.dtype == torch.bfloat16, "MoE runs its routed experts on Triton kernels: its input must be bfloat16 on CUDA"
        shape, k = x.shape, self.num_experts_per_token
        x = x.reshape(-1, shape[-1])
        s = torch.sigmoid(F.linear(x, self.router.to(x.dtype)).float())
        top = (s + self.bias).topk(min(k + 1, self.num_experts), dim=1)  # (the k selected and, if there is one, the next)
        sel = top.indices[:, :k]
        if self.training:  # balance for the next batch on this one's margins
            self.load.scatter_add_(0, sel.reshape(-1), torch.ones(sel.numel(), device=sel.device))
            # The cutoff: midway between the token's k-th and (k+1)-th largest biased scores (the k-th alone when every
            # expert is selected). With few experts per token the two are far apart, and at the (k+1)-th an expert the
            # token did not select looks closer to getting in than it is: loads then stay 10-30 % off an even split,
            # against a few % at the midpoint.
            cutoff = top.values[:, k - 1:].mean(1)
            self.bias.copy_(quantile_balance(margin_counts(s.detach(), cutoff.detach(), self.bias), self.bias, k))
        g = s.gather(1, sel)
        # In training the op takes the weights as they are and keeps its own copies in x's dtype from one optimizer step to
        # the next (moe_kernels._kernel_weights). In eval they are cast here on every call, as a Linear's are.
        wu, wd = (self.wu, self.wd) if self.training else (self.wu.to(x.dtype), self.wd.to(x.dtype))
        return routed_experts(x, wu, wd, sel, g / g.sum(1, keepdim=True), self.activation).view(shape)

    def forward(self, x):
        if self.num_experts == 0:
            return self.shared(x)
        if self.num_shared_experts == 0:
            return self.routed(x)
        return self.routed(x) + self.shared(x)
