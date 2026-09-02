"""The velocity field, an action expert with one token per horizon position.

Shapes:

    condition:    (B, C)      normalized observation features
    action_chunk: (B, H, A)   the flow state at time tau
    tau:          (B, 1)      one flow time per chunk, or
                  (B, H, 1)   one flow time per horizon position
    returns:      (B, H, A)   velocity, same coordinates as action_chunk

Both time regimes go through one encoder and one set of parameters: the time
encoder maps the last axis, so a (B, H, 1) profile produces a per-token context
of width d_model where a (B, 1) scalar produces a per-chunk one, and the
block's modulation broadcasts accordingly. Supporting mixed time therefore adds
no parameters, which is what makes a horizon-coupled model and a synchronized
one comparable at equal capacity.

The forward signature keeps a 4th history=None argument so a history-bearing
model could drop in without touching every call site; every arm here runs
single-frame and passes None.

Construction order is load-bearing. __init__ is the initialization RNG stream,
so modules must be constructed in exactly this order, and anything new must be
constructed last.
"""

from __future__ import annotations

import torch
from torch import nn


class ActionExpertBlock(nn.Module):
    """One bidirectional action-token block with global adaptive normalization.

    AdaLN: the (state + time) context produces per-block scale/shift pairs for
    the attention and FFN branches. Residuals are ungated (no adaLN-Zero),
    matching the published behaviour.
    """

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        # Affine normalization parameters are replaced by the context modulation.
        self.attention_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True, bias=True)
        self.ffn_norm = nn.LayerNorm(d_model, elementwise_affine=False)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.modulation = nn.Sequential(nn.SiLU(), nn.Linear(d_model, 4 * d_model))

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # (B, d) is one context for the whole chunk; (B, H, d) is one per token.
        # Giving the former an explicit token axis keeps both cases on the same
        # broadcast, rather than relying on a rank coincidence.
        if context.dim() == 2:
            context = context.unsqueeze(1)
        modulation = self.modulation(context)
        attention_scale, attention_shift, ffn_scale, ffn_shift = modulation.chunk(4, dim=-1)

        normalized = self.attention_norm(tokens) * (1.0 + attention_scale)
        normalized = normalized + attention_shift
        # Every future action token may read every other one; there is no mask.
        attended, _ = self.attention(normalized, normalized, normalized, need_weights=False)
        tokens = tokens + attended

        normalized = self.ffn_norm(tokens) * (1.0 + ffn_scale) + ffn_shift
        return tokens + self.ffn(normalized)


class TransformerVelocityModel(nn.Module):
    """Small action expert: one token per future action, no causal mask.

    Selected over the flat MLP at matched parameters on held-out rollouts
    (~144k parameters at the published size). The chunk keeps its (B, H, A)
    structure end to end, each future timestep is a token with a learned
    position, and the observation and flow time modulate every block through
    adaptive normalization instead of being concatenated once at the input.

    Attention is full and bidirectional: the complete noisy chunk is known at
    every ODE step, so this is velocity regression over a whole chunk, not
    autoregressive decoding.

    Flow time enters as a raw scalar through a small MLP. A Fourier expansion
    was tried and dropped; raw tau won on held-out rollouts.
    """

    def __init__(
        self,
        condition_dim: int,
        action_dim: int,
        chunk_size: int,
        *,
        d_model: int = 64,
        num_blocks: int = 3,
        num_heads: int = 4,
        # No default on purpose: a stray default here would let a run train at
        # a width nobody selected. ModelSpec is the single source of the
        # published value.
        ffn_dim: int,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model {d_model} must be divisible by num_heads {num_heads}.")
        self.condition_dim = condition_dim
        self.action_dim = action_dim
        self.chunk_size = chunk_size
        self.d_model = d_model

        self.input_projection = nn.Linear(action_dim, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, chunk_size, d_model))
        nn.init.normal_(self.position_embedding, std=0.02)
        self.state_encoder = nn.Sequential(
            nn.Linear(condition_dim, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.time_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.blocks = nn.ModuleList(
            ActionExpertBlock(d_model, num_heads, ffn_dim) for _ in range(num_blocks)
        )
        self.output_norm = nn.LayerNorm(d_model)
        self.output_projection = nn.Linear(d_model, action_dim)

    def forward(
        self,
        condition: torch.Tensor,
        action_chunk: torch.Tensor,
        tau: torch.Tensor,
        history: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if history is not None:
            # The 4-arg signature is the compatibility seam. A history-bearing
            # model must be constructed last to preserve this init RNG stream
            # (see the module docstring).
            raise ValueError("This model was built without an observation history.")
        batch_size = condition.shape[0]
        if condition.shape != (batch_size, self.condition_dim):
            raise ValueError(
                f"Expected condition shape {(batch_size, self.condition_dim)}, "
                f"got {tuple(condition.shape)}."
            )
        expected_chunk_shape = (batch_size, self.chunk_size, self.action_dim)
        if action_chunk.shape != expected_chunk_shape:
            raise ValueError(
                f"Expected action chunk shape {expected_chunk_shape}, got "
                f"{tuple(action_chunk.shape)}."
            )
        scalar_time = tau.shape == (batch_size, 1)
        if not scalar_time and tau.shape != (batch_size, self.chunk_size, 1):
            raise ValueError(
                f"Expected tau shape {(batch_size, 1)} or "
                f"{(batch_size, self.chunk_size, 1)}, got {tuple(tau.shape)}. A (B, H) "
                "profile missing its trailing axis is the usual cause."
            )

        # The observation context never reads the noisy actions.
        state_context = self.state_encoder(condition)
        time_context = self.time_encoder(tau)
        if not scalar_time:
            # Give the observation an explicit token axis so the sum is per
            # token, rather than letting (B, d) + (B, H, d) broadcast the
            # observation across the horizon by rank accident.
            state_context = state_context.unsqueeze(1)
        context = state_context + time_context
        tokens = self.input_projection(action_chunk) + self.position_embedding
        for block in self.blocks:
            tokens = block(tokens, context)
        return self.output_projection(self.output_norm(tokens))
