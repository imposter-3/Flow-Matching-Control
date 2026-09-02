"""The velocity field: a small action expert over one token per horizon position.

Interface:

    condition:    (B, C)          normalized observation frame
    action_chunk: (B, H, A)       the flow state
    tau:          (B, 1)          one flow time for the whole chunk
              or  (B, H, 1)       one flow time per horizon position
    returns:      (B, H, A)       velocity, in the same coordinates

Both tau ranks are supported. A scalar time is the ordinary synchronized case;
a profile is the mixed-time case, where different horizon positions sit at
different stages of generation. Both go through the same encoder and the same
parameters: the encoder maps the last axis, so a profile produces a per-token
context where a scalar produces a per-chunk one, and the modulation broadcasts
accordingly. Mixed time therefore adds no parameters, which is what makes a
mixed-time model and a synchronized model comparable at equal capacity.

Attention is full and bidirectional. The entire noisy chunk is known at every
integration step, so this is velocity regression over a whole chunk rather than
autoregressive decoding, and a causal mask would hide information the model
needs.
"""

from __future__ import annotations

import torch
from torch import nn

NORM_EPS = 1e-5


class ActionExpertBlock(nn.Module):
    """Self-attention + FFN, both modulated by the conditioning context.

    Normalization is affine-free because the affine parameters are exactly what
    the context supplies: each sublayer is normalized, then scaled and shifted
    by a projection of the context (the adaLN construction). The projection
    emits four vectors: attention scale/shift and FFN scale/shift.
    """

    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.attention_norm = nn.LayerNorm(
            d_model, eps=NORM_EPS, elementwise_affine=False
        )
        self.ffn_norm = nn.LayerNorm(d_model, eps=NORM_EPS, elementwise_affine=False)
        self.attention = nn.MultiheadAttention(d_model, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Linear(ffn_dim, d_model),
        )
        self.modulation = nn.Sequential(
            nn.GELU(),
            nn.Linear(d_model, 4 * d_model),
        )

    def forward(self, tokens: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        # context is (B, d) for a scalar flow time and (B, H, d) for a profile.
        # Giving the per-chunk case an explicit token axis makes both broadcast
        # against (B, H, d) tokens under one code path.
        if context.dim() == 2:
            context = context.unsqueeze(1)
        scales = self.modulation(context).chunk(4, dim=-1)
        attention_scale, attention_shift, ffn_scale, ffn_shift = scales

        normalized = self.attention_norm(tokens) * (1.0 + attention_scale)
        normalized = normalized + attention_shift
        attended, _ = self.attention(
            normalized, normalized, normalized, need_weights=False
        )
        tokens = tokens + attended

        normalized = self.ffn_norm(tokens) * (1.0 + ffn_scale) + ffn_shift
        return tokens + self.ffn(normalized)


class VelocityModel(nn.Module):
    """One token per future action; conditioning enters only through modulation.

    The submodule construction order below is frozen: __init__ is the parameter
    initialization RNG stream under torch.manual_seed (note the explicit
    nn.init.normal_ on the position embedding), so inserting or reordering a
    module shifts every subsequent draw and breaks bit-identical retraining.
    The order is: input_projection, position_embedding, state_encoder,
    time_encoder, blocks, output_norm, output_projection.
    """

    def __init__(
        self,
        condition_dim: int,
        action_dim: int,
        chunk_size: int,
        *,
        d_model: int,
        num_blocks: int,
        num_heads: int,
        ffn_dim: int,
    ) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(
                f"d_model {d_model} must be divisible by num_heads {num_heads}."
            )
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
        # Raw flow time, not a Fourier expansion. The encoder maps the last
        # axis, so it is agnostic to whether that axis sits on (B, 1) or
        # (B, H, 1), which is what lets one set of weights serve both regimes.
        self.time_encoder = nn.Sequential(
            nn.Linear(1, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        self.blocks = nn.ModuleList(
            ActionExpertBlock(d_model, num_heads, ffn_dim) for _ in range(num_blocks)
        )
        self.output_norm = nn.LayerNorm(d_model, eps=NORM_EPS)
        self.output_projection = nn.Linear(d_model, action_dim)

    def forward(
        self,
        condition: torch.Tensor,
        action_chunk: torch.Tensor,
        tau: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = condition.shape[0]
        if condition.shape != (batch_size, self.condition_dim):
            raise ValueError(
                f"Expected condition {(batch_size, self.condition_dim)}, got "
                f"{tuple(condition.shape)}."
            )
        expected_chunk = (batch_size, self.chunk_size, self.action_dim)
        if action_chunk.shape != expected_chunk:
            raise ValueError(
                f"Expected action chunk {expected_chunk}, got "
                f"{tuple(action_chunk.shape)}."
            )
        scalar_time = (batch_size, 1)
        profile_time = (batch_size, self.chunk_size, 1)
        if tau.shape not in (scalar_time, profile_time):
            raise ValueError(
                f"Expected tau {scalar_time} or {profile_time}, got "
                f"{tuple(tau.shape)}. A (B, H) profile missing its trailing "
                "axis is the usual cause."
            )

        # The observation context never reads the noisy actions.
        state_context = self.state_encoder(condition)
        time_context = self.time_encoder(tau)
        if time_context.dim() == 3:
            # Per-position time: give the observation an explicit token axis so
            # the sum is per token instead of broadcasting across positions.
            state_context = state_context.unsqueeze(1)
        context = state_context + time_context

        tokens = self.input_projection(action_chunk) + self.position_embedding
        for block in self.blocks:
            tokens = block(tokens, context)
        return self.output_projection(self.output_norm(tokens))


def build_model(*, condition_dim: int, action_dim: int, recipe) -> VelocityModel:
    """Construct the velocity field from the shared recipe.

    Everything architectural comes from the recipe, which no method may
    override, so all four arms are compared at identical capacity.
    """

    return VelocityModel(
        condition_dim=condition_dim,
        action_dim=action_dim,
        chunk_size=recipe.chunk_size,
        d_model=recipe.d_model,
        num_blocks=recipe.num_blocks,
        num_heads=recipe.num_heads,
        ffn_dim=recipe.ffn_dim,
    )


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
