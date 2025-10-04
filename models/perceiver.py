"""
Perceiver Encoder Module
========================

A Perceiver-style cross-attention encoder that processes variable-length inputs
into a fixed-size latent representation. Designed for ARC puzzle preprocessing.

The Perceiver uses learned latent queries to extract relevant features from
input sequences through iterative cross-attention, enabling efficient processing
of high-dimensional inputs (30x30 grids = 900 tokens).

Key features:
- Fixed-size latent bottleneck (256 latents)
- Cross-attention from latents to inputs
- Self-attention among latents
- Residual connections and layer normalization
- Configurable depth and attention parameters

References:
    Jaegle et al. "Perceiver: General Perception with Iterative Attention" (2021)
    https://arxiv.org/abs/2103.03206

Author: Claude Code
Date: October 2025
"""

from typing import Optional, Tuple
from dataclasses import dataclass
import math
import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
from pydantic import BaseModel


logger = logging.getLogger(__name__)


class PerceiverConfig(BaseModel):
    """Configuration for Perceiver encoder."""

    # Latent configuration
    num_latents: int = 256
    latent_dim: int = 512

    # Cross-attention configuration
    num_cross_attention_heads: int = 8
    num_self_attention_heads: int = 8
    cross_attention_widening_factor: int = 1
    self_attention_widening_factor: int = 1

    # Architecture
    num_blocks: int = 2  # Number of cross-attention + self-attention blocks
    num_self_attends_per_block: int = 6  # Self-attention layers per block

    # Dropout and regularization
    dropout_prob: float = 0.0
    attention_dropout_prob: float = 0.0
    layer_norm_eps: float = 1e-5

    # Input projection
    input_dim: int = 512  # Dimension after initial embedding
    use_query_residual: bool = True

    # Forward pass configuration
    forward_dtype: str = "bfloat16"


class PerceiverAttention(nn.Module):
    """
    Multi-head attention layer for Perceiver.

    Supports both self-attention (Q, K, V from same source) and
    cross-attention (Q from latents, K/V from inputs).
    """

    def __init__(
        self,
        q_dim: int,
        kv_dim: int,
        num_heads: int,
        head_dim: Optional[int] = None,
        dropout_prob: float = 0.0,
        use_query_residual: bool = True
    ):
        """
        Initialize attention layer.

        Args:
            q_dim: Dimension of query vectors
            kv_dim: Dimension of key/value vectors
            num_heads: Number of attention heads
            head_dim: Dimension per head (defaults to q_dim // num_heads)
            dropout_prob: Dropout probability
            use_query_residual: Whether to use residual connection on queries
        """
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = head_dim or (q_dim // num_heads)
        self.use_query_residual = use_query_residual

        # Ensure dimensions are valid
        if q_dim % num_heads != 0 and head_dim is None:
            raise ValueError(
                f"q_dim ({q_dim}) must be divisible by num_heads ({num_heads}) "
                f"or head_dim must be specified"
            )

        self.qkv_dim = self.num_heads * self.head_dim
        self.scale = self.head_dim ** -0.5

        # Projections
        self.q_proj = nn.Linear(q_dim, self.qkv_dim, bias=False)
        self.k_proj = nn.Linear(kv_dim, self.qkv_dim, bias=False)
        self.v_proj = nn.Linear(kv_dim, self.qkv_dim, bias=False)
        self.out_proj = nn.Linear(self.qkv_dim, q_dim, bias=False)

        self.dropout = nn.Dropout(dropout_prob)

        logger.debug(
            f"PerceiverAttention initialized: q_dim={q_dim}, kv_dim={kv_dim}, "
            f"num_heads={num_heads}, head_dim={self.head_dim}"
        )

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of attention.

        Args:
            query: Query tensor [batch, q_len, q_dim]
            key: Key tensor [batch, kv_len, kv_dim]
            value: Value tensor [batch, kv_len, kv_dim]
            attention_mask: Optional mask [batch, q_len, kv_len] or broadcastable

        Returns:
            Output tensor [batch, q_len, q_dim]
        """
        batch_size, q_len, _ = query.shape
        kv_len = key.shape[1]

        # Project and reshape to [batch, num_heads, len, head_dim]
        q = self.q_proj(query).view(batch_size, q_len, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(key).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(value).view(batch_size, kv_len, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        # [batch, num_heads, q_len, kv_len]
        attn_weights = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        # Apply attention mask if provided
        if attention_mask is not None:
            # Mask should be broadcastable to [batch, num_heads, q_len, kv_len]
            if attention_mask.dim() == 2:
                attention_mask = attention_mask.unsqueeze(1).unsqueeze(1)
            elif attention_mask.dim() == 3:
                attention_mask = attention_mask.unsqueeze(1)

            attn_weights = attn_weights.masked_fill(attention_mask == 0, float('-inf'))

        attn_weights = F.softmax(attn_weights, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # Apply attention to values
        # [batch, num_heads, q_len, head_dim]
        attn_output = torch.matmul(attn_weights, v)

        # Reshape and project
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, q_len, self.qkv_dim)
        output = self.out_proj(attn_output)

        # Query residual connection (used in cross-attention)
        if self.use_query_residual and query.shape[-1] == output.shape[-1]:
            output = output + query

        return output


class PerceiverMLP(nn.Module):
    """Feed-forward MLP with GELU activation."""

    def __init__(
        self,
        input_dim: int,
        widening_factor: int = 1,
        dropout_prob: float = 0.0
    ):
        """
        Initialize MLP.

        Args:
            input_dim: Input dimension
            widening_factor: Multiplier for hidden dimension
            dropout_prob: Dropout probability
        """
        super().__init__()

        hidden_dim = input_dim * widening_factor

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, input_dim)
        self.dropout = nn.Dropout(dropout_prob)

        logger.debug(f"PerceiverMLP initialized: input_dim={input_dim}, hidden_dim={hidden_dim}")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor [..., input_dim]

        Returns:
            Output tensor [..., input_dim]
        """
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class PerceiverSelfAttentionBlock(nn.Module):
    """Self-attention block for processing latents."""

    def __init__(self, config: PerceiverConfig):
        """
        Initialize self-attention block.

        Args:
            config: Perceiver configuration
        """
        super().__init__()

        self.attention = PerceiverAttention(
            q_dim=config.latent_dim,
            kv_dim=config.latent_dim,
            num_heads=config.num_self_attention_heads,
            dropout_prob=config.attention_dropout_prob,
            use_query_residual=True
        )

        self.mlp = PerceiverMLP(
            input_dim=config.latent_dim,
            widening_factor=config.self_attention_widening_factor,
            dropout_prob=config.dropout_prob
        )

        self.layernorm1 = nn.LayerNorm(config.latent_dim, eps=config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(config.latent_dim, eps=config.layer_norm_eps)

    def forward(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of self-attention block.

        Args:
            latents: Latent tensor [batch, num_latents, latent_dim]

        Returns:
            Updated latents [batch, num_latents, latent_dim]
        """
        # Self-attention with residual
        latents = self.layernorm1(latents)
        latents = latents + self.attention(latents, latents, latents)

        # MLP with residual
        latents = self.layernorm2(latents)
        latents = latents + self.mlp(latents)

        return latents


class PerceiverCrossAttentionBlock(nn.Module):
    """Cross-attention block for encoding inputs into latents."""

    def __init__(self, config: PerceiverConfig):
        """
        Initialize cross-attention block.

        Args:
            config: Perceiver configuration
        """
        super().__init__()

        self.cross_attention = PerceiverAttention(
            q_dim=config.latent_dim,
            kv_dim=config.input_dim,
            num_heads=config.num_cross_attention_heads,
            dropout_prob=config.attention_dropout_prob,
            use_query_residual=config.use_query_residual
        )

        self.mlp = PerceiverMLP(
            input_dim=config.latent_dim,
            widening_factor=config.cross_attention_widening_factor,
            dropout_prob=config.dropout_prob
        )

        self.layernorm1 = nn.LayerNorm(config.latent_dim, eps=config.layer_norm_eps)
        self.layernorm2 = nn.LayerNorm(config.latent_dim, eps=config.layer_norm_eps)
        self.layernorm_input = nn.LayerNorm(config.input_dim, eps=config.layer_norm_eps)

    def forward(
        self,
        latents: torch.Tensor,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass of cross-attention block.

        Args:
            latents: Latent queries [batch, num_latents, latent_dim]
            inputs: Input data [batch, input_len, input_dim]
            attention_mask: Optional mask for inputs

        Returns:
            Updated latents [batch, num_latents, latent_dim]
        """
        # Normalize inputs
        inputs = self.layernorm_input(inputs)

        # Cross-attention: latents attend to inputs
        latents = self.layernorm1(latents)
        latents = latents + self.cross_attention(
            query=latents,
            key=inputs,
            value=inputs,
            attention_mask=attention_mask
        )

        # MLP with residual
        latents = self.layernorm2(latents)
        latents = latents + self.mlp(latents)

        return latents


class PerceiverEncoder(nn.Module):
    """
    Perceiver encoder: encodes variable-length inputs into fixed-size latents.

    Architecture:
        1. Initialize learned latent array
        2. For each block:
            a. Cross-attention: latents attend to inputs
            b. Multiple self-attention layers on latents
        3. Final layer norm on latents

    This creates a compressed representation that the HRM can process efficiently.
    """

    def __init__(self, config: PerceiverConfig):
        """
        Initialize Perceiver encoder.

        Args:
            config: Perceiver configuration
        """
        super().__init__()

        self.config = config

        # Learned latent array - this is the bottleneck
        self.latent_array = nn.Parameter(
            torch.randn(1, config.num_latents, config.latent_dim)
        )

        # Initialize latent array with Xavier uniform
        nn.init.xavier_uniform_(self.latent_array)

        # Build blocks: each has cross-attention + self-attention
        self.blocks = nn.ModuleList()

        for block_idx in range(config.num_blocks):
            # Cross-attention block
            cross_attn = PerceiverCrossAttentionBlock(config)

            # Self-attention blocks
            self_attns = nn.ModuleList([
                PerceiverSelfAttentionBlock(config)
                for _ in range(config.num_self_attends_per_block)
            ])

            self.blocks.append(nn.ModuleDict({
                'cross_attention': cross_attn,
                'self_attentions': self_attns
            }))

        # Final layer norm
        self.final_layernorm = nn.LayerNorm(config.latent_dim, eps=config.layer_norm_eps)

        # Convert to target dtype
        target_dtype = getattr(torch, config.forward_dtype)
        self.to(dtype=target_dtype)

        logger.info(
            f"PerceiverEncoder initialized: "
            f"num_latents={config.num_latents}, latent_dim={config.latent_dim}, "
            f"num_blocks={config.num_blocks}, dtype={config.forward_dtype}"
        )

    def forward(
        self,
        inputs: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Encode inputs into latent representation.

        Args:
            inputs: Input tensor [batch, input_len, input_dim]
            attention_mask: Optional mask [batch, input_len] where 1=valid, 0=masked

        Returns:
            Latent representation [batch, num_latents, latent_dim]
        """
        batch_size = inputs.shape[0]

        # Expand latent array for batch
        latents = self.latent_array.expand(batch_size, -1, -1)

        # Process through blocks
        for block in self.blocks:
            # Cross-attention: latents attend to inputs
            latents = block['cross_attention'](
                latents=latents,
                inputs=inputs,
                attention_mask=attention_mask
            )

            # Self-attention: latents attend to themselves
            for self_attn in block['self_attentions']:
                latents = self_attn(latents)

        # Final normalization
        latents = self.final_layernorm(latents)

        return latents

    def get_num_params(self) -> int:
        """Get total number of parameters."""
        return sum(p.numel() for p in self.parameters())


def create_perceiver_encoder(
    num_latents: int = 256,
    latent_dim: int = 512,
    input_dim: int = 512,
    num_blocks: int = 2,
    **kwargs
) -> PerceiverEncoder:
    """
    Factory function to create Perceiver encoder.

    Args:
        num_latents: Number of latent vectors
        latent_dim: Dimension of latent vectors
        input_dim: Dimension of input vectors
        num_blocks: Number of cross-attention + self-attention blocks
        **kwargs: Additional configuration parameters

    Returns:
        Initialized PerceiverEncoder
    """
    config = PerceiverConfig(
        num_latents=num_latents,
        latent_dim=latent_dim,
        input_dim=input_dim,
        num_blocks=num_blocks,
        **kwargs
    )

    return PerceiverEncoder(config)
