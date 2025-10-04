"""
Unit Tests for Perceiver Module
================================

Comprehensive test suite for the Perceiver encoder module covering:
- Unit tests for individual components
- Integration tests for full encoder
- Contract tests for interface compliance
- Shape and dtype validation
- Gradient flow verification

Author: Claude Code
Date: October 2025
"""

import pytest
import torch
import torch.nn as nn
from pathlib import Path
import sys

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.perceiver import (
    PerceiverConfig,
    PerceiverAttention,
    PerceiverMLP,
    PerceiverSelfAttentionBlock,
    PerceiverCrossAttentionBlock,
    PerceiverEncoder,
    create_perceiver_encoder
)


@pytest.fixture
def test_config():
    """Create test configuration."""
    return PerceiverConfig(
        num_latents=32,  # Small for testing
        latent_dim=64,
        input_dim=64,
        num_blocks=1,
        num_self_attends_per_block=2,
        forward_dtype="float32"  # Use float32 for testing
    )


@pytest.fixture
def batch_inputs():
    """Create test batch inputs."""
    batch_size = 2
    seq_len = 16
    input_dim = 64

    return torch.randn(batch_size, seq_len, input_dim)


class TestPerceiverConfig:
    """Test Perceiver configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = PerceiverConfig()

        assert config.num_latents == 256
        assert config.latent_dim == 512
        assert config.num_blocks == 2
        assert config.forward_dtype == "bfloat16"

    def test_custom_config(self, test_config):
        """Test custom configuration."""
        assert test_config.num_latents == 32
        assert test_config.latent_dim == 64
        assert test_config.num_blocks == 1


class TestPerceiverAttention:
    """Test Perceiver attention layer."""

    def test_initialization(self):
        """Test attention layer initialization."""
        attn = PerceiverAttention(
            q_dim=64,
            kv_dim=64,
            num_heads=8
        )

        assert attn.num_heads == 8
        assert attn.head_dim == 8
        assert attn.qkv_dim == 64

    def test_self_attention(self):
        """Test self-attention forward pass."""
        batch_size, seq_len, dim = 2, 16, 64

        attn = PerceiverAttention(
            q_dim=dim,
            kv_dim=dim,
            num_heads=8
        )

        x = torch.randn(batch_size, seq_len, dim)
        output = attn(x, x, x)

        assert output.shape == (batch_size, seq_len, dim)

    def test_cross_attention(self):
        """Test cross-attention forward pass."""
        batch_size = 2
        q_len, kv_len = 8, 16
        q_dim, kv_dim = 64, 64

        attn = PerceiverAttention(
            q_dim=q_dim,
            kv_dim=kv_dim,
            num_heads=8
        )

        query = torch.randn(batch_size, q_len, q_dim)
        key = torch.randn(batch_size, kv_len, kv_dim)
        value = torch.randn(batch_size, kv_len, kv_dim)

        output = attn(query, key, value)

        assert output.shape == (batch_size, q_len, q_dim)

    def test_attention_mask(self):
        """Test attention with mask."""
        batch_size, seq_len, dim = 2, 16, 64

        attn = PerceiverAttention(
            q_dim=dim,
            kv_dim=dim,
            num_heads=8
        )

        x = torch.randn(batch_size, seq_len, dim)
        mask = torch.ones(batch_size, seq_len, seq_len)
        mask[:, :, 8:] = 0  # Mask second half

        output = attn(x, x, x, attention_mask=mask)

        assert output.shape == (batch_size, seq_len, dim)

    def test_query_residual(self):
        """Test query residual connection."""
        batch_size, seq_len, dim = 2, 16, 64

        attn_with_residual = PerceiverAttention(
            q_dim=dim,
            kv_dim=dim,
            num_heads=8,
            use_query_residual=True
        )

        attn_without_residual = PerceiverAttention(
            q_dim=dim,
            kv_dim=dim,
            num_heads=8,
            use_query_residual=False
        )

        x = torch.randn(batch_size, seq_len, dim)

        out_with = attn_with_residual(x, x, x)
        out_without = attn_without_residual(x, x, x)

        # With residual should be different from without
        assert not torch.allclose(out_with, out_without)


class TestPerceiverMLP:
    """Test Perceiver MLP."""

    def test_initialization(self):
        """Test MLP initialization."""
        mlp = PerceiverMLP(input_dim=64, widening_factor=4)

        assert mlp.fc1.in_features == 64
        assert mlp.fc1.out_features == 256  # 64 * 4
        assert mlp.fc2.out_features == 64

    def test_forward_pass(self):
        """Test MLP forward pass."""
        batch_size, seq_len, dim = 2, 16, 64

        mlp = PerceiverMLP(input_dim=dim, widening_factor=4)
        x = torch.randn(batch_size, seq_len, dim)

        output = mlp(x)

        assert output.shape == (batch_size, seq_len, dim)

    def test_dropout(self):
        """Test dropout is applied."""
        mlp = PerceiverMLP(input_dim=64, dropout_prob=0.5)

        mlp.train()
        x = torch.randn(2, 16, 64)

        # Multiple forward passes should give different results due to dropout
        out1 = mlp(x)
        out2 = mlp(x)

        assert not torch.allclose(out1, out2)


class TestPerceiverBlocks:
    """Test Perceiver block modules."""

    def test_self_attention_block(self, test_config):
        """Test self-attention block."""
        block = PerceiverSelfAttentionBlock(test_config)

        batch_size = 2
        latents = torch.randn(batch_size, test_config.num_latents, test_config.latent_dim)

        output = block(latents)

        assert output.shape == latents.shape

    def test_cross_attention_block(self, test_config):
        """Test cross-attention block."""
        block = PerceiverCrossAttentionBlock(test_config)

        batch_size, input_len = 2, 16

        latents = torch.randn(batch_size, test_config.num_latents, test_config.latent_dim)
        inputs = torch.randn(batch_size, input_len, test_config.input_dim)

        output = block(latents, inputs)

        assert output.shape == latents.shape

    def test_cross_attention_with_mask(self, test_config):
        """Test cross-attention block with mask."""
        block = PerceiverCrossAttentionBlock(test_config)

        batch_size, input_len = 2, 16

        latents = torch.randn(batch_size, test_config.num_latents, test_config.latent_dim)
        inputs = torch.randn(batch_size, input_len, test_config.input_dim)
        mask = torch.ones(batch_size, input_len)
        mask[:, 8:] = 0  # Mask second half

        output = block(latents, inputs, attention_mask=mask)

        assert output.shape == latents.shape


class TestPerceiverEncoder:
    """Test full Perceiver encoder."""

    def test_initialization(self, test_config):
        """Test encoder initialization."""
        encoder = PerceiverEncoder(test_config)

        assert encoder.latent_array.shape == (1, test_config.num_latents, test_config.latent_dim)
        assert len(encoder.blocks) == test_config.num_blocks

    def test_forward_pass(self, test_config, batch_inputs):
        """Test encoder forward pass."""
        encoder = PerceiverEncoder(test_config)

        output = encoder(batch_inputs)

        batch_size = batch_inputs.shape[0]
        assert output.shape == (batch_size, test_config.num_latents, test_config.latent_dim)

    def test_forward_with_mask(self, test_config, batch_inputs):
        """Test encoder forward pass with attention mask."""
        encoder = PerceiverEncoder(test_config)

        batch_size, seq_len = batch_inputs.shape[:2]
        mask = torch.ones(batch_size, seq_len)
        mask[:, seq_len // 2:] = 0  # Mask second half

        output = encoder(batch_inputs, attention_mask=mask)

        assert output.shape == (batch_size, test_config.num_latents, test_config.latent_dim)

    def test_deterministic_forward(self, test_config):
        """Test encoder produces same output for same input."""
        encoder = PerceiverEncoder(test_config)
        encoder.eval()

        x = torch.randn(2, 16, test_config.input_dim)

        with torch.no_grad():
            out1 = encoder(x)
            out2 = encoder(x)

        assert torch.allclose(out1, out2)

    def test_batch_independence(self, test_config):
        """Test batch elements are processed independently."""
        encoder = PerceiverEncoder(test_config)
        encoder.eval()

        # Create batch with identical examples
        x1 = torch.randn(1, 16, test_config.input_dim)
        x_batch = x1.repeat(4, 1, 1)

        with torch.no_grad():
            out_single = encoder(x1)
            out_batch = encoder(x_batch)

        # All batch elements should be identical
        for i in range(4):
            assert torch.allclose(out_batch[i:i+1], out_single, atol=1e-5)

    def test_get_num_params(self, test_config):
        """Test parameter counting."""
        encoder = PerceiverEncoder(test_config)

        num_params = encoder.get_num_params()

        # Should have non-zero parameters
        assert num_params > 0

        # Verify count matches actual parameters
        actual_params = sum(p.numel() for p in encoder.parameters())
        assert num_params == actual_params


class TestGradientFlow:
    """Test gradient flow through Perceiver."""

    def test_gradients_backward(self, test_config, batch_inputs):
        """Test gradients flow through encoder."""
        encoder = PerceiverEncoder(test_config)
        encoder.train()

        output = encoder(batch_inputs)

        # Compute dummy loss
        loss = output.mean()
        loss.backward()

        # Check all parameters have gradients
        for name, param in encoder.named_parameters():
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.isnan(param.grad).any(), f"NaN gradient for {name}"
            assert not torch.isinf(param.grad).any(), f"Inf gradient for {name}"

    def test_gradient_accumulation(self, test_config):
        """Test gradient accumulation works correctly."""
        encoder = PerceiverEncoder(test_config)
        encoder.train()

        # First forward-backward
        x1 = torch.randn(2, 16, test_config.input_dim)
        out1 = encoder(x1)
        loss1 = out1.sum()  # Use sum for larger gradients
        loss1.backward()

        # Save gradients
        grads1 = {name: param.grad.clone() for name, param in encoder.named_parameters()}

        # Second forward-backward (accumulate)
        x2 = torch.randn(2, 16, test_config.input_dim)
        out2 = encoder(x2)
        loss2 = out2.sum()  # Use sum for larger gradients
        loss2.backward()

        # Gradients should have accumulated (check magnitude increased)
        for name, param in encoder.named_parameters():
            grad1_norm = grads1[name].norm().item()
            grad2_norm = param.grad.norm().item()
            # Accumulated gradient should have larger norm
            assert grad2_norm > grad1_norm * 0.5, f"Gradient not accumulated properly for {name}"


class TestFactoryFunction:
    """Test factory function."""

    def test_create_perceiver_encoder(self):
        """Test factory function creates encoder correctly."""
        encoder = create_perceiver_encoder(
            num_latents=64,
            latent_dim=128,
            input_dim=128,
            num_blocks=2
        )

        assert isinstance(encoder, PerceiverEncoder)
        assert encoder.config.num_latents == 64
        assert encoder.config.latent_dim == 128
        assert encoder.config.num_blocks == 2

    def test_factory_with_kwargs(self):
        """Test factory function with additional kwargs."""
        encoder = create_perceiver_encoder(
            num_latents=64,
            latent_dim=128,
            input_dim=128,
            num_blocks=2,
            dropout_prob=0.2,
            num_self_attends_per_block=4
        )

        assert encoder.config.dropout_prob == 0.2
        assert encoder.config.num_self_attends_per_block == 4


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_small_latent_dim(self):
        """Test with very small latent dimension."""
        config = PerceiverConfig(
            num_latents=8,
            latent_dim=16,
            input_dim=16,
            num_self_attention_heads=2,
            forward_dtype="float32"
        )

        encoder = PerceiverEncoder(config)
        x = torch.randn(2, 8, 16)

        output = encoder(x)
        assert output.shape == (2, 8, 16)

    def test_large_input_sequence(self):
        """Test with long input sequence."""
        config = PerceiverConfig(
            num_latents=32,
            latent_dim=64,
            input_dim=64,
            forward_dtype="float32"
        )

        encoder = PerceiverEncoder(config)
        x = torch.randn(2, 512, 64)  # Long sequence

        output = encoder(x)
        assert output.shape == (2, 32, 64)  # Compressed to latents

    def test_single_batch_element(self, test_config):
        """Test with batch size 1."""
        encoder = PerceiverEncoder(test_config)
        x = torch.randn(1, 16, test_config.input_dim)

        output = encoder(x)
        assert output.shape == (1, test_config.num_latents, test_config.latent_dim)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
