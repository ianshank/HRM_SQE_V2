"""
Unit tests for CastedEmbedding index validation.

Tests the index bounds checking in CastedEmbedding to ensure
IndexError prevention before CUDA kernel execution.
"""

import unittest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.layers import CastedEmbedding


class TestCastedEmbeddingValidation(unittest.TestCase):
    """Test CastedEmbedding index validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_embeddings = 12  # vocab_size for ARC
        self.embedding_dim = 256
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_valid_indices(self):
        """Test that valid indices work correctly."""
        embedding = CastedEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            init_std=1.0,
            cast_to=torch.float32
        ).to(self.device)

        # Valid indices [0, 11]
        valid_indices = torch.randint(0, self.num_embeddings, (16, 100), device=self.device)

        # Should work without error
        output = embedding(valid_indices)

        self.assertEqual(output.shape, (16, 100, self.embedding_dim))
        self.assertEqual(output.device, self.device)

    def test_out_of_bounds_upper(self):
        """Test that out-of-bounds indices (too high) raise IndexError."""
        embedding = CastedEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            init_std=1.0,
            cast_to=torch.float32
        ).to(self.device)

        # Index at upper bound (should fail)
        invalid_indices = torch.full((16, 100), self.num_embeddings, dtype=torch.long, device=self.device)

        with self.assertRaises(IndexError) as context:
            _ = embedding(invalid_indices)

        self.assertIn("out of bounds", str(context.exception).lower())
        self.assertIn(str(self.num_embeddings), str(context.exception))

    def test_out_of_bounds_negative(self):
        """Test that negative indices raise IndexError."""
        embedding = CastedEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            init_std=1.0,
            cast_to=torch.float32
        ).to(self.device)

        # Negative index (should fail)
        invalid_indices = torch.full((16, 100), -1, dtype=torch.long, device=self.device)

        with self.assertRaises(IndexError) as context:
            _ = embedding(invalid_indices)

        self.assertIn("out of bounds", str(context.exception).lower())

    def test_boundary_indices(self):
        """Test indices at valid boundaries."""
        embedding = CastedEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            init_std=1.0,
            cast_to=torch.float32
        ).to(self.device)

        # Test index 0 (lower boundary)
        lower_boundary = torch.zeros((16, 100), dtype=torch.long, device=self.device)
        output_lower = embedding(lower_boundary)
        self.assertEqual(output_lower.shape, (16, 100, self.embedding_dim))

        # Test index num_embeddings-1 (upper boundary)
        upper_boundary = torch.full((16, 100), self.num_embeddings - 1, dtype=torch.long, device=self.device)
        output_upper = embedding(upper_boundary)
        self.assertEqual(output_upper.shape, (16, 100, self.embedding_dim))

    def test_mixed_valid_invalid_indices(self):
        """Test batch with mix of valid and invalid indices."""
        embedding = CastedEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            init_std=1.0,
            cast_to=torch.float32
        ).to(self.device)

        # Mix of valid and one invalid index
        mixed_indices = torch.randint(0, self.num_embeddings, (16, 100), device=self.device)
        mixed_indices[0, 0] = self.num_embeddings  # Add one invalid index

        with self.assertRaises(IndexError):
            _ = embedding(mixed_indices)

    def test_dtype_casting(self):
        """Test that output is cast to correct dtype."""
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            if dtype == torch.bfloat16 and not torch.cuda.is_available():
                continue  # Skip bfloat16 on CPU

            embedding = CastedEmbedding(
                num_embeddings=self.num_embeddings,
                embedding_dim=self.embedding_dim,
                init_std=1.0,
                cast_to=dtype
            ).to(self.device)

            indices = torch.randint(0, self.num_embeddings, (16, 100), device=self.device)
            output = embedding(indices)

            self.assertEqual(output.dtype, dtype)

    def test_padding_indices(self):
        """Test with padding index (0)."""
        embedding = CastedEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            init_std=1.0,
            cast_to=torch.float32
        ).to(self.device)

        # All padding (index 0)
        padding_indices = torch.zeros((16, 100), dtype=torch.long, device=self.device)
        output = embedding(padding_indices)

        self.assertEqual(output.shape, (16, 100, self.embedding_dim))

    def test_with_actual_arc_vocab(self):
        """Test with actual ARC dataset vocab size."""
        vocab_size = 12
        seq_len = 900
        batch_size = 16

        embedding = CastedEmbedding(
            num_embeddings=vocab_size,
            embedding_dim=self.embedding_dim,
            init_std=1.0,
            cast_to=torch.float32
        ).to(self.device)

        # Simulate actual input tokens [0-11]
        tokens = torch.randint(0, vocab_size, (batch_size, seq_len), device=self.device)
        output = embedding(tokens)

        self.assertEqual(output.shape, (batch_size, seq_len, self.embedding_dim))

        # Test with max valid token (11)
        max_tokens = torch.full((batch_size, seq_len), vocab_size - 1, dtype=torch.long, device=self.device)
        output_max = embedding(max_tokens)
        self.assertEqual(output_max.shape, (batch_size, seq_len, self.embedding_dim))

        # Test that vocab_size is out of bounds
        invalid_tokens = torch.full((batch_size, seq_len), vocab_size, dtype=torch.long, device=self.device)
        with self.assertRaises(IndexError):
            _ = embedding(invalid_tokens)


class TestCastedEmbeddingGradients(unittest.TestCase):
    """Test gradient flow through CastedEmbedding."""

    def test_gradient_flow(self):
        """Test that gradients flow correctly through embedding."""
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        embedding = CastedEmbedding(
            num_embeddings=100,
            embedding_dim=128,
            init_std=1.0,
            cast_to=torch.float32
        ).to(device)

        indices = torch.randint(0, 100, (16, 50), device=device)
        output = embedding(indices)

        # Compute loss and backward
        loss = output.sum()
        loss.backward()

        # Check gradients exist
        self.assertIsNotNone(embedding.embedding_weight.grad)
        self.assertEqual(embedding.embedding_weight.grad.shape, embedding.embedding_weight.shape)


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
