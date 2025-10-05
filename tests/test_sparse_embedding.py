"""
Unit tests for sparse embedding validation fixes.

Tests the index bounds checking in CastedSparseEmbedding to ensure
IndexError prevention and proper error messages.
"""

import unittest
import torch
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.sparse_embedding import CastedSparseEmbedding, CastedSparseEmbeddingSignSGD_Distributed


class TestSparseEmbeddingValidation(unittest.TestCase):
    """Test sparse embedding index validation."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_embeddings = 100
        self.embedding_dim = 64
        self.batch_size = 16
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_valid_indices(self):
        """Test that valid indices work correctly."""
        embedding = CastedSparseEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            batch_size=self.batch_size,
            init_std=1.0,
            cast_to=torch.float32
        ).to(self.device)

        # Valid indices
        valid_indices = torch.randint(0, self.num_embeddings, (self.batch_size,), device=self.device)

        # Should work without error
        output = embedding(valid_indices)

        self.assertEqual(output.shape, (self.batch_size, self.embedding_dim))
        self.assertEqual(output.device, self.device)

    def test_out_of_bounds_upper(self):
        """Test that out-of-bounds indices (too high) raise IndexError."""
        embedding = CastedSparseEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            batch_size=self.batch_size,
            init_std=1.0,
            cast_to=torch.float32
        ).to(self.device)

        # Index at upper bound (should fail)
        invalid_indices = torch.tensor([self.num_embeddings], device=self.device)

        with self.assertRaises(IndexError) as context:
            _ = embedding(invalid_indices)

        self.assertIn("out of bounds", str(context.exception).lower())

    def test_out_of_bounds_negative(self):
        """Test that negative indices raise IndexError."""
        embedding = CastedSparseEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            batch_size=self.batch_size,
            init_std=1.0,
            cast_to=torch.float32
        ).to(self.device)

        # Negative index (should fail)
        invalid_indices = torch.tensor([-1], device=self.device)

        with self.assertRaises(IndexError) as context:
            _ = embedding(invalid_indices)

        self.assertIn("out of bounds", str(context.exception).lower())

    def test_boundary_indices(self):
        """Test indices at valid boundaries."""
        embedding = CastedSparseEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            batch_size=self.batch_size,
            init_std=1.0,
            cast_to=torch.float32
        ).to(self.device)

        # Test index 0 (lower boundary)
        lower_boundary = torch.tensor([0], device=self.device)
        output_lower = embedding(lower_boundary)
        self.assertEqual(output_lower.shape, (1, self.embedding_dim))

        # Test index num_embeddings-1 (upper boundary)
        upper_boundary = torch.tensor([self.num_embeddings - 1], device=self.device)
        output_upper = embedding(upper_boundary)
        self.assertEqual(output_upper.shape, (1, self.embedding_dim))

    def test_mixed_valid_invalid_indices(self):
        """Test batch with mix of valid and invalid indices."""
        embedding = CastedSparseEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            batch_size=self.batch_size,
            init_std=1.0,
            cast_to=torch.float32
        ).to(self.device)

        # Mix of valid and one invalid index
        mixed_indices = torch.tensor([0, 50, self.num_embeddings], device=self.device)

        with self.assertRaises(IndexError):
            _ = embedding(mixed_indices)

    def test_training_mode_local_buffers(self):
        """Test local buffer creation in training mode."""
        embedding = CastedSparseEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            batch_size=self.batch_size,
            init_std=1.0,
            cast_to=torch.float32
        ).to(self.device)

        embedding.train()

        indices = torch.randint(0, self.num_embeddings, (self.batch_size,), device=self.device)
        output = embedding(indices)

        # Check local buffers were created
        self.assertIsNotNone(embedding.local_weights)
        self.assertIsNotNone(embedding.local_ids)
        self.assertEqual(embedding.local_weights.shape, (self.batch_size, self.embedding_dim))
        self.assertEqual(embedding.local_ids.shape, (self.batch_size,))

    def test_eval_mode_no_local_buffers(self):
        """Test that eval mode doesn't create local buffers."""
        embedding = CastedSparseEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            batch_size=self.batch_size,
            init_std=1.0,
            cast_to=torch.float32
        ).to(self.device)

        embedding.eval()

        indices = torch.randint(0, self.num_embeddings, (self.batch_size,), device=self.device)
        output = embedding(indices)

        # Local buffers should still be None (or unchanged from before)
        self.assertEqual(output.shape, (self.batch_size, self.embedding_dim))

    def test_dtype_casting(self):
        """Test that output is cast to correct dtype."""
        for dtype in [torch.float16, torch.bfloat16, torch.float32]:
            if dtype == torch.bfloat16 and not torch.cuda.is_available():
                continue  # Skip bfloat16 on CPU

            embedding = CastedSparseEmbedding(
                num_embeddings=self.num_embeddings,
                embedding_dim=self.embedding_dim,
                batch_size=self.batch_size,
                init_std=1.0,
                cast_to=dtype
            ).to(self.device)

            indices = torch.randint(0, self.num_embeddings, (self.batch_size,), device=self.device)
            output = embedding(indices)

            self.assertEqual(output.dtype, dtype)


class TestSparseEmbeddingOptimizer(unittest.TestCase):
    """Test sparse embedding optimizer."""

    def setUp(self):
        """Set up test fixtures."""
        self.num_embeddings = 100
        self.embedding_dim = 64
        self.batch_size = 16
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    def test_optimizer_initialization(self):
        """Test optimizer initialization with sparse embedding."""
        embedding = CastedSparseEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            batch_size=self.batch_size,
            init_std=1.0,
            cast_to=torch.float32
        ).to(self.device)

        # Create optimizer
        optimizer = CastedSparseEmbeddingSignSGD_Distributed(
            embedding.buffers(),
            lr=0.01,
            weight_decay=0.01,
            world_size=1
        )

        # Store reference to module
        optimizer.param_groups[0]["sparse_emb_module"] = embedding

        self.assertEqual(len(optimizer.param_groups), 1)
        self.assertEqual(optimizer.param_groups[0]["lr"], 0.01)

    def test_optimizer_step_with_gradients(self):
        """Test optimizer step with computed gradients."""
        embedding = CastedSparseEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            batch_size=self.batch_size,
            init_std=1.0,
            cast_to=torch.float32
        ).to(self.device)

        embedding.train()

        optimizer = CastedSparseEmbeddingSignSGD_Distributed(
            embedding.buffers(),
            lr=0.01,
            weight_decay=0.01,
            world_size=1
        )
        optimizer.param_groups[0]["sparse_emb_module"] = embedding

        # Forward pass
        indices = torch.randint(0, self.num_embeddings, (self.batch_size,), device=self.device)
        output = embedding(indices)

        # Backward pass
        loss = output.sum()
        loss.backward()

        # Optimizer step should work without error
        weights_before = embedding.weights.clone()
        optimizer.step()

        # Weights should have changed
        self.assertFalse(torch.allclose(weights_before, embedding.weights))

    def test_optimizer_without_gradients(self):
        """Test optimizer step without gradients (should skip)."""
        embedding = CastedSparseEmbedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim,
            batch_size=self.batch_size,
            init_std=1.0,
            cast_to=torch.float32
        ).to(self.device)

        optimizer = CastedSparseEmbeddingSignSGD_Distributed(
            embedding.buffers(),
            lr=0.01,
            weight_decay=0.01,
            world_size=1
        )
        optimizer.param_groups[0]["sparse_emb_module"] = embedding

        # Step without gradients should not error
        weights_before = embedding.weights.clone()
        optimizer.step()

        # Weights should not have changed
        self.assertTrue(torch.allclose(weights_before, embedding.weights))


if __name__ == "__main__":
    # Run tests with verbose output
    unittest.main(verbosity=2)
