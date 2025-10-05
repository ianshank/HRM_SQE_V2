#!/usr/bin/env python3
"""
Test script to reproduce the embedding indexing bug.
"""

import torch
import numpy as np
import json
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.sparse_embedding import CastedSparseEmbedding

def test_embedding_with_actual_data():
    """Test embedding with actual dataset parameters."""

    # Load actual dataset metadata
    metadata_path = "c:/Users/iansh/Documents/HRM/data/arc-aug-100/train/dataset.json"
    with open(metadata_path) as f:
        metadata = json.load(f)

    num_puzzle_identifiers = metadata['num_puzzle_identifiers']
    blank_identifier_id = metadata['blank_identifier_id']

    print(f"num_puzzle_identifiers: {num_puzzle_identifiers}")
    print(f"blank_identifier_id: {blank_identifier_id}")

    # Load actual puzzle identifiers
    ids_path = "c:/Users/iansh/Documents/HRM/data/arc-aug-100/train/all__puzzle_identifiers.npy"
    puzzle_ids = np.load(ids_path)

    print(f"Puzzle IDs shape: {puzzle_ids.shape}")
    print(f"Puzzle IDs range: [{puzzle_ids.min()}, {puzzle_ids.max()}]")

    # Create embedding with exact same size as model
    batch_size = 16
    puzzle_emb_ndim = 64  # from config

    embedding = CastedSparseEmbedding(
        num_embeddings=num_puzzle_identifiers,
        embedding_dim=puzzle_emb_ndim,
        batch_size=batch_size,
        init_std=0,
        cast_to=torch.float32
    )

    if torch.cuda.is_available():
        embedding = embedding.cuda()
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    # Test 1: Sample from actual puzzle IDs
    print("\nTest 1: Sample from actual puzzle IDs")
    sample_ids = torch.from_numpy(np.random.choice(puzzle_ids, size=batch_size)).to(device)
    print(f"Sample IDs: min={sample_ids.min()}, max={sample_ids.max()}")

    try:
        output = embedding(sample_ids)
        print(f"[OK] Test 1 passed: output shape = {output.shape}")
    except Exception as e:
        print(f"[FAIL] Test 1 failed: {e}")
        return False

    # Test 2: Test with blank_identifier_id (padding)
    print("\nTest 2: Test with blank_identifier_id (padding)")
    blank_ids = torch.full((batch_size,), blank_identifier_id, dtype=torch.long, device=device)
    print(f"Blank IDs: {blank_ids[0]}")

    try:
        output = embedding(blank_ids)
        print(f"[OK] Test 2 passed: output shape = {output.shape}")
    except Exception as e:
        print(f"[FAIL] Test 2 failed: {e}")
        return False

    # Test 3: Test with maximum valid index
    print("\nTest 3: Test with maximum valid index")
    max_valid_idx = num_puzzle_identifiers - 1
    max_ids = torch.full((batch_size,), max_valid_idx, dtype=torch.long, device=device)
    print(f"Max valid index: {max_valid_idx}")

    try:
        output = embedding(max_ids)
        print(f"[OK] Test 3 passed: output shape = {output.shape}")
    except Exception as e:
        print(f"[FAIL] Test 3 failed: {e}")
        return False

    # Test 4: Test with out-of-bounds index (should fail)
    print("\nTest 4: Test with out-of-bounds index (should fail)")
    oob_ids = torch.full((batch_size,), num_puzzle_identifiers, dtype=torch.long, device=device)
    print(f"Out-of-bounds index: {num_puzzle_identifiers}")

    try:
        output = embedding(oob_ids)
        print(f"[FAIL] Test 4 failed: Should have raised IndexError but didn't")
        return False
    except IndexError as e:
        print(f"[OK] Test 4 passed: Correctly caught IndexError: {e}")
    except RuntimeError as e:
        print(f"[FAIL] Test 4 failed with RuntimeError (CUDA assertion): {e}")
        return False

    # Test 5: Mixed batch with padding
    print("\nTest 5: Mixed batch with padding")
    mixed_ids = torch.cat([
        torch.from_numpy(np.random.choice(puzzle_ids, size=batch_size//2)),
        torch.full((batch_size//2,), blank_identifier_id, dtype=torch.long)
    ]).to(device)
    print(f"Mixed IDs: min={mixed_ids.min()}, max={mixed_ids.max()}")

    try:
        output = embedding(mixed_ids)
        print(f"[OK] Test 5 passed: output shape = {output.shape}")
    except Exception as e:
        print(f"[FAIL] Test 5 failed: {e}")
        return False

    print("\n" + "="*60)
    print("ALL TESTS PASSED")
    print("="*60)
    return True


if __name__ == "__main__":
    success = test_embedding_with_actual_data()
    sys.exit(0 if success else 1)
