#!/usr/bin/env python3
"""
Simple test script to validate SQE-HRM integration
"""

import sys
from pathlib import Path
import torch
from omegaconf import OmegaConf

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from sqe_integration import SQEEnhancedHRM

def test_model_initialization():
    """Test that the SQE-HRM model can be initialized"""
    print("Testing model initialization...")
    
    # Load config
    config_path = "config/sqe_enhanced.yaml"
    cfg = OmegaConf.load(config_path)
    
    # Convert to dict
    arch_config = OmegaConf.to_container(cfg.arch, resolve=True)
    if not isinstance(arch_config, dict):
        raise ValueError("arch config must be a dictionary")
    
    # Initialize model
    model = SQEEnhancedHRM(arch_config)
    
    print(f"✓ Model initialized successfully")
    print(f"  - Model type: {type(model).__name__}")
    print(f"  - Hidden size: {model.hidden_size}")
    print(f"  - SQE layers: {model.sqe_layers}")
    print(f"  - SQE heads: {model.sqe_heads}")
    
    return model

def test_forward_pass(model):
    """Test a simple forward pass"""
    print("\nTesting forward pass...")
    
    batch_size = 4
    seq_len = 512  # Must match config.arch.seq_len
    
    # Create dummy batch
    batch = {
        "inputs": torch.randint(0, 100, (batch_size, seq_len)),
        "targets": torch.randint(0, 100, (batch_size, seq_len)),
        "puzzle_identifiers": torch.randint(0, 1000, (batch_size,))
    }
    
    # Initialize carry
    carry = model.initial_carry(batch)
    print(f"✓ Carry initialized")
    
    # Forward pass
    model.eval()
    with torch.no_grad():
        new_carry, outputs = model(carry, batch)
    
    print(f"✓ Forward pass successful")
    print(f"  - Output logits shape: {outputs['logits'].shape}")
    print(f"  - Expected shape: ({batch_size}, {seq_len}, vocab_size)")
    
    # Check for enhanced outputs
    if "enhanced_logits" in outputs:
        print(f"  - Enhanced logits available: {outputs['enhanced_logits'].shape}")
    
    return outputs

def test_gradient_flow(model):
    """Test that gradients flow properly during training"""
    print("\nTesting gradient flow...")
    
    model.train()
    
    batch_size = 2
    seq_len = 512  # Must match config.arch.seq_len
    
    # Create dummy batch
    batch = {
        "inputs": torch.randint(0, 100, (batch_size, seq_len)),
        "targets": torch.randint(0, 100, (batch_size, seq_len)),
        "puzzle_identifiers": torch.randint(0, 1000, (batch_size,))
    }
    
    # Initialize carry
    carry = model.initial_carry(batch)
    
    # Forward pass
    new_carry, outputs = model(carry, batch)
    
    # Compute loss
    logits = outputs.get("enhanced_logits", outputs["logits"])
    targets = batch["targets"]
    
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, logits.size(-1)),
        targets.view(-1)
    )
    
    print(f"✓ Loss computed: {loss.item():.4f}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    has_grad = False
    grad_norms = []
    
    for name, param in model.named_parameters():
        if param.grad is not None:
            has_grad = True
            grad_norm = param.grad.norm().item()
            grad_norms.append(grad_norm)
    
    if has_grad:
        print(f"✓ Gradients computed successfully")
        print(f"  - Parameters with gradients: {len(grad_norms)}")
        print(f"  - Mean gradient norm: {sum(grad_norms)/len(grad_norms):.6f}")
        print(f"  - Max gradient norm: {max(grad_norms):.6f}")
    else:
        print(f"✗ No gradients found!")
        return False
    
    return True

def test_sqe_components(model):
    """Test SQE-specific components"""
    print("\nTesting SQE components...")

    # Check SQE layers exist
    assert hasattr(model, 'sqe_projection'), "Missing sqe_projection"
    assert hasattr(model, 'sqe_norm'), "Missing sqe_norm"
    assert hasattr(model, 'sqe_attention_layers'), "Missing sqe_attention_layers"

    print(f"✓ SQE components present")
    print(f"  - Projection layer: {model.sqe_projection}")
    print(f"  - Normalization: {model.sqe_norm}")
    print(f"  - Attention layers: {len(model.sqe_attention_layers)}")

    return True


def test_perceiver_integration(model):
    """Test Perceiver integration (Phase 2)"""
    print("\nTesting Perceiver integration...")

    if not model.enable_perceiver:
        print("  ⚠ Perceiver disabled in config, skipping...")
        return True

    # Check Perceiver components exist
    assert hasattr(model, 'perceiver'), "Missing perceiver"
    assert model.perceiver is not None, "Perceiver is None"
    assert hasattr(model, 'perceiver_to_seq'), "Missing perceiver_to_seq projection"

    print(f"✓ Perceiver components present")
    print(f"  - Num latents: {model.perceiver_num_latents}")
    print(f"  - Num blocks: {model.perceiver_num_blocks}")
    print(f"  - Perceiver params: {model.perceiver.get_num_params():,}")

    # Test Perceiver forward pass
    batch_size = 2
    seq_len = 512

    batch = {
        "inputs": torch.randint(0, 100, (batch_size, seq_len)),
        "targets": torch.randint(0, 100, (batch_size, seq_len)),
        "puzzle_identifiers": torch.randint(0, 1000, (batch_size,))
    }

    # Test preprocessing method
    model.eval()
    with torch.no_grad():
        preprocessed_batch = model._apply_perceiver_preprocessing(batch)

    # Check perceiver features were added
    if model.enable_perceiver:
        assert "perceiver_features" in preprocessed_batch, "Missing perceiver_features in batch"
        feat_shape = preprocessed_batch["perceiver_features"].shape
        print(f"✓ Perceiver preprocessing successful")
        print(f"  - Feature shape: {feat_shape}")
        assert feat_shape[0] == batch_size, "Batch size mismatch"
        assert feat_shape[1] == seq_len, "Sequence length mismatch"

    return True

def main():
    """Run all tests"""
    print("=" * 60)
    print("SQE-HRM + Perceiver Integration Test Suite (Phase 2)")
    print("=" * 60)

    try:
        # Test 1: Model initialization
        model = test_model_initialization()

        # Test 2: Forward pass
        outputs = test_forward_pass(model)

        # Test 3: Gradient flow
        grad_ok = test_gradient_flow(model)

        # Test 4: SQE components
        sqe_ok = test_sqe_components(model)

        # Test 5: Perceiver integration (Phase 2)
        perceiver_ok = test_perceiver_integration(model)

        # Summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"✓ Model Initialization: PASSED")
        print(f"✓ Forward Pass: PASSED")
        print(f"{'✓' if grad_ok else '✗'} Gradient Flow: {'PASSED' if grad_ok else 'FAILED'}")
        print(f"{'✓' if sqe_ok else '✗'} SQE Components: {'PASSED' if sqe_ok else 'FAILED'}")
        print(f"{'✓' if perceiver_ok else '✗'} Perceiver Integration: {'PASSED' if perceiver_ok else 'FAILED'}")

        all_passed = grad_ok and sqe_ok and perceiver_ok

        if all_passed:
            print("\nAll tests passed! Phase 2 integration is working correctly.")
            return 0
        else:
            print("\nSome tests failed. Please review the errors above.")
            return 1

    except Exception as e:
        print(f"\nTest failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

