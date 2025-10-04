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

def main():
    """Run all tests"""
    print("=" * 60)
    print("SQE-HRM Integration Test Suite")
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
        
        # Summary
        print("\n" + "=" * 60)
        print("Test Summary")
        print("=" * 60)
        print(f"✓ Model Initialization: PASSED")
        print(f"✓ Forward Pass: PASSED")
        print(f"{'✓' if grad_ok else '✗'} Gradient Flow: {'PASSED' if grad_ok else 'FAILED'}")
        print(f"{'✓' if sqe_ok else '✗'} SQE Components: {'PASSED' if sqe_ok else 'FAILED'}")
        
        if grad_ok and sqe_ok:
            print("\n🎉 All tests passed! SQE-HRM integration is working correctly.")
            return 0
        else:
            print("\n⚠️  Some tests failed. Please review the errors above.")
            return 1
            
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit(main())

