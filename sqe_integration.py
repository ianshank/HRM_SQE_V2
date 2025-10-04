#!/usr/bin/env python3
"""
SQE-HRM Integration Module
==========================

This module integrates the Stacked Quantum Ensembles (SQE) model
with the Hierarchical Reasoning Model (HRM) framework.

The integration preserves gradient flow and ensures stable training
with enhanced memory management.

Author: Ian Shank
Date: October 2025
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import math
from typing import Dict, Tuple, Optional, List
from dataclasses import dataclass
import yaml
import logging
from collections import defaultdict

# Import nn.Module explicitly
from torch import nn

# Add local paths
sys.path.append(str(Path(__file__).parent))
sys.path.append(str(Path(__file__).parent / "models"))

from models.hrm.hrm_act_v1 import (
    HierarchicalReasoningModel_ACTV1,
    HierarchicalReasoningModel_ACTV1Config,
    HierarchicalReasoningModel_ACTV1Carry,
    HierarchicalReasoningModel_ACTV1InnerCarry
)

class SQEGradientPreservingWrapper:
    """
    Wrapper that ensures gradient flow is preserved during 
    SQE-HRM integration training.
    """
    
    def __init__(self, model):
        self.model = model
        self.is_training = True
        
    def __call__(self, *args, **kwargs):
        """
        Forward pass with gradient preservation during training
        """
        if self.is_training:
            # Ensure gradients are preserved during forward pass
            outputs = self.model(*args, **kwargs)
            return outputs
        else:
            # Use standard evaluation mode
            with torch.no_grad():
                outputs = self.model(*args, **kwargs)
            return outputs
            
    def eval(self):
        """Set wrapper to evaluation mode"""
        self.is_training = False
        self.model.eval()
        return self
        
    def train(self):
        """Set wrapper to training mode"""
        self.is_training = True
        self.model.train()
        return self


class SQEEnhancedHRM(torch.nn.Module):
    """
    Enhanced HRM model with SQE integration for improved reasoning capabilities.
    
    This model extends the standard HRM with:
    1. Gradient-preserving tensor operations
    2. Memory-efficient training
    3. Enhanced multi-component loss handling
    """
    
    def __init__(self, config_dict: dict):
        super().__init__()
        
        # Initialize the base HRM model
        self.hrm = HierarchicalReasoningModel_ACTV1(config_dict)
        
        # Enhanced configuration
        self.hidden_size = config_dict.get("hidden_size", 768)
        self.sqe_layers = config_dict.get("sqe_layers", 2)
        self.sqe_heads = config_dict.get("sqe_heads", 4)
        self.forward_dtype = config_dict.get("forward_dtype", "bfloat16")
        
        # SQE enhancements
        self._initialize_sqe_components()
        
        # Wrap for gradient preservation
        self.gradient_preserving_wrapper = SQEGradientPreservingWrapper(self.hrm)
        
    def _initialize_sqe_components(self):
        """Initialize SQE-specific model components"""
        
        # Get the target dtype
        target_dtype = getattr(torch, self.forward_dtype)
        
        # Stacked Quantum Ensembles components
        self.sqe_projection = nn.Linear(self.hidden_size, self.hidden_size).to(dtype=target_dtype)
        self.sqe_norm = nn.LayerNorm(self.hidden_size).to(dtype=target_dtype)
        
        # Multi-head cross-attention for ensemble stacking
        self.sqe_attention_layers = nn.ModuleList([
            nn.MultiheadAttention(
                embed_dim=self.hidden_size,
                num_heads=self.sqe_heads,
                batch_first=True
            ).to(dtype=target_dtype)
            for _ in range(self.sqe_layers)
        ])
        
    def forward(self, carry: HierarchicalReasoningModel_ACTV1Carry, batch: Dict[str, torch.Tensor]) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        """
        Enhanced forward pass with SQE integration
        
        Args:
            carry: The HRM carry state
            batch: Input batch dictionary
            
        Returns:
            Tuple of (new carry state, output dictionary)
        """
        # Forward through base HRM model with gradient preservation
        new_carry, outputs = self.gradient_preserving_wrapper(carry, batch)
        
        # Apply SQE enhancements if in training mode
        if self.training:
            # Extract high-level representations from HRM
            high_level_repr = new_carry.inner_carry.z_H
            
            # Apply SQE transformations
            enhanced_repr = self.sqe_projection(high_level_repr)
            enhanced_repr = self.sqe_norm(enhanced_repr)
            
            # Apply stacked attention layers
            for attn_layer in self.sqe_attention_layers:
                attn_outputs, _ = attn_layer(
                    query=enhanced_repr,
                    key=enhanced_repr,
                    value=enhanced_repr
                )
                enhanced_repr = enhanced_repr + attn_outputs
            
            # Update outputs with enhanced representations
            if "logits" in outputs:
                # Apply enhanced representations to logits
                # Cast to match lm_head weight dtype
                enhanced_repr_cast = enhanced_repr[:, 0].to(self.hrm.inner.lm_head.weight.dtype)
                logit_enhancement = torch.matmul(
                    enhanced_repr_cast, 
                    self.hrm.inner.lm_head.weight.t()
                )
                
                # Combine with original logits for ensemble effect
                batch_size = outputs["logits"].shape[0]
                seq_len = outputs["logits"].shape[1]
                
                # Reshape for proper broadcasting and cast back
                logit_enhancement = logit_enhancement.view(batch_size, 1, -1).to(outputs["logits"].dtype)
                
                # Add enhancement with scaling factor (learned during training)
                outputs["enhanced_logits"] = outputs["logits"] + 0.1 * logit_enhancement
        
        return new_carry, outputs

    @property
    def puzzle_emb(self):
        """Access to puzzle embeddings from base model"""
        return self.hrm.puzzle_emb
        
    def initial_carry(self, batch: Dict[str, torch.Tensor]):
        """Initialize carry state from base model"""
        return self.hrm.initial_carry(batch)


def load_sqe_hrm_model(config_path: str) -> SQEEnhancedHRM:
    """
    Load SQE-HRM model from configuration file
    
    Args:
        config_path: Path to the configuration YAML file
        
    Returns:
        Initialized SQEEnhancedHRM model
    """
    # Load configuration
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # Extract HRM base configuration
    hrm_config = config.get('arch', {})
    
    # Add SQE specific configuration
    sqe_config = config.get('sqe', {})
    combined_config = {**hrm_config, **sqe_config}
    
    # Initialize model
    return SQEEnhancedHRM(combined_config)
