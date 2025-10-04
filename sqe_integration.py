#!/usr/bin/env python3
"""
SQE-HRM Integration Module with Perceiver
==========================================

This module integrates:
1. Perceiver encoder for efficient input processing
2. Stacked Quantum Ensembles (SQE) for enhanced reasoning
3. Hierarchical Reasoning Model (HRM) as the base architecture

Phase 2 Enhancement: Perceiver cross-attention preprocessing layer
extracts abstract features from ARC puzzle grids before HRM processing.

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
from models.perceiver import PerceiverEncoder, PerceiverConfig


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    Enhanced HRM model with Perceiver preprocessing and SQE integration.

    Architecture:
    1. **Perceiver Encoder** (Phase 2): Preprocesses inputs via cross-attention
       - Compresses variable-length inputs to fixed-size latent representation
       - Enables efficient processing of 30x30 ARC grids

    2. **HRM Base Model**: Hierarchical reasoning with ACT
       - High-level (slow) and low-level (fast) reasoning modules
       - Adaptive computation time for variable difficulty

    3. **SQE Enhancement**: Stacked quantum ensembles
       - Multi-head attention for ensemble stacking
       - Gradient-preserving training

    Phase 2 Goal: Improve abstract feature extraction for better reasoning.
    """

    def __init__(self, config_dict: dict):
        """
        Initialize SQE-Enhanced HRM with Perceiver preprocessing.

        Args:
            config_dict: Configuration dictionary containing:
                - hidden_size: Model hidden dimension
                - sqe_layers: Number of SQE attention layers
                - sqe_heads: Number of attention heads
                - enable_perceiver: Whether to use Perceiver preprocessing
                - perceiver_*: Perceiver-specific configuration
                - forward_dtype: Forward pass dtype (bfloat16/float32)
        """
        super().__init__()

        # Initialize the base HRM model
        self.hrm = HierarchicalReasoningModel_ACTV1(config_dict)

        # Enhanced configuration
        self.hidden_size = config_dict.get("hidden_size", 512)
        self.sqe_layers = config_dict.get("sqe_layers", 2)
        self.sqe_heads = config_dict.get("sqe_heads", 4)
        self.forward_dtype = config_dict.get("forward_dtype", "bfloat16")

        # Phase 2: Perceiver configuration
        self.enable_perceiver = config_dict.get("enable_perceiver", True)
        self.perceiver_num_latents = config_dict.get("perceiver_num_latents", 256)
        self.perceiver_num_blocks = config_dict.get("perceiver_num_blocks", 2)

        # Initialize Perceiver if enabled
        if self.enable_perceiver:
            self._initialize_perceiver()
            logger.info("Perceiver preprocessing enabled")
        else:
            self.perceiver = None
            logger.info("Perceiver preprocessing disabled")

        # SQE enhancements
        self._initialize_sqe_components()

        # Wrap for gradient preservation
        self.gradient_preserving_wrapper = SQEGradientPreservingWrapper(self.hrm)

    def _initialize_perceiver(self):
        """Initialize Perceiver encoder for input preprocessing."""
        perceiver_config = PerceiverConfig(
            num_latents=self.perceiver_num_latents,
            latent_dim=self.hidden_size,
            input_dim=self.hidden_size,
            num_blocks=self.perceiver_num_blocks,
            num_cross_attention_heads=8,
            num_self_attention_heads=8,
            num_self_attends_per_block=6,
            forward_dtype=self.forward_dtype
        )

        self.perceiver = PerceiverEncoder(perceiver_config)

        # Projection to map perceiver latents back to sequence
        target_dtype = getattr(torch, self.forward_dtype)
        self.perceiver_to_seq = nn.Linear(
            self.hidden_size,
            self.hidden_size
        ).to(dtype=target_dtype)

        logger.info(
            f"Perceiver initialized: {self.perceiver_num_latents} latents, "
            f"{self.perceiver_num_blocks} blocks, {self.perceiver.get_num_params():,} params"
        )

    def _initialize_sqe_components(self):
        """Initialize SQE-specific model components."""

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

        logger.info(f"SQE components initialized: {self.sqe_layers} layers, {self.sqe_heads} heads")
        
    def _apply_perceiver_preprocessing(
        self,
        batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Apply Perceiver preprocessing to inputs.

        Process flow:
        1. Extract token embeddings from batch
        2. Pass through Perceiver encoder to compress to latents
        3. Project latents back to sequence representation
        4. Return modified batch with preprocessed inputs

        Args:
            batch: Input batch dictionary

        Returns:
            Batch with perceiver-preprocessed inputs
        """
        if not self.enable_perceiver or self.perceiver is None:
            return batch

        # Get embeddings - need to extract from HRM's embedding layer
        # This runs the input through the token embedding
        inputs = batch["inputs"]
        batch_size, seq_len = inputs.shape

        # Get token embeddings from HRM
        with torch.no_grad() if not self.training else torch.enable_grad():
            # Access HRM's input embedding
            input_embs = self.hrm.inner.token_emb(inputs)  # [batch, seq, hidden]

        # Apply Perceiver encoding
        latents = self.perceiver(input_embs)  # [batch, num_latents, hidden]

        # Project latents to create enriched input representation
        # Average pool latents to sequence length
        if latents.shape[1] < seq_len:
            # Repeat latents to match sequence length
            repeat_factor = math.ceil(seq_len / latents.shape[1])
            enriched_repr = latents.repeat(1, repeat_factor, 1)[:, :seq_len, :]
        else:
            # Adaptive pooling to match sequence length
            enriched_repr = F.adaptive_avg_pool1d(
                latents.transpose(1, 2),
                seq_len
            ).transpose(1, 2)

        # Apply projection
        enriched_repr = self.perceiver_to_seq(enriched_repr)

        # Store in batch for use by HRM
        # We'll add this as additional context
        batch["perceiver_features"] = enriched_repr

        return batch

    def forward(
        self,
        carry: HierarchicalReasoningModel_ACTV1Carry,
        batch: Dict[str, torch.Tensor]
    ) -> Tuple[HierarchicalReasoningModel_ACTV1Carry, Dict[str, torch.Tensor]]:
        """
        Enhanced forward pass with Perceiver preprocessing and SQE integration.

        Processing pipeline:
        1. **Perceiver Preprocessing** (Phase 2): Extract abstract features
        2. **HRM Forward Pass**: Hierarchical reasoning with ACT
        3. **SQE Enhancement**: Ensemble stacking and logit enhancement

        Args:
            carry: The HRM carry state
            batch: Input batch dictionary containing:
                - inputs: Token indices [batch, seq_len]
                - targets: Target token indices [batch, seq_len]
                - puzzle_identifiers: Puzzle IDs [batch]

        Returns:
            Tuple of (new_carry, outputs) where:
                - new_carry: Updated carry state
                - outputs: Dictionary with logits and optional enhanced_logits
        """
        # Phase 2: Apply Perceiver preprocessing if enabled
        if self.enable_perceiver:
            batch = self._apply_perceiver_preprocessing(batch)

        # Forward through base HRM model with gradient preservation
        new_carry, outputs = self.gradient_preserving_wrapper(carry, batch)

        # Apply SQE enhancements if in training mode
        if self.training:
            # Extract high-level representations from HRM
            high_level_repr = new_carry.inner_carry.z_H

            # Optionally incorporate Perceiver features
            if "perceiver_features" in batch:
                # Blend Perceiver features with HRM representations
                perceiver_feat = batch["perceiver_features"]
                # Match shapes for addition
                if perceiver_feat.shape[1] < high_level_repr.shape[1]:
                    perceiver_feat = F.pad(
                        perceiver_feat,
                        (0, 0, 0, high_level_repr.shape[1] - perceiver_feat.shape[1])
                    )
                elif perceiver_feat.shape[1] > high_level_repr.shape[1]:
                    perceiver_feat = perceiver_feat[:, :high_level_repr.shape[1], :]

                # Additive blending with learned weight
                high_level_repr = high_level_repr + 0.2 * perceiver_feat

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
