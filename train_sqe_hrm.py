#!/usr/bin/env python3
"""
SQE-HRM Training Script
=======================

Training script for the SQE-Enhanced Hierarchical Reasoning Model
with comprehensive gradient flow preservation, memory management,
and enhanced supervision.

Usage:
    python train_sqe_hrm.py --config config/sqe_enhanced.yaml
    python train_sqe_hrm.py --config config/sqe_enhanced.yaml --dataset sudoku
    python train_sqe_hrm.py --config config/sqe_enhanced.yaml --dataset maze

Author: Ian Shank
Date: October 2025
"""

import argparse
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import OmegaConf
import wandb

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Import base HRM components
# Import our own versions of these functions to avoid import errors
# Alternatively, you could use the existing ones from the base HRM
def get_device():
    """Get appropriate device for training"""
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_world_info():
    """Get world size and rank for distributed training"""
    # Default values for non-distributed
    world_size = 1
    rank = 0
    
    # Check for distributed training
    if torch.distributed.is_initialized():
        world_size = torch.distributed.get_world_size()
        rank = torch.distributed.get_rank()
    
    return world_size, rank

def cast_to_precision(model, precision="full", device=None):
    """Cast model to appropriate precision"""
    if device is not None:
        model = model.to(device)
    
    if precision == "bf16" and torch.cuda.is_available() and torch.cuda.is_bf16_supported():
        model = model.to(torch.bfloat16)
    elif precision == "fp16" and torch.cuda.is_available():
        model = model.to(torch.float16)
    
    return model

# Import data loader from puzzle_dataset
sys.path.insert(0, str(Path(__file__).resolve().parent))
from puzzle_dataset import PuzzleDataset

def get_data_loader(data_path, batch_size, seq_len, world_size=1, rank=0, config=None):
    """Create data loaders for training and validation"""
    # Create custom dataset implementation that implements __len__
    class HRMDataset(torch.utils.data.Dataset):
        def __init__(self, path, seq_len):
            self.examples = []
            self.path = path
            self.seq_len = seq_len
            self._load_data()
            
        def _load_data(self):
            # Simple implementation to load puzzles from data_path
            import glob
            import json
            
            puzzle_files = glob.glob(os.path.join(self.path, "*.json"))
            for pfile in puzzle_files:
                with open(pfile, 'r') as f:
                    self.examples.append(json.load(f))
                    
        def __len__(self):
            return len(self.examples)
            
        def __getitem__(self, idx):
            # Simple implementation - in practice, use proper tokenization
            example = self.examples[idx]
            
            # Convert to tensor format expected by the model
            inputs = torch.zeros(self.seq_len, dtype=torch.long)
            targets = torch.ones(self.seq_len, dtype=torch.long) * -100
            
            # Fill with real data when available
            if "inputs" in example and len(example["inputs"]) > 0:
                length = min(len(example["inputs"]), self.seq_len)
                inputs[:length] = torch.tensor(example["inputs"][:length])
                
            if "targets" in example and len(example["targets"]) > 0:
                length = min(len(example["targets"]), self.seq_len)
                targets[:length] = torch.tensor(example["targets"][:length])
            
            # Add puzzle identifier
            puzzle_id = example.get("puzzle_id", 0)
            
            return {
                "inputs": inputs,
                "targets": targets,
                "puzzle_identifiers": torch.tensor([puzzle_id], dtype=torch.long)
            }
    
    # Create dataset
    dataset = HRMDataset(path=data_path, seq_len=seq_len)
    
    # Split into train and validation
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )
    
    # Create samplers for distributed training
    train_sampler = torch.utils.data.DistributedSampler(
        train_dataset, 
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
    ) if world_size > 1 else None
    
    val_sampler = torch.utils.data.DistributedSampler(
        val_dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=False,
    ) if world_size > 1 else None
    
    # Create data loaders
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=2,
        pin_memory=True,
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=2,
        pin_memory=True,
    )
    
    return train_loader, val_loader

def load_config_and_args(default_config_path):
    """Load configuration from file and command line arguments"""
    # Parse command line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=default_config_path,
                       help="Path to YAML config file")
    args, unknown = parser.parse_known_args()
    
    # Load configuration
    cfg = OmegaConf.load(args.config)
    
    # Update with command line arguments
    if unknown:
        override_conf = OmegaConf.from_dotlist(unknown)
        cfg = OmegaConf.merge(cfg, override_conf)
    
    return cfg

# Import SQE integration
from sqe_integration import SQEEnhancedHRM

# Setup logging
logging.basicConfig(level=logging.INFO, 
                    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_optimizer(model: SQEEnhancedHRM, cfg) -> Dict:
    """
    Set up optimizers with specialized parameter groups for SQE components
    
    Args:
        model: The SQE-HRM model
        cfg: Configuration object
        
    Returns:
        Dictionary containing optimizer and scheduler
    """
    # Parameter groups with different learning rates
    param_groups = [
        # Base HRM parameters
        {
            "params": [p for n, p in model.hrm.named_parameters() 
                      if "puzzle_emb" not in n and p.requires_grad],
            "lr": cfg.training.lr,
            "weight_decay": cfg.training.weight_decay
        },
        # Puzzle embedding parameters
        {
            "params": [p for n, p in model.hrm.named_parameters() 
                      if "puzzle_emb" in n and p.requires_grad],
            "lr": cfg.training.puzzle_emb_lr,
            "weight_decay": cfg.training.puzzle_emb_weight_decay
        },
        # SQE-specific parameters
        {
            "params": [p for n, p in model.named_parameters() 
                      if n.startswith("sqe_") and p.requires_grad],
            "lr": cfg.sqe.sqe_lr,
            "weight_decay": cfg.sqe.sqe_weight_decay
        }
    ]
    
    # Create optimizer
    optimizer = torch.optim.AdamW(param_groups)
    
    # Create scheduler
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=cfg.training.epochs
    )
    
    return {
        "optimizer": optimizer,
        "scheduler": scheduler
    }


def train_epoch(model: SQEEnhancedHRM, 
                data_loader, 
                optimizer, 
                device, 
                cfg,
                epoch: int) -> Dict:
    """
    Train a single epoch with gradient preservation and memory management
    
    Args:
        model: The SQE-HRM model
        data_loader: DataLoader for training data
        optimizer: Model optimizer
        device: Training device
        cfg: Configuration object
        epoch: Current epoch number
        
    Returns:
        Dictionary of training metrics
    """
    model.train()
    
    # Track metrics
    metrics = {
        "loss": 0.0,
        "samples": 0,
        "batches": 0
    }
    
    # Gradient scaler for mixed precision
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.training.use_mixed_precision)
    
    # Configure batch size and accumulation
    grad_acc_steps = cfg.training.gradient_accumulation_steps
    total_batches = len(data_loader)
    
    # Initialize carry state
    carry = None
    
    # Training loop
    for batch_idx, batch in enumerate(data_loader):
        # Move batch to device
        batch = {k: v.to(device) for k, v in batch.items()}
        
        # Initialize or update carry state
        if carry is None:
            carry = model.initial_carry(batch)
        
        # Mixed precision forward pass
        with torch.cuda.amp.autocast(enabled=cfg.training.use_mixed_precision):
            # Forward pass
            new_carry, outputs = model(carry, batch)
            carry = new_carry
            
            # Compute loss (use enhanced logits if available)
            logits = outputs.get("enhanced_logits", outputs["logits"])
            targets = batch["targets"]
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
            
            # Scale loss for gradient accumulation
            loss = loss / grad_acc_steps
        
        # Backward pass with mixed precision
        scaler.scale(loss).backward()
        
        # Only step optimizer after accumulating gradients
        if (batch_idx + 1) % grad_acc_steps == 0 or batch_idx == total_batches - 1:
            # Clip gradients
            if cfg.training.gradient_clip_norm > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    model.parameters(), 
                    cfg.training.gradient_clip_norm
                )
            
            # Optimizer step
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
        
        # Update metrics
        metrics["loss"] += loss.item() * grad_acc_steps
        metrics["samples"] += batch["inputs"].size(0)
        metrics["batches"] += 1
        
        # Memory cleanup
        if cfg.memory.cleanup_frequency > 0 and batch_idx % cfg.memory.cleanup_frequency == 0:
            torch.cuda.empty_cache()
        
        # Log progress
        if batch_idx % 10 == 0:
            logger.info(f"Epoch {epoch}, Batch {batch_idx}/{total_batches}, "
                       f"Loss: {metrics['loss'] / metrics['batches']:.4f}")
    
    # Compute final metrics
    metrics["loss"] /= metrics["batches"]
    
    return metrics


def evaluate(model: SQEEnhancedHRM, data_loader, device) -> Dict:
    """
    Evaluate model performance
    
    Args:
        model: The SQE-HRM model
        data_loader: DataLoader for evaluation data
        device: Evaluation device
        
    Returns:
        Dictionary of evaluation metrics
    """
    model.eval()
    
    # Track metrics
    metrics = {
        "loss": 0.0,
        "correct": 0,
        "total": 0,
        "samples": 0,
        "batches": 0
    }
    
    # Evaluation loop
    with torch.no_grad():
        carry = None
        
        for batch_idx, batch in enumerate(data_loader):
            # Move batch to device
            batch = {k: v.to(device) for k, v in batch.items()}
            
            # Initialize carry state if needed
            if carry is None:
                carry = model.initial_carry(batch)
            
            # Forward pass
            new_carry, outputs = model(carry, batch)
            carry = new_carry
            
            # Compute loss
            logits = outputs["logits"]
            targets = batch["targets"]
            
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100
            )
            
            # Calculate accuracy
            pred = logits.argmax(dim=-1)
            mask = targets != -100
            correct = (pred[mask] == targets[mask]).sum().item()
            total = mask.sum().item()
            
            # Update metrics
            metrics["loss"] += loss.item()
            metrics["correct"] += correct
            metrics["total"] += total
            metrics["samples"] += batch["inputs"].size(0)
            metrics["batches"] += 1
    
    # Compute final metrics
    metrics["loss"] /= metrics["batches"]
    metrics["accuracy"] = metrics["correct"] / max(metrics["total"], 1)
    
    return metrics


def main():
    """Main training function"""
    # Load configuration
    cfg = load_config_and_args(default_config_path="config/sqe_enhanced.yaml")
    
    # Parse additional arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default="sudoku", 
                       choices=["sudoku", "maze", "arc"],
                       help="Dataset to train on")
    args, unknown = parser.parse_known_args()
    
    # Setup device and distributed training
    device = get_device()
    world_size, rank = get_world_info()
    logger.info(f"Using device: {device}, world_size: {world_size}, rank: {rank}")
    
    # Setup dataset
    if args.dataset == "sudoku":
        data_path = "data/sudoku-extreme-1k-aug-1000"
    elif args.dataset == "maze":
        data_path = "data/maze-30x30-hard-1k"
    else:  # arc
        data_path = "data/arc-2-aug-1000"
    
    # Create data loaders
    train_loader, valid_loader = get_data_loader(
        data_path=data_path,
        batch_size=cfg.arch.batch_size,
        seq_len=cfg.arch.seq_len,
        world_size=world_size,
        rank=rank,
        config=cfg
    )
    
    # Initialize model
    arch_config = OmegaConf.to_container(cfg.arch, resolve=True)
    if not isinstance(arch_config, dict):
        raise ValueError("arch config must be a dictionary")
    model = SQEEnhancedHRM(arch_config)
    model = cast_to_precision(model, precision="full", device=device)
    
    # Setup optimizer
    optim_dict = setup_optimizer(model, cfg)
    optimizer = optim_dict["optimizer"]
    scheduler = optim_dict["scheduler"]
    
    # Initialize WandB
    if rank == 0:
        # Convert OmegaConf to plain Python dict for wandb
        wandb_config_raw = OmegaConf.to_container(cfg, resolve=True)
        if not isinstance(wandb_config_raw, dict):
            wandb_config_raw = {}
        # Convert keys to strings to ensure compatibility
        wandb_config: Dict[str, Any] = {str(k): v for k, v in wandb_config_raw.items()}
        
        wandb.init(
            project="sqe-hrm",
            name=f"sqe-hrm-{args.dataset}",
            config=wandb_config
        )
    
    # Create output directory
    output_dir = Path("checkpoints") / f"sqe-hrm-{args.dataset}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    best_eval_loss = float('inf')
    
    for epoch in range(cfg.training.epochs):
        # Train
        train_metrics = train_epoch(
            model=model,
            data_loader=train_loader,
            optimizer=optimizer,
            device=device,
            cfg=cfg,
            epoch=epoch
        )
        
        # Evaluate periodically
        if epoch % cfg.training.eval_interval == 0 or epoch == cfg.training.epochs - 1:
            eval_metrics = evaluate(
                model=model,
                data_loader=valid_loader,
                device=device
            )
            
            # Log metrics
            if rank == 0:
                logger.info(f"Epoch {epoch}, Train Loss: {train_metrics['loss']:.4f}, "
                           f"Eval Loss: {eval_metrics['loss']:.4f}, "
                           f"Eval Accuracy: {eval_metrics['accuracy']:.4f}")
                
                wandb.log({
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "eval/loss": eval_metrics["loss"],
                    "eval/accuracy": eval_metrics["accuracy"]
                })
                
                # Save best model
                if eval_metrics["loss"] < best_eval_loss:
                    best_eval_loss = eval_metrics["loss"]
                    torch.save({
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "eval_metrics": eval_metrics
                    }, output_dir / "best_model.pt")
        
        # Update learning rate
        scheduler.step()
    
    # Save final model
    if rank == 0:
        torch.save({
            "epoch": cfg.training.epochs - 1,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict()
        }, output_dir / "final_model.pt")
        
        wandb.finish()


if __name__ == "__main__":
    main()
