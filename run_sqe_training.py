"""
Run SQE-HRM Training with Processed Data
========================================

Trains the SQE-Enhanced HRM model using data from the integration pipeline.

Usage:
    python run_sqe_training.py --data-dir data/processed
    python run_sqe_training.py --data-dir data/processed --epochs 10

Author: Data Integration Team
Date: October 2025
"""

import argparse
import sys
from pathlib import Path
import torch
import wandb
from omegaconf import OmegaConf

# Import SQE-HRM components
from sqe_integration import SQEEnhancedHRM
from hrm_data_integration.models.hrm_data import HRMDataset


def parse_args():
    parser = argparse.ArgumentParser(description="Train SQE-HRM with processed data")
    parser.add_argument("--data-dir", type=str, default="data/processed",
                       help="Directory containing train/val/test splits")
    parser.add_argument("--config", type=str, default="config/sqe_enhanced.yaml",
                       help="Path to HRM config file")
    parser.add_argument("--epochs", type=int, default=5,
                       help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8,
                       help="Training batch size")
    parser.add_argument("--lr", type=float, default=1e-4,
                       help="Learning rate")
    parser.add_argument("--checkpoint-dir", type=str, default="checkpoints",
                       help="Directory for saving checkpoints")
    parser.add_argument("--use-wandb", action="store_true",
                       help="Enable Weights & Biases logging")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu",
                       help="Device to use for training")
    return parser.parse_args()


def load_datasets(data_dir):
    """Load train/val/test datasets from processed JSONL files"""
    data_dir = Path(data_dir)
    
    print(f"\n📂 Loading datasets from {data_dir}...")
    
    train_dataset = HRMDataset.load(str(data_dir / "train.jsonl"))
    val_dataset = HRMDataset.load(str(data_dir / "val.jsonl"))
    test_dataset = HRMDataset.load(str(data_dir / "test.jsonl"))
    
    print(f"  ✓ Train: {len(train_dataset.examples)} examples")
    print(f"  ✓ Val:   {len(val_dataset.examples)} examples")
    print(f"  ✓ Test:  {len(test_dataset.examples)} examples")
    
    return train_dataset, val_dataset, test_dataset


def create_data_loaders(train_dataset, val_dataset, batch_size, max_length=512):
    """Create PyTorch data loaders"""
    from hrm_data_integration.models.hrm_data import HRMBatch
    
    def collate_fn(batch):
        """Collate examples into batches"""
        return HRMBatch.from_examples(batch, max_length=max_length).to_tensors()
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset.examples,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset.examples,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    
    return train_loader, val_loader


def train_epoch(model, train_loader, optimizer, device, epoch):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    num_batches = 0
    
    print(f"\n🔄 Epoch {epoch}:")
    
    for batch_idx, batch in enumerate(train_loader):
        # Move batch to device
        inputs = batch["inputs"].to(device)
        targets = batch["targets"].to(device)
        puzzle_ids = batch["puzzle_identifiers"].to(device)
        
        # Initialize carry
        batch_dict = {
            "inputs": inputs,
            "targets": targets,
            "puzzle_identifiers": puzzle_ids
        }
        
        carry = model.initial_carry(batch_dict)
        
        # Forward pass
        new_carry, outputs = model(carry, batch_dict)
        
        # Compute loss (simple cross-entropy on logits)
        if "logits" in outputs:
            logits = outputs["logits"]
            # Reshape for loss computation
            logits_flat = logits.view(-1, logits.size(-1))
            targets_flat = targets.view(-1)
            
            # Ignore padding (-100)
            loss = torch.nn.functional.cross_entropy(
                logits_flat, 
                targets_flat, 
                ignore_index=-100
            )
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 10 == 0:
                print(f"  Batch {batch_idx}/{len(train_loader)}, Loss: {loss.item():.4f}")
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f"  Average Loss: {avg_loss:.4f}")
    return avg_loss


def validate(model, val_loader, device):
    """Validate the model"""
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in val_loader:
            inputs = batch["inputs"].to(device)
            targets = batch["targets"].to(device)
            puzzle_ids = batch["puzzle_identifiers"].to(device)
            
            batch_dict = {
                "inputs": inputs,
                "targets": targets,
                "puzzle_identifiers": puzzle_ids
            }
            
            carry = model.initial_carry(batch_dict)
            new_carry, outputs = model(carry, batch_dict)
            
            if "logits" in outputs:
                logits = outputs["logits"]
                logits_flat = logits.view(-1, logits.size(-1))
                targets_flat = targets.view(-1)
                
                loss = torch.nn.functional.cross_entropy(
                    logits_flat,
                    targets_flat,
                    ignore_index=-100
                )
                
                total_loss += loss.item()
                num_batches += 1
    
    avg_loss = total_loss / num_batches if num_batches > 0 else 0
    print(f"  Validation Loss: {avg_loss:.4f}")
    return avg_loss


def main():
    args = parse_args()
    
    print("\n" + "="*80)
    print("SQE-HRM TRAINING")
    print("="*80)
    
    # Load configuration
    print(f"\n⚙️  Loading config from {args.config}...")
    cfg = OmegaConf.load(args.config)
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = load_datasets(args.data_dir)
    
    # Create data loaders
    print(f"\n🔧 Creating data loaders (batch_size={args.batch_size})...")
    train_loader, val_loader = create_data_loaders(
        train_dataset, 
        val_dataset, 
        batch_size=args.batch_size,
        max_length=cfg.arch.seq_len
    )
    
    # Initialize model
    print(f"\n🤖 Initializing SQE-Enhanced HRM model...")
    arch_config_raw = OmegaConf.to_container(cfg.arch, resolve=True)
    arch_config: dict = arch_config_raw if isinstance(arch_config_raw, dict) else {}
    model = SQEEnhancedHRM(arch_config)
    
    # Move to device
    device = torch.device(args.device)
    model = model.to(device)
    print(f"  Device: {device}")
    
    # Initialize optimizer
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    
    # Initialize W&B if requested
    if args.use_wandb:
        wandb.init(
            project="sqe-hrm-training",
            config={
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "learning_rate": args.lr,
                "train_size": len(train_dataset.examples),
                "val_size": len(val_dataset.examples)
            }
        )
    
    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    
    # Training loop
    print(f"\n🚀 Starting training for {args.epochs} epochs...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device)
        
        # Log to W&B
        if args.use_wandb:
            wandb.log({
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss
            })
        
        # Save checkpoint if best
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint_path = checkpoint_dir / "best_model.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
            }, checkpoint_path)
            print(f"  ✓ Saved best model checkpoint to {checkpoint_path}")
    
    print("\n" + "="*80)
    print("✅ TRAINING COMPLETE")
    print("="*80)
    print(f"\nBest Validation Loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {checkpoint_dir}")
    
    if args.use_wandb:
        wandb.finish()


if __name__ == "__main__":
    main()

