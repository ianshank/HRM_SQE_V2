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

# Import curriculum learning components
from dataset.curriculum_learning import create_curriculum_dataset


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
    parser.add_argument("--curriculum", action="store_true",
                       help="Enable curriculum learning with difficulty-based sorting")
    return parser.parse_args()


def load_datasets(data_dir, enable_curriculum=False, curriculum_config=None):
    """
    Load train/val/test datasets from processed JSONL files.

    Args:
        data_dir: Directory containing train/val/test splits
        enable_curriculum: Whether to wrap training set with curriculum learning
        curriculum_config: Curriculum configuration dict (from YAML)

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    data_dir = Path(data_dir)

    print(f"\n📂 Loading datasets from {data_dir}...")

    train_dataset = HRMDataset.load(str(data_dir / "train.jsonl"))
    val_dataset = HRMDataset.load(str(data_dir / "val.jsonl"))
    test_dataset = HRMDataset.load(str(data_dir / "test.jsonl"))

    print(f"  ✓ Train: {len(train_dataset.examples)} examples")
    print(f"  ✓ Val:   {len(val_dataset.examples)} examples")
    print(f"  ✓ Test:  {len(test_dataset.examples)} examples")

    # Wrap training set with curriculum learning if enabled
    if enable_curriculum:
        print(f"\n📚 Enabling curriculum learning...")

        # Wrap the examples list (which is the actual dataset)
        from torch.utils.data import TensorDataset

        # Create wrapper for the examples
        train_dataset_wrapped = create_curriculum_dataset(
            base_dataset=train_dataset.examples,
            curriculum_config=curriculum_config
        )

        # Replace examples with wrapped version
        # Store original for reference
        train_dataset._original_examples = train_dataset.examples
        train_dataset.examples = train_dataset_wrapped
        train_dataset.is_curriculum = True

        print(f"  ✓ Curriculum dataset: {len(train_dataset_wrapped)} / "
              f"{len(train_dataset._original_examples)} examples available at current threshold")

        # Print difficulty stats
        stats = train_dataset_wrapped.get_difficulty_stats()
        print(f"  ✓ Difficulty stats: mean={stats['mean']:.3f}, "
              f"std={stats['std']:.3f}, range=[{stats['min']:.3f}, {stats['max']:.3f}]")
    else:
        train_dataset.is_curriculum = False

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
    if args.curriculum:
        print("Mode: Curriculum Learning Enabled")
    print("="*80)

    # Load configuration
    print(f"\n⚙️  Loading config from {args.config}...")
    cfg = OmegaConf.load(args.config)

    # Extract curriculum config if enabled
    curriculum_config = None
    if args.curriculum:
        if "curriculum_learning" in cfg:
            curriculum_config = OmegaConf.to_container(
                cfg.curriculum_learning,
                resolve=True
            )
            # Override total_epochs from args
            if curriculum_config and "schedule" in curriculum_config:
                curriculum_config["schedule"]["total_epochs"] = args.epochs
        else:
            # Use default curriculum config
            curriculum_config = {
                "enable": True,
                "schedule": {"total_epochs": args.epochs}
            }
            print("  ⚠ No curriculum_learning section in config, using defaults")

    # Load datasets
    train_dataset, val_dataset, test_dataset = load_datasets(
        args.data_dir,
        enable_curriculum=args.curriculum,
        curriculum_config=curriculum_config
    )
    
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
        # Update curriculum threshold if enabled
        if args.curriculum and hasattr(train_dataset, 'is_curriculum') and train_dataset.is_curriculum:
            new_threshold = train_dataset.examples.step_epoch()
            stats = train_dataset.examples.get_difficulty_stats()

            print(f"\n📚 Curriculum Update (Epoch {epoch}):")
            print(f"  Threshold: {stats['threshold']:.4f}")
            print(f"  Available puzzles: {stats['count']} / {len(train_dataset._original_examples)}")
            print(f"  Difficulty range: [{stats['min']:.3f}, {stats['max']:.3f}], mean={stats['mean']:.3f}")

            # Recreate data loader with new filtered dataset
            train_loader, _ = create_data_loaders(
                train_dataset,
                val_dataset,
                batch_size=args.batch_size,
                max_length=cfg.arch.seq_len
            )

        train_loss = train_epoch(model, train_loader, optimizer, device, epoch)
        val_loss = validate(model, val_loader, device)

        # Log to W&B
        if args.use_wandb:
            log_dict = {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss
            }

            # Add curriculum stats if enabled
            if args.curriculum and hasattr(train_dataset, 'is_curriculum') and train_dataset.is_curriculum:
                curriculum_stats = train_dataset.examples.get_difficulty_stats()
                log_dict.update({
                    "curriculum/threshold": curriculum_stats["threshold"],
                    "curriculum/available_count": curriculum_stats["count"],
                    "curriculum/difficulty_mean": curriculum_stats["mean"],
                    "curriculum/difficulty_std": curriculum_stats["std"]
                })

            wandb.log(log_dict)

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

