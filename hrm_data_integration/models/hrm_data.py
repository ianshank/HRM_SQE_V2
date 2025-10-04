"""
HRM Data Models
===============

Defines data structures for HRM training format.
Matches the expected input format for HRM model training.

Author: Data Integration Team
Date: October 2025
"""

from pydantic import BaseModel, Field, field_validator
from typing import List, Dict, Optional, Any
from enum import Enum
from pathlib import Path
import hashlib


class HRMDataType(str, Enum):
    """Types of HRM training data"""
    PUZZLE = "puzzle"
    REASONING = "reasoning"
    CODE = "code"
    QA = "qa"


class HRMExample(BaseModel):
    """Single HRM training example"""
    
    # Core fields
    inputs: List[int] = Field(description="Input token IDs")
    targets: List[int] = Field(description="Target token IDs")
    puzzle_identifiers: List[int] = Field(description="Puzzle/task identifier")
    
    # Metadata
    example_id: str = Field(description="Unique example identifier")
    source: str = Field(description="Source dataset/file")
    data_type: HRMDataType = Field(default=HRMDataType.REASONING)
    
    # Quality indicators
    length: int = Field(ge=0, description="Sequence length")
    difficulty: Optional[str] = None
    
    # Original data (for debugging/analysis)
    original_prompt: Optional[str] = None
    original_completion: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    @field_validator('inputs', 'targets')
    @classmethod
    def validate_non_empty(cls, v):
        """Ensure sequences are not empty"""
        if not v:
            raise ValueError("Sequence cannot be empty")
        return v
    
    @field_validator('puzzle_identifiers')
    @classmethod
    def validate_puzzle_ids(cls, v):
        """Ensure puzzle IDs are valid"""
        if not v or any(pid < 0 for pid in v):
            raise ValueError("Invalid puzzle identifiers")
        return v
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            "inputs": self.inputs,
            "targets": self.targets,
            "puzzle_identifiers": self.puzzle_identifiers,
            "example_id": self.example_id,
            "source": self.source,
            "data_type": self.data_type.value,
            "length": self.length,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'HRMExample':
        """Create from dictionary"""
        # Ensure length is computed if not present
        if "length" not in data and "inputs" in data:
            data["length"] = len(data["inputs"])
        return cls(**data)


class HRMDataset(BaseModel):
    """Collection of HRM examples with metadata"""
    
    dataset_id: str = Field(description="Unique dataset identifier")
    examples: List[HRMExample] = Field(default_factory=list)
    
    # Dataset metadata
    name: str = Field(description="Dataset name")
    version: str = Field(default="1.0.0")
    description: Optional[str] = None
    
    # Statistics
    total_examples: int = Field(default=0, ge=0)
    total_tokens: int = Field(default=0, ge=0)
    avg_length: float = Field(default=0.0, ge=0.0)
    
    # Vocabulary info
    vocab_size: int = Field(gt=0)
    num_puzzle_types: int = Field(default=1, ge=1)
    
    # Source tracking
    sources: List[str] = Field(default_factory=list)
    processing_history: List[str] = Field(default_factory=list)
    
    def model_post_init(self, __context) -> None:
        """Update statistics after initialization"""
        if self.examples and self.total_examples == 0:
            self.total_examples = len(self.examples)
            self.total_tokens = sum(len(ex.inputs) + len(ex.targets) for ex in self.examples)
            if self.examples:
                lengths = [ex.length for ex in self.examples]
                self.avg_length = sum(lengths) / len(lengths)
    
    def add_example(self, example: HRMExample) -> None:
        """Add an example to the dataset"""
        self.examples.append(example)
        self.total_examples += 1
        self.total_tokens += len(example.inputs) + len(example.targets)
        
        # Update sources
        if example.source not in self.sources:
            self.sources.append(example.source)
    
    def compute_statistics(self) -> Dict[str, Any]:
        """Compute dataset statistics"""
        if not self.examples:
            return {}
        
        lengths = [ex.length for ex in self.examples]
        self.avg_length = sum(lengths) / len(lengths)
        
        # Get unique puzzle types
        puzzle_types = set()
        for ex in self.examples:
            puzzle_types.update(ex.puzzle_identifiers)
        self.num_puzzle_types = len(puzzle_types)
        
        return {
            "total_examples": self.total_examples,
            "total_tokens": self.total_tokens,
            "avg_length": self.avg_length,
            "min_length": min(lengths),
            "max_length": max(lengths),
            "num_puzzle_types": self.num_puzzle_types,
            "vocab_size": self.vocab_size,
            "sources": self.sources
        }
    
    def split(self, ratios: Dict[str, float]) -> Dict[str, 'HRMDataset']:
        """Split dataset into train/val/test"""
        import random
        
        # Validate ratios
        total = sum(ratios.values())
        if not (0.99 <= total <= 1.01):
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        
        # Shuffle examples
        examples = self.examples.copy()
        random.shuffle(examples)
        
        # Calculate split indices
        total_examples = len(examples)
        splits = {}
        start_idx = 0
        
        for split_name, ratio in ratios.items():
            split_size = int(total_examples * ratio)
            end_idx = start_idx + split_size
            
            # Create split dataset
            split_dataset = HRMDataset(
                dataset_id=f"{self.dataset_id}_{split_name}",
                examples=examples[start_idx:end_idx],
                name=f"{self.name}_{split_name}",
                version=self.version,
                vocab_size=self.vocab_size,
                sources=self.sources.copy()
            )
            split_dataset.compute_statistics()
            splits[split_name] = split_dataset
            
            start_idx = end_idx
        
        return splits
    
    def to_jsonl(self) -> List[str]:
        """Convert to JSONL format"""
        import json
        return [json.dumps(ex.to_dict()) for ex in self.examples]
    
    def save(self, output_path: str | Path) -> None:
        """Save dataset to file"""
        import json
        from pathlib import Path
        
        output_path = Path(output_path)
        
        with open(output_path, 'w') as f:
            for example in self.examples:
                f.write(json.dumps(example.to_dict()) + '\n')
    
    @classmethod
    def load(cls, input_path: str | Path) -> 'HRMDataset':
        """Load dataset from file"""
        import json
        
        input_path = Path(input_path)
        examples = []
        
        with open(input_path, 'r') as f:
            for line in f:
                if line.strip():
                    data = json.loads(line)
                    examples.append(HRMExample.from_dict(data))
        
        # Create dataset
        dataset = cls(
            dataset_id=hashlib.md5(str(input_path).encode()).hexdigest()[:8],
            examples=examples,
            name=input_path.stem,
            vocab_size=max(
                max(ex.inputs + ex.targets) for ex in examples
            ) + 1 if examples else 65536
        )
        dataset.compute_statistics()
        
        return dataset


class HRMBatch(BaseModel):
    """Batch of HRM examples for training"""
    
    inputs: List[List[int]] = Field(description="Batch of input sequences")
    targets: List[List[int]] = Field(description="Batch of target sequences")
    puzzle_identifiers: List[List[int]] = Field(description="Batch of puzzle IDs")
    
    batch_size: int = Field(ge=1)
    max_length: int = Field(ge=1)
    
    def to_tensors(self):
        """Convert to PyTorch tensors (if torch is available)"""
        try:
            import torch
            return {
                "inputs": torch.tensor(self.inputs),
                "targets": torch.tensor(self.targets),
                "puzzle_identifiers": torch.tensor(self.puzzle_identifiers)
            }
        except ImportError:
            raise ImportError("PyTorch is required for tensor conversion")
    
    @classmethod
    def from_examples(cls, examples: List[HRMExample], max_length: int) -> 'HRMBatch':
        """Create batch from examples"""
        # Pad sequences to max_length
        inputs = []
        targets = []
        puzzle_ids = []
        
        for ex in examples:
            # Pad or truncate inputs
            padded_inputs = ex.inputs[:max_length]
            padded_inputs += [0] * (max_length - len(padded_inputs))
            inputs.append(padded_inputs)
            
            # Pad or truncate targets
            padded_targets = ex.targets[:max_length]
            padded_targets += [-100] * (max_length - len(padded_targets))
            targets.append(padded_targets)
            
            # Puzzle IDs
            puzzle_ids.append(ex.puzzle_identifiers)
        
        return cls(
            inputs=inputs,
            targets=targets,
            puzzle_identifiers=puzzle_ids,
            batch_size=len(examples),
            max_length=max_length
        )

