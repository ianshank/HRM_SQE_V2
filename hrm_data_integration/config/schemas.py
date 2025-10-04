"""
Configuration Schemas for HRM Data Integration
==============================================

Defines Pydantic models for all configuration parameters.
No hard-coded values - all configuration is externalized.

Author: Data Integration Team
Date: October 2025
"""

from pydantic import BaseModel, Field, field_validator
from typing import Dict, List, Optional, Any, Union
from pathlib import Path
from enum import Enum


class DataFormat(str, Enum):
    """Supported data formats"""
    JSONL = "jsonl"
    JSON = "json"
    CSV = "csv"
    PARQUET = "parquet"


class HRMTaskType(str, Enum):
    """HRM task types"""
    REASONING = "reasoning"
    CODE_GENERATION = "code_generation"
    QUESTION_ANSWERING = "qa"
    CLASSIFICATION = "classification"
    SEQUENCE_LABELING = "sequence_labeling"


class ProcessingMode(str, Enum):
    """Data processing modes"""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    DISTRIBUTED = "distributed"


class DataSourceConfig(BaseModel):
    """Configuration for a single data source"""
    
    name: str = Field(description="Unique identifier for this data source")
    path: Path = Field(description="Path to data files")
    format: DataFormat = Field(default=DataFormat.JSONL)
    pattern: str = Field(default="*.jsonl", description="File pattern to match")
    encoding: str = Field(default="utf-8")
    
    # Field mapping
    prompt_field: str = Field(default="prompt", description="Field containing prompts")
    completion_field: str = Field(default="completion", description="Field containing completions")
    metadata_fields: List[str] = Field(default_factory=list)
    
    # Sampling configuration
    sample_size: Optional[int] = Field(default=None, description="Number of samples to use")
    sample_random: bool = Field(default=False)
    random_seed: Optional[int] = Field(default=None)
    
    @field_validator('path')
    @classmethod
    def validate_path(cls, v):
        """Validate that path exists"""
        if not Path(v).exists():
            raise ValueError(f"Path does not exist: {v}")
        return v


class TransformationConfig(BaseModel):
    """Configuration for data transformations"""
    
    # Text processing
    max_length: Optional[int] = Field(default=None, description="Maximum sequence length")
    truncate_strategy: str = Field(default="end", description="How to truncate: start, end, middle")
    normalize_whitespace: bool = Field(default=True)
    remove_special_chars: bool = Field(default=False)
    
    # Tokenization
    tokenizer_type: str = Field(default="character", description="character, word, or custom")
    vocab_size: int = Field(default=65536, gt=0)
    special_tokens: Dict[str, int] = Field(default_factory=dict)
    
    # HRM-specific
    sequence_length: int = Field(default=512, gt=0)
    add_puzzle_embedding: bool = Field(default=True)
    puzzle_id_strategy: str = Field(default="hash", description="hash, index, or custom")


class ValidationConfig(BaseModel):
    """Configuration for data validation"""
    
    # Quality checks
    min_length: int = Field(default=1, ge=0)
    max_length: int = Field(default=100000, gt=0)
    check_duplicates: bool = Field(default=True)
    check_completeness: bool = Field(default=True)
    
    # Filtering
    filter_empty: bool = Field(default=True)
    filter_malformed: bool = Field(default=True)
    filter_duplicates: bool = Field(default=True)
    
    # Thresholds
    quality_threshold: float = Field(default=0.8, ge=0.0, le=1.0)
    completeness_threshold: float = Field(default=0.9, ge=0.0, le=1.0)


class PipelineConfig(BaseModel):
    """Configuration for the entire data pipeline"""
    
    # Pipeline metadata
    pipeline_name: str = Field(description="Name of this pipeline configuration")
    version: str = Field(default="1.0.0")
    description: Optional[str] = None
    
    # Data sources
    sources: List[DataSourceConfig] = Field(description="List of data sources")
    
    # Processing configuration
    processing_mode: ProcessingMode = Field(default=ProcessingMode.SEQUENTIAL)
    batch_size: int = Field(default=100, gt=0)
    num_workers: int = Field(default=1, ge=1)
    
    # Transformations
    transformation: TransformationConfig = Field(default_factory=TransformationConfig)
    
    # Validation
    validation: ValidationConfig = Field(default_factory=ValidationConfig)
    
    # Output configuration
    output_dir: Path = Field(description="Directory for output files")
    output_format: DataFormat = Field(default=DataFormat.JSONL)
    split_ratios: Dict[str, float] = Field(
        default_factory=lambda: {"train": 0.8, "val": 0.1, "test": 0.1}
    )
    
    # HRM-specific
    hrm_task_type: HRMTaskType = Field(default=HRMTaskType.REASONING)
    hrm_config_path: Optional[Path] = None
    
    # Logging and monitoring
    log_level: str = Field(default="INFO")
    log_file: Optional[Path] = None
    enable_metrics: bool = Field(default=True)
    metrics_output: Optional[Path] = None
    
    # Error handling
    error_handling: str = Field(default="continue", description="continue, stop, or skip")
    max_errors: int = Field(default=100, ge=0)
    
    @field_validator('split_ratios')
    @classmethod
    def validate_splits(cls, v):
        """Validate split ratios sum to 1.0"""
        total = sum(v.values())
        if not (0.99 <= total <= 1.01):  # Allow for floating point errors
            raise ValueError(f"Split ratios must sum to 1.0, got {total}")
        return v
    
    @field_validator('output_dir')
    @classmethod
    def create_output_dir(cls, v):
        """Create output directory if it doesn't exist"""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v


class ExperimentConfig(BaseModel):
    """Configuration for experimental runs"""
    
    experiment_name: str = Field(description="Unique experiment identifier")
    experiment_dir: Path = Field(description="Directory for experiment outputs")
    
    # Multiple pipeline configurations for A/B testing
    pipelines: List[PipelineConfig] = Field(description="List of pipeline configurations to run")
    
    # Comparison metrics
    comparison_metrics: List[str] = Field(
        default_factory=lambda: ["quality_score", "processing_time", "data_coverage"]
    )
    
    # Resource limits
    max_memory_gb: float = Field(default=8.0, gt=0)
    max_time_hours: float = Field(default=24.0, gt=0)
    
    @field_validator('experiment_dir')
    @classmethod
    def create_experiment_dir(cls, v):
        """Create experiment directory if it doesn't exist"""
        Path(v).mkdir(parents=True, exist_ok=True)
        return v

