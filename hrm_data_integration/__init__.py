"""
HRM Data Integration Package
=============================

Modular, dynamic, and reusable data integration system for HRM training.
Supports multiple data sources, transformations, validations, and outputs.

Author: Data Integration Team
Date: October 2025
Version: 1.0.0
"""

__version__ = "1.0.0"
__author__ = "Data Integration Team"

from .config.schemas import PipelineConfig, DataSourceConfig, TransformationConfig, ValidationConfig
from .config.loader import YAMLConfigLoader, ConfigBuilder
from .pipeline.orchestrator import HRMDataPipeline
from .models.hrm_data import HRMExample, HRMDataset, HRMBatch

__all__ = [
    "PipelineConfig",
    "DataSourceConfig",
    "TransformationConfig",
    "ValidationConfig",
    "YAMLConfigLoader",
    "ConfigBuilder",
    "HRMDataPipeline",
    "HRMExample",
    "HRMDataset",
    "HRMBatch",
]

