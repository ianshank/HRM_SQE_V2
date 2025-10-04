"""Core interfaces and data structures"""

from .interfaces import (
    DataRecord,
    ProcessingResult,
    IDataSource,
    IDataParser,
    IDataTransformer,
    IDataValidator,
    IDataFilter,
    IDataWriter,
    ITokenizer,
    IPipelineStage,
    IPipeline,
    IConfigLoader,
    IMetricsCollector,
    IStateManager,
    IProgressTracker
)

__all__ = [
    "DataRecord",
    "ProcessingResult",
    "IDataSource",
    "IDataParser",
    "IDataTransformer",
    "IDataValidator",
    "IDataFilter",
    "IDataWriter",
    "ITokenizer",
    "IPipelineStage",
    "IPipeline",
    "IConfigLoader",
    "IMetricsCollector",
    "IStateManager",
    "IProgressTracker",
]

