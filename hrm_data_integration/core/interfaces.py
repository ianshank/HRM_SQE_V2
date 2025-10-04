"""
Core Interfaces for HRM Data Integration
========================================

Defines abstract base classes and protocols for all components.
Ensures modularity, testability, and extensibility.

Author: Data Integration Team
Date: October 2025
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Iterator, Optional, Protocol, TypeVar, Generic
from pathlib import Path
from dataclasses import dataclass
from datetime import datetime


# Type variables for generics
T = TypeVar('T')
InputType = TypeVar('InputType')
OutputType = TypeVar('OutputType')


@dataclass
class DataRecord:
    """Immutable data record"""
    id: str
    prompt: str
    completion: str
    metadata: Dict[str, Any]
    source: str
    timestamp: datetime
    
    def __hash__(self):
        return hash(self.id)


@dataclass
class ProcessingResult(Generic[T]):
    """Result of a processing operation"""
    success: bool
    data: Optional[T]
    error: Optional[str]
    metadata: Dict[str, Any]
    processing_time: float


class IDataSource(ABC, Generic[T]):
    """Interface for data sources"""
    
    @abstractmethod
    def read(self) -> Iterator[T]:
        """Read data from source"""
        pass
    
    @abstractmethod
    def count(self) -> int:
        """Count total records"""
        pass
    
    @abstractmethod
    def validate_source(self) -> bool:
        """Validate that source is accessible and well-formed"""
        pass


class IDataParser(ABC, Generic[InputType, OutputType]):
    """Interface for data parsers"""
    
    @abstractmethod
    def parse(self, data: InputType) -> ProcessingResult[OutputType]:
        """Parse raw data into structured format"""
        pass
    
    @abstractmethod
    def can_parse(self, data: InputType) -> bool:
        """Check if this parser can handle the data"""
        pass


class IDataTransformer(ABC, Generic[InputType, OutputType]):
    """Interface for data transformers"""
    
    @abstractmethod
    def transform(self, data: InputType) -> ProcessingResult[OutputType]:
        """Transform data from one format to another"""
        pass
    
    @abstractmethod
    def validate_input(self, data: InputType) -> bool:
        """Validate input data before transformation"""
        pass
    
    @abstractmethod
    def validate_output(self, data: OutputType) -> bool:
        """Validate output data after transformation"""
        pass


class IDataValidator(ABC, Generic[T]):
    """Interface for data validators"""
    
    @abstractmethod
    def validate(self, data: T) -> ProcessingResult[T]:
        """Validate data and return result"""
        pass
    
    @abstractmethod
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get validation rules configuration"""
        pass


class IDataFilter(ABC, Generic[T]):
    """Interface for data filters"""
    
    @abstractmethod
    def filter(self, data: List[T]) -> List[T]:
        """Filter data based on criteria"""
        pass
    
    @abstractmethod
    def should_keep(self, item: T) -> bool:
        """Determine if single item should be kept"""
        pass


class IDataWriter(ABC, Generic[T]):
    """Interface for data writers"""
    
    @abstractmethod
    def write(self, data: List[T], output_path: Path) -> ProcessingResult[Path]:
        """Write data to destination"""
        pass
    
    @abstractmethod
    def append(self, data: T, output_path: Path) -> ProcessingResult[None]:
        """Append single record to destination"""
        pass
    
    @abstractmethod
    def validate_destination(self, output_path: Path) -> bool:
        """Validate destination is writable"""
        pass


class ITokenizer(ABC):
    """Interface for tokenizers"""
    
    @abstractmethod
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        pass
    
    @abstractmethod
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        pass
    
    @abstractmethod
    def get_vocab_size(self) -> int:
        """Get vocabulary size"""
        pass
    
    @abstractmethod
    def get_special_tokens(self) -> Dict[str, int]:
        """Get special tokens mapping"""
        pass


class IMetricsCollector(ABC):
    """Interface for metrics collection"""
    
    @abstractmethod
    def record_metric(self, name: str, value: float, tags: Optional[Dict[str, str]] = None) -> None:
        """Record a metric value"""
        pass
    
    @abstractmethod
    def get_metrics(self, name: Optional[str] = None) -> Dict[str, Any]:
        """Get collected metrics"""
        pass
    
    @abstractmethod
    def export_metrics(self, output_path: Path) -> None:
        """Export metrics to file"""
        pass


class IPipelineStage(ABC, Generic[InputType, OutputType]):
    """Interface for pipeline stages"""
    
    @abstractmethod
    def execute(self, input_data: InputType) -> ProcessingResult[OutputType]:
        """Execute this stage of the pipeline"""
        pass
    
    @abstractmethod
    def get_stage_name(self) -> str:
        """Get unique stage identifier"""
        pass
    
    @abstractmethod
    def validate_stage(self) -> bool:
        """Validate stage configuration"""
        pass


class IPipeline(ABC, Generic[InputType, OutputType]):
    """Interface for complete pipeline"""
    
    @abstractmethod
    def add_stage(self, stage: IPipelineStage) -> 'IPipeline':
        """Add a stage to the pipeline"""
        pass
    
    @abstractmethod
    def execute(self, input_data: InputType) -> ProcessingResult[OutputType]:
        """Execute the entire pipeline"""
        pass
    
    @abstractmethod
    def validate_pipeline(self) -> bool:
        """Validate entire pipeline configuration"""
        pass
    
    @abstractmethod
    def get_stage_results(self) -> Dict[str, ProcessingResult]:
        """Get results from each stage"""
        pass


class IConfigLoader(ABC, Generic[T]):
    """Interface for configuration loaders"""
    
    @abstractmethod
    def load(self, config_path: Path) -> T:
        """Load configuration from file"""
        pass
    
    @abstractmethod
    def validate(self, config: T) -> bool:
        """Validate configuration"""
        pass
    
    @abstractmethod
    def get_defaults(self) -> T:
        """Get default configuration"""
        pass


class IStateManager(ABC):
    """Interface for state management"""
    
    @abstractmethod
    def save_state(self, state: Dict[str, Any], checkpoint_id: str) -> None:
        """Save pipeline state"""
        pass
    
    @abstractmethod
    def load_state(self, checkpoint_id: str) -> Dict[str, Any]:
        """Load pipeline state"""
        pass
    
    @abstractmethod
    def list_checkpoints(self) -> List[str]:
        """List available checkpoints"""
        pass


class ILogger(Protocol):
    """Protocol for logging"""
    
    def info(self, message: str, **kwargs) -> None: ...
    def warning(self, message: str, **kwargs) -> None: ...
    def error(self, message: str, **kwargs) -> None: ...
    def debug(self, message: str, **kwargs) -> None: ...


class IProgressTracker(ABC):
    """Interface for progress tracking"""
    
    @abstractmethod
    def start(self, total: int, description: str) -> None:
        """Start tracking progress"""
        pass
    
    @abstractmethod
    def update(self, increment: int = 1) -> None:
        """Update progress"""
        pass
    
    @abstractmethod
    def finish(self) -> None:
        """Finish tracking"""
        pass

