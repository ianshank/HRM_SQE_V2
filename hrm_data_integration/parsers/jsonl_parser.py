"""
JSONL Data Parser
=================

Parses JSONL files with various schemas into standardized format.
Supports multiple field name variations and schema types.

Author: Data Integration Team
Date: October 2025
"""

import json
from typing import Dict, List, Any, Optional, Iterator
from pathlib import Path
from datetime import datetime
import hashlib

from ..core.interfaces import IDataParser, IDataSource, DataRecord, ProcessingResult
from ..config.schemas import DataSourceConfig


class JSONLDataSource(IDataSource[str]):
    """Data source for JSONL files"""
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.files = list(Path(config.path).glob(config.pattern))
        
        if not self.files:
            raise ValueError(f"No files found matching pattern: {config.pattern} in {config.path}")
    
    def read(self) -> Iterator[str]:
        """Read lines from all JSONL files"""
        for file_path in self.files:
            with open(file_path, 'r', encoding=self.config.encoding) as f:
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if line:
                        yield line
    
    def count(self) -> int:
        """Count total records across all files"""
        total = 0
        for file_path in self.files:
            with open(file_path, 'r', encoding=self.config.encoding) as f:
                total += sum(1 for line in f if line.strip())
        return total
    
    def validate_source(self) -> bool:
        """Validate that all files are readable JSONL"""
        try:
            for file_path in self.files:
                with open(file_path, 'r', encoding=self.config.encoding) as f:
                    # Try to parse first line
                    first_line = f.readline().strip()
                    if first_line:
                        json.loads(first_line)
            return True
        except Exception:
            return False


class JSONLParser(IDataParser[str, DataRecord]):
    """Parser for JSONL data with flexible schema support"""
    
    # Common field name variations
    PROMPT_FIELDS = ['prompt', 'input', 'question', 'instruction', 'text']
    COMPLETION_FIELDS = ['completion', 'output', 'answer', 'response', 'target']
    
    def __init__(self, config: DataSourceConfig):
        self.config = config
        self.source_name = config.name
        
        # Build field mappings
        self.prompt_field = config.prompt_field
        self.completion_field = config.completion_field
        self.metadata_fields = config.metadata_fields
        
        # Statistics
        self.parsed_count = 0
        self.error_count = 0
    
    def parse(self, data: str) -> ProcessingResult[DataRecord]:
        """Parse single JSONL line into DataRecord"""
        start_time = datetime.now()
        
        try:
            # Parse JSON
            obj = json.loads(data)
            
            # Extract prompt
            prompt = self._extract_field(obj, self.prompt_field, self.PROMPT_FIELDS)
            if prompt is None:
                raise ValueError(f"Could not find prompt field in: {list(obj.keys())}")
            
            # Extract completion
            completion = self._extract_field(obj, self.completion_field, self.COMPLETION_FIELDS)
            if completion is None:
                raise ValueError(f"Could not find completion field in: {list(obj.keys())}")
            
            # Extract metadata
            metadata = self._extract_metadata(obj)
            
            # Generate unique ID
            record_id = self._generate_id(prompt, completion)
            
            # Create record
            record = DataRecord(
                id=record_id,
                prompt=str(prompt),
                completion=str(completion),
                metadata=metadata,
                source=self.source_name,
                timestamp=datetime.now()
            )
            
            self.parsed_count += 1
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                data=record,
                error=None,
                metadata={"parsing_time": processing_time},
                processing_time=processing_time
            )
            
        except Exception as e:
            self.error_count += 1
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=False,
                data=None,
                error=str(e),
                metadata={"raw_data": data[:200]},  # First 200 chars for debugging
                processing_time=processing_time
            )
    
    def can_parse(self, data: str) -> bool:
        """Check if data can be parsed"""
        try:
            obj = json.loads(data)
            
            # Check if we can find required fields
            has_prompt = self._extract_field(obj, self.prompt_field, self.PROMPT_FIELDS) is not None
            has_completion = self._extract_field(obj, self.completion_field, self.COMPLETION_FIELDS) is not None
            
            return has_prompt and has_completion
        except:
            return False
    
    def _extract_field(self, obj: Dict, primary_field: str, alternatives: List[str]) -> Optional[Any]:
        """Extract field with fallback to alternatives"""
        # Try primary field
        if primary_field in obj:
            return obj[primary_field]
        
        # Try alternatives
        for field_name in alternatives:
            if field_name in obj:
                return obj[field_name]
        
        # Try case-insensitive match
        obj_lower = {k.lower(): v for k, v in obj.items()}
        for field_name in [primary_field] + alternatives:
            if field_name.lower() in obj_lower:
                return obj_lower[field_name.lower()]
        
        return None
    
    def _extract_metadata(self, obj: Dict) -> Dict[str, Any]:
        """Extract metadata fields"""
        metadata = {}
        
        # Extract specified metadata fields
        for field_name in self.metadata_fields:
            if field_name in obj:
                metadata[field_name] = obj[field_name]
        
        # Extract common metadata
        common_fields = ['agent_type', 'model', 'domain', 'complexity', 'generated_at']
        for field in common_fields:
            if field in obj and field not in metadata:
                metadata[field] = obj[field]
        
        return metadata
    
    def _generate_id(self, prompt: str, completion: str) -> str:
        """Generate deterministic ID from content"""
        content = f"{prompt}|{completion}"
        return hashlib.md5(content.encode()).hexdigest()
    
    def get_statistics(self) -> Dict[str, int]:
        """Get parser statistics"""
        return {
            "parsed_count": self.parsed_count,
            "error_count": self.error_count,
            "success_rate": self.parsed_count / (self.parsed_count + self.error_count) 
                if (self.parsed_count + self.error_count) > 0 else 0.0
        }


class BatchJSONLParser:
    """Batch parser for efficient processing of large JSONL files"""
    
    def __init__(self, parser: JSONLParser, batch_size: int = 1000):
        self.parser = parser
        self.batch_size = batch_size
    
    def parse_batch(self, lines: List[str]) -> List[ProcessingResult[DataRecord]]:
        """Parse a batch of lines"""
        return [self.parser.parse(line) for line in lines]
    
    def parse_file(self, file_path: Path) -> Iterator[ProcessingResult[DataRecord]]:
        """Parse entire file in batches"""
        batch = []
        
        with open(file_path, 'r') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                
                batch.append(line)
                
                if len(batch) >= self.batch_size:
                    yield from self.parse_batch(batch)
                    batch = []
            
            # Process remaining batch
            if batch:
                yield from self.parse_batch(batch)

