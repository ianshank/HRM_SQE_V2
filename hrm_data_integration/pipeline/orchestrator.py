"""
Pipeline Orchestrator
=====================

Orchestrates the entire data processing pipeline.
Coordinates parsing, transformation, validation, and output.

Author: Data Integration Team
Date: October 2025
"""

from typing import List, Dict, Any, Optional
from pathlib import Path
from datetime import datetime
import logging
from tqdm import tqdm

from ..core.interfaces import (
    IPipeline, IPipelineStage, ProcessingResult, DataRecord
)
from ..config.schemas import PipelineConfig
from ..models.hrm_data import HRMExample, HRMDataset
from ..parsers.jsonl_parser import JSONLDataSource, JSONLParser
from ..transformers.hrm_transformer import HRMDataTransformer, CharacterTokenizer
from ..validators.data_validators import (
    CompositeValidator, LengthValidator, CompletenessValidator,
    QualityValidator, DuplicateFilter, EmptyFilter
)


class ParsingStage(IPipelineStage[PipelineConfig, List[DataRecord]]):
    """Stage for parsing raw data"""
    
    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.stage_results: Dict[str, Any] = {}
    
    def execute(self, input_data: PipelineConfig) -> ProcessingResult[List[DataRecord]]:
        """Execute parsing stage"""
        start_time = datetime.now()
        self.logger.info(f"Starting parsing stage with {len(self.config.sources)} sources")
        
        all_records = []
        source_stats = {}
        
        for source_config in self.config.sources:
            self.logger.info(f"Processing source: {source_config.name}")
            
            try:
                # Create data source and parser
                data_source = JSONLDataSource(source_config)
                parser = JSONLParser(source_config)
                
                # Validate source
                if not data_source.validate_source():
                    self.logger.warning(f"Source validation failed: {source_config.name}")
                    continue
                
                # Parse data
                total_lines = data_source.count()
                self.logger.info(f"Found {total_lines} records in {source_config.name}")
                
                parsed_records = []
                error_count = 0
                
                for line in tqdm(data_source.read(), total=total_lines, desc=f"Parsing {source_config.name}"):
                    result = parser.parse(line)
                    
                    if result.success and result.data:
                        parsed_records.append(result.data)
                    else:
                        error_count += 1
                        if error_count <= 5:  # Log first few errors
                            self.logger.error(f"Parse error: {result.error}")
                
                all_records.extend(parsed_records)
                
                # Store statistics
                source_stats[source_config.name] = {
                    "total_lines": total_lines,
                    "parsed": len(parsed_records),
                    "errors": error_count,
                    "success_rate": len(parsed_records) / total_lines if total_lines > 0 else 0
                }
                
                self.logger.info(f"Parsed {len(parsed_records)} records from {source_config.name}")
                
            except Exception as e:
                self.logger.error(f"Error processing source {source_config.name}: {e}")
                source_stats[source_config.name] = {"error": str(e)}
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        self.stage_results = {
            "total_records": len(all_records),
            "source_stats": source_stats,
            "processing_time": processing_time
        }
        
        self.logger.info(f"Parsing stage complete: {len(all_records)} total records")
        
        return ProcessingResult(
            success=True,
            data=all_records,
            error=None,
            metadata=self.stage_results,
            processing_time=processing_time
        )
    
    def get_stage_name(self) -> str:
        return "parsing"
    
    def validate_stage(self) -> bool:
        return len(self.config.sources) > 0


class TransformationStage(IPipelineStage[List[DataRecord], List[HRMExample]]):
    """Stage for transforming data to HRM format"""
    
    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.stage_results: Dict[str, Any] = {}
        
        # Create tokenizer and transformer
        self.tokenizer = CharacterTokenizer(
            vocab_size=config.transformation.vocab_size,
            special_tokens=config.transformation.special_tokens
        )
        self.transformer = HRMDataTransformer(config.transformation, self.tokenizer)
    
    def execute(self, input_data: List[DataRecord]) -> ProcessingResult[List[HRMExample]]:
        """Execute transformation stage"""
        start_time = datetime.now()
        self.logger.info(f"Starting transformation stage with {len(input_data)} records")
        
        examples = []
        error_count = 0
        
        for record in tqdm(input_data, desc="Transforming"):
            result = self.transformer.transform(record)
            
            if result.success and result.data:
                examples.append(result.data)
            else:
                error_count += 1
                if error_count <= 5:
                    self.logger.error(f"Transform error: {result.error}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        self.stage_results = {
            "input_records": len(input_data),
            "output_examples": len(examples),
            "errors": error_count,
            "success_rate": len(examples) / len(input_data) if input_data else 0,
            "transformer_stats": self.transformer.get_statistics(),
            "processing_time": processing_time
        }
        
        self.logger.info(f"Transformation stage complete: {len(examples)} examples created")
        
        return ProcessingResult(
            success=True,
            data=examples,
            error=None,
            metadata=self.stage_results,
            processing_time=processing_time
        )
    
    def get_stage_name(self) -> str:
        return "transformation"
    
    def validate_stage(self) -> bool:
        return self.tokenizer is not None and self.transformer is not None


class ValidationStage(IPipelineStage[List[HRMExample], List[HRMExample]]):
    """Stage for validating and filtering data"""
    
    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.stage_results: Dict[str, Any] = {}
        
        # Create validators
        self.validators = [
            LengthValidator(config.validation),
            CompletenessValidator(config.validation),
            QualityValidator(config.validation)
        ]
        self.composite_validator = CompositeValidator(self.validators)
        
        # Create filters
        self.filters = [
            EmptyFilter(config.validation),
            DuplicateFilter(config.validation)
        ]
    
    def execute(self, input_data: List[HRMExample]) -> ProcessingResult[List[HRMExample]]:
        """Execute validation stage"""
        start_time = datetime.now()
        self.logger.info(f"Starting validation stage with {len(input_data)} examples")
        
        # Apply filters first
        filtered_data = input_data
        filter_stats = {}
        
        for filter_obj in self.filters:
            before_count = len(filtered_data)
            filtered_data = filter_obj.filter(filtered_data)
            after_count = len(filtered_data)
            
            filter_name = filter_obj.__class__.__name__
            filter_stats[filter_name] = {
                "before": before_count,
                "after": after_count,
                "removed": before_count - after_count
            }
            
            self.logger.info(f"{filter_name}: {before_count} -> {after_count}")
        
        # Validate remaining examples
        validated_data = []
        validation_failures = 0
        
        for example in tqdm(filtered_data, desc="Validating"):
            result = self.composite_validator.validate(example)
            
            if result.success:
                validated_data.append(example)
            else:
                validation_failures += 1
                if validation_failures <= 5:
                    self.logger.warning(f"Validation failed: {result.error}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Collect statistics
        validator_stats = {}
        for validator in self.validators:
            if hasattr(validator, 'get_statistics'):
                validator_stats[validator.__class__.__name__] = validator.get_statistics()
        
        self.stage_results = {
            "input_examples": len(input_data),
            "after_filtering": len(filtered_data),
            "after_validation": len(validated_data),
            "validation_failures": validation_failures,
            "filter_stats": filter_stats,
            "validator_stats": validator_stats,
            "processing_time": processing_time
        }
        
        self.logger.info(f"Validation stage complete: {len(validated_data)} valid examples")
        
        return ProcessingResult(
            success=True,
            data=validated_data,
            error=None,
            metadata=self.stage_results,
            processing_time=processing_time
        )
    
    def get_stage_name(self) -> str:
        return "validation"
    
    def validate_stage(self) -> bool:
        return len(self.validators) > 0


class OutputStage(IPipelineStage[List[HRMExample], HRMDataset]):
    """Stage for writing output"""
    
    def __init__(self, config: PipelineConfig, logger: Optional[logging.Logger] = None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        self.stage_results: Dict[str, Any] = {}
    
    def execute(self, input_data: List[HRMExample]) -> ProcessingResult[HRMDataset]:
        """Execute output stage"""
        start_time = datetime.now()
        self.logger.info(f"Starting output stage with {len(input_data)} examples")
        
        # Create dataset
        dataset = HRMDataset(
            dataset_id=self.config.pipeline_name,
            examples=input_data,
            name=self.config.pipeline_name,
            version=self.config.version,
            description=self.config.description,
            vocab_size=self.config.transformation.vocab_size
        )
        
        # Compute statistics
        stats = dataset.compute_statistics()
        self.logger.info(f"Dataset statistics: {stats}")
        
        # Split dataset
        splits = dataset.split(self.config.split_ratios)
        
        # Save splits
        output_files = {}
        for split_name, split_dataset in splits.items():
            output_file = self.config.output_dir / f"{split_name}.jsonl"
            split_dataset.save(str(output_file))
            output_files[split_name] = output_file
            self.logger.info(f"Saved {split_name} split: {len(split_dataset.examples)} examples -> {output_file}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        self.stage_results = {
            "total_examples": len(input_data),
            "splits": {name: len(ds.examples) for name, ds in splits.items()},
            "output_files": {name: str(path) for name, path in output_files.items()},
            "statistics": stats,
            "processing_time": processing_time
        }
        
        self.logger.info("Output stage complete")
        
        return ProcessingResult(
            success=True,
            data=dataset,
            error=None,
            metadata=self.stage_results,
            processing_time=processing_time
        )
    
    def get_stage_name(self) -> str:
        return "output"
    
    def validate_stage(self) -> bool:
        return self.config.output_dir.exists()


class HRMDataPipeline:
    """Complete HRM data processing pipeline"""
    
    def __init__(self, config: PipelineConfig):
        self.config = config
        self.logger = self._setup_logging()
        
        # Create stages
        self.stages = [
            ParsingStage(config, self.logger),
            TransformationStage(config, self.logger),
            ValidationStage(config, self.logger),
            OutputStage(config, self.logger)
        ]
        
        self.stage_results: Dict[str, ProcessingResult] = {}
    
    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration"""
        logger = logging.getLogger(self.config.pipeline_name)
        logger.setLevel(getattr(logging, self.config.log_level))
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        # File handler if specified
        if self.config.log_file:
            file_handler = logging.FileHandler(self.config.log_file)
            file_handler.setLevel(logging.DEBUG)
            file_handler.setFormatter(formatter)
            logger.addHandler(file_handler)
        
        return logger
    
    def execute(self) -> ProcessingResult[HRMDataset]:
        """Execute the complete pipeline"""
        start_time = datetime.now()
        self.logger.info(f"Starting pipeline: {self.config.pipeline_name}")
        
        try:
            # Validate pipeline
            if not self.validate_pipeline():
                raise ValueError("Pipeline validation failed")
            
            # Execute stages sequentially
            current_data: Any = self.config
            
            for stage in self.stages:
                self.logger.info(f"Executing stage: {stage.get_stage_name()}")
                
                result = stage.execute(current_data)
                self.stage_results[stage.get_stage_name()] = result
                
                if not result.success:
                    error_msg = f"Stage {stage.get_stage_name()} failed: {result.error}"
                    self.logger.error(error_msg)
                    raise RuntimeError(error_msg)
                
                current_data = result.data
            
            total_time = (datetime.now() - start_time).total_seconds()
            
            self.logger.info(f"Pipeline complete in {total_time:.2f}s")
            
            # Write metrics if enabled
            if self.config.enable_metrics:
                self._write_metrics()
            
            return ProcessingResult(
                success=True,
                data=current_data,
                error=None,
                metadata={"total_time": total_time, "stages": self.stage_results},
                processing_time=total_time
            )
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}", exc_info=True)
            total_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=False,
                data=None,
                error=str(e),
                metadata={"total_time": total_time, "stages": self.stage_results},
                processing_time=total_time
            )
    
    def validate_pipeline(self) -> bool:
        """Validate all stages"""
        for stage in self.stages:
            if not stage.validate_stage():
                self.logger.error(f"Stage validation failed: {stage.get_stage_name()}")
                return False
        return True
    
    def get_stage_results(self) -> Dict[str, ProcessingResult]:
        """Get results from all stages"""
        return self.stage_results.copy()
    
    def _write_metrics(self) -> None:
        """Write pipeline metrics to file"""
        if not self.config.metrics_output:
            return
        
        import json
        
        metrics = {
            "pipeline_name": self.config.pipeline_name,
            "timestamp": datetime.now().isoformat(),
            "stages": {
                name: {
                    "success": result.success,
                    "processing_time": result.processing_time,
                    "metadata": result.metadata
                }
                for name, result in self.stage_results.items()
            }
        }
        
        with open(self.config.metrics_output, 'w') as f:
            json.dump(metrics, f, indent=2)
        
        self.logger.info(f"Metrics written to {self.config.metrics_output}")

