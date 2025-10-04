"""
Data Validators
===============

Validation components for data quality assurance.
Supports multiple validation rules and quality metrics.

Author: Data Integration Team
Date: October 2025
"""

from typing import Dict, Any, List, Set
from datetime import datetime
from abc import ABC, abstractmethod

from ..core.interfaces import IDataValidator, IDataFilter, ProcessingResult
from ..models.hrm_data import HRMExample
from ..config.schemas import ValidationConfig


class BaseValidator(IDataValidator[HRMExample], ABC):
    """Base class for validators"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.validation_count = 0
        self.pass_count = 0
        self.fail_count = 0
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get validation statistics"""
        return {
            "validation_count": self.validation_count,
            "pass_count": self.pass_count,
            "fail_count": self.fail_count,
            "pass_rate": self.pass_count / self.validation_count if self.validation_count > 0 else 0.0
        }


class LengthValidator(BaseValidator):
    """Validates sequence lengths"""
    
    def validate(self, data: HRMExample) -> ProcessingResult[HRMExample]:
        """Validate sequence length constraints"""
        start_time = datetime.now()
        self.validation_count += 1
        
        errors = []
        
        # Check minimum length
        if len(data.inputs) < self.config.min_length:
            errors.append(f"Input length {len(data.inputs)} < minimum {self.config.min_length}")
        
        if len(data.targets) < self.config.min_length:
            errors.append(f"Target length {len(data.targets)} < minimum {self.config.min_length}")
        
        # Check maximum length
        if len(data.inputs) > self.config.max_length:
            errors.append(f"Input length {len(data.inputs)} > maximum {self.config.max_length}")
        
        if len(data.targets) > self.config.max_length:
            errors.append(f"Target length {len(data.targets)} > maximum {self.config.max_length}")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if errors:
            self.fail_count += 1
            return ProcessingResult(
                success=False,
                data=None,
                error="; ".join(errors),
                metadata={"validator": "length"},
                processing_time=processing_time
            )
        
        self.pass_count += 1
        return ProcessingResult(
            success=True,
            data=data,
            error=None,
            metadata={"validator": "length"},
            processing_time=processing_time
        )
    
    def get_validation_rules(self) -> Dict[str, Any]:
        return {
            "min_length": self.config.min_length,
            "max_length": self.config.max_length
        }


class CompletenessValidator(BaseValidator):
    """Validates data completeness"""
    
    def validate(self, data: HRMExample) -> ProcessingResult[HRMExample]:
        """Validate data completeness"""
        start_time = datetime.now()
        self.validation_count += 1
        
        errors = []
        completeness_score = 0.0
        total_checks = 0
        
        # Check required fields
        required_checks = [
            (data.inputs, "inputs"),
            (data.targets, "targets"),
            (data.puzzle_identifiers, "puzzle_identifiers"),
            (data.example_id, "example_id"),
            (data.source, "source")
        ]
        
        for field, name in required_checks:
            total_checks += 1
            if field:
                completeness_score += 1
            else:
                errors.append(f"Missing required field: {name}")
        
        # Check optional but recommended fields
        optional_checks = [
            (data.original_prompt, "original_prompt"),
            (data.original_completion, "original_completion"),
            (data.metadata, "metadata")
        ]
        
        for field, name in optional_checks:
            total_checks += 1
            if field:
                completeness_score += 1
        
        completeness_score /= total_checks
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        # Check against threshold
        if completeness_score < self.config.completeness_threshold:
            self.fail_count += 1
            return ProcessingResult(
                success=False,
                data=None,
                error=f"Completeness {completeness_score:.2f} < threshold {self.config.completeness_threshold}",
                metadata={"completeness_score": completeness_score, "errors": errors},
                processing_time=processing_time
            )
        
        self.pass_count += 1
        return ProcessingResult(
            success=True,
            data=data,
            error=None,
            metadata={"completeness_score": completeness_score},
            processing_time=processing_time
        )
    
    def get_validation_rules(self) -> Dict[str, Any]:
        return {
            "completeness_threshold": self.config.completeness_threshold,
            "required_fields": ["inputs", "targets", "puzzle_identifiers", "example_id", "source"]
        }


class QualityValidator(BaseValidator):
    """Validates overall data quality"""
    
    def validate(self, data: HRMExample) -> ProcessingResult[HRMExample]:
        """Validate data quality"""
        start_time = datetime.now()
        self.validation_count += 1
        
        quality_score = 1.0
        issues = []
        
        # Check for extremely short sequences
        if len(data.inputs) < 5 or len(data.targets) < 5:
            quality_score -= 0.2
            issues.append("Very short sequences")
        
        # Check for extreme length differences
        length_ratio = max(len(data.inputs), len(data.targets)) / min(len(data.inputs), len(data.targets))
        if length_ratio > 10:
            quality_score -= 0.2
            issues.append(f"Extreme length ratio: {length_ratio:.1f}")
        
        # Check for repetitive content
        if self._is_repetitive(data.inputs):
            quality_score -= 0.3
            issues.append("Repetitive input sequence")
        
        if self._is_repetitive(data.targets):
            quality_score -= 0.3
            issues.append("Repetitive target sequence")
        
        # Check for valid token ranges
        max_reasonable_token = 65536
        if max(data.inputs + data.targets) > max_reasonable_token:
            quality_score -= 0.4
            issues.append("Token IDs out of reasonable range")
        
        processing_time = (datetime.now() - start_time).total_seconds()
        
        if quality_score < self.config.quality_threshold:
            self.fail_count += 1
            return ProcessingResult(
                success=False,
                data=None,
                error=f"Quality score {quality_score:.2f} < threshold {self.config.quality_threshold}",
                metadata={"quality_score": quality_score, "issues": issues},
                processing_time=processing_time
            )
        
        self.pass_count += 1
        return ProcessingResult(
            success=True,
            data=data,
            error=None,
            metadata={"quality_score": quality_score},
            processing_time=processing_time
        )
    
    def get_validation_rules(self) -> Dict[str, Any]:
        return {
            "quality_threshold": self.config.quality_threshold,
            "checks": ["length", "ratio", "repetition", "token_range"]
        }
    
    def _is_repetitive(self, sequence: List[int], threshold: float = 0.8) -> bool:
        """Check if sequence is highly repetitive"""
        if len(sequence) < 10:
            return False
        
        unique_ratio = len(set(sequence)) / len(sequence)
        return unique_ratio < (1 - threshold)


class CompositeValidator(IDataValidator[HRMExample]):
    """Combines multiple validators"""
    
    def __init__(self, validators: List[IDataValidator[HRMExample]]):
        self.validators = validators
    
    def validate(self, data: HRMExample) -> ProcessingResult[HRMExample]:
        """Run all validators"""
        all_metadata = {}
        
        for validator in self.validators:
            result = validator.validate(data)
            
            # Collect metadata
            all_metadata[validator.__class__.__name__] = result.metadata
            
            # If any validator fails, return failure
            if not result.success:
                return ProcessingResult(
                    success=False,
                    data=None,
                    error=result.error,
                    metadata=all_metadata,
                    processing_time=result.processing_time
                )
        
        # All validators passed
        return ProcessingResult(
            success=True,
            data=data,
            error=None,
            metadata=all_metadata,
            processing_time=sum(v.validation_count for v in self.validators)
        )
    
    def get_validation_rules(self) -> Dict[str, Any]:
        """Get all validation rules"""
        rules = {}
        for validator in self.validators:
            rules[validator.__class__.__name__] = validator.get_validation_rules()
        return rules


class DuplicateFilter(IDataFilter[HRMExample]):
    """Filters duplicate examples"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.seen_ids: Set[str] = set()
        self.duplicate_count = 0
    
    def filter(self, data: List[HRMExample]) -> List[HRMExample]:
        """Filter duplicates from list"""
        filtered = []
        for example in data:
            if self.should_keep(example):
                filtered.append(example)
        return filtered
    
    def should_keep(self, item: HRMExample) -> bool:
        """Check if item should be kept"""
        if not self.config.check_duplicates:
            return True
        
        if item.example_id in self.seen_ids:
            self.duplicate_count += 1
            return False
        
        self.seen_ids.add(item.example_id)
        return True
    
    def get_statistics(self) -> Dict[str, int]:
        """Get filter statistics"""
        return {
            "unique_count": len(self.seen_ids),
            "duplicate_count": self.duplicate_count
        }


class EmptyFilter(IDataFilter[HRMExample]):
    """Filters empty or malformed examples"""
    
    def __init__(self, config: ValidationConfig):
        self.config = config
        self.filtered_count = 0
    
    def filter(self, data: List[HRMExample]) -> List[HRMExample]:
        """Filter empty examples from list"""
        filtered = []
        for example in data:
            if self.should_keep(example):
                filtered.append(example)
        return filtered
    
    def should_keep(self, item: HRMExample) -> bool:
        """Check if item should be kept"""
        if not self.config.filter_empty:
            return True
        
        # Check for empty sequences
        if not item.inputs or not item.targets:
            self.filtered_count += 1
            return False
        
        # Check for all-zero sequences
        if all(x == 0 for x in item.inputs):
            self.filtered_count += 1
            return False
        
        return True
    
    def get_statistics(self) -> Dict[str, int]:
        """Get filter statistics"""
        return {
            "filtered_count": self.filtered_count
        }

