"""
Unit Tests for HRM Data Integration Pipeline
=============================================

Comprehensive tests for all pipeline components.
Uses pytest framework with fixtures and parametrization.

Author: Data Integration Team
Date: October 2025
"""

import pytest
import json
import tempfile
from pathlib import Path
from typing import List

from ..config.schemas import (
    PipelineConfig, DataSourceConfig, TransformationConfig, ValidationConfig
)
from ..config.loader import YAMLConfigLoader, ConfigBuilder
from ..parsers.jsonl_parser import JSONLParser, JSONLDataSource
from ..transformers.hrm_transformer import HRMDataTransformer, CharacterTokenizer
from ..validators.data_validators import LengthValidator, CompletenessValidator, QualityValidator
from ..models.hrm_data import HRMExample, HRMDataset
from ..core.interfaces import DataRecord


@pytest.fixture
def temp_data_dir(tmp_path):
    """Create temporary directory with sample JSONL data"""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    
    # Create sample JSONL file
    sample_data = [
        {"prompt": "What is 2+2?", "completion": "4", "agent_type": "math"},
        {"prompt": "Define recursion", "completion": "A function that calls itself", "agent_type": "cs"},
        {"prompt": "Hello", "completion": "Hi there!", "agent_type": "chat"},
    ]
    
    test_file = data_dir / "test_data.jsonl"
    with open(test_file, 'w') as f:
        for item in sample_data:
            f.write(json.dumps(item) + '\n')
    
    return data_dir


@pytest.fixture
def sample_config(temp_data_dir):
    """Create sample pipeline configuration"""
    return PipelineConfig(
        pipeline_name="test_pipeline",
        sources=[
            DataSourceConfig(
                name="test_source",
                path=temp_data_dir,
                pattern="*.jsonl",
                prompt_field="prompt",
                completion_field="completion"
            )
        ],
        output_dir=temp_data_dir / "output",
        transformation=TransformationConfig(
            max_length=100,
            vocab_size=1000,
            sequence_length=50
        ),
        validation=ValidationConfig(
            min_length=1,
            max_length=200
        ),
        split_ratios={"train": 0.8, "val": 0.1, "test": 0.1}
    )


@pytest.fixture
def sample_data_record():
    """Create sample data record"""
    from datetime import datetime
    return DataRecord(
        id="test123",
        prompt="What is AI?",
        completion="Artificial Intelligence",
        metadata={"domain": "tech"},
        source="test",
        timestamp=datetime.now()
    )


class TestConfigLoader:
    """Test configuration loading and validation"""
    
    def test_yaml_loader(self, temp_data_dir):
        """Test YAML configuration loading"""
        # Create config file
        config_file = temp_data_dir / "config.yaml"
        config_data = {
            "pipeline_name": "test",
            "sources": [{
                "name": "test",
                "path": str(temp_data_dir),
                "pattern": "*.jsonl"
            }],
            "output_dir": str(temp_data_dir / "output")
        }
        
        import yaml
        with open(config_file, 'w') as f:
            yaml.dump(config_data, f)
        
        # Load and validate
        loader = YAMLConfigLoader()
        config = loader.load(config_file)
        
        assert config.pipeline_name == "test"
        assert len(config.sources) == 1
        assert loader.validate(config)
    
    def test_config_builder(self, temp_data_dir):
        """Test programmatic configuration building"""
        builder = ConfigBuilder()
        config = (builder
                  .set_name("custom_pipeline")
                  .add_source("source1", str(temp_data_dir))
                  .set_output_dir(str(temp_data_dir / "output"))
                  .set_split_ratios(0.7, 0.2, 0.1)
                  .build())
        
        assert config.pipeline_name == "custom_pipeline"
        assert len(config.sources) == 1
        assert config.split_ratios["train"] == 0.7


class TestJSONLParser:
    """Test JSONL parsing functionality"""
    
    def test_data_source(self, temp_data_dir):
        """Test JSONL data source"""
        source_config = DataSourceConfig(
            name="test",
            path=temp_data_dir,
            pattern="*.jsonl"
        )
        
        source = JSONLDataSource(source_config)
        
        assert source.validate_source()
        assert source.count() == 3
        
        lines = list(source.read())
        assert len(lines) == 3
    
    def test_parser(self, temp_data_dir):
        """Test JSONL parser"""
        source_config = DataSourceConfig(
            name="test",
            path=temp_data_dir,
            pattern="*.jsonl",
            prompt_field="prompt",
            completion_field="completion"
        )
        
        parser = JSONLParser(source_config)
        
        test_line = '{"prompt": "Test", "completion": "Result"}'
        result = parser.parse(test_line)
        
        assert result.success
        assert result.data is not None
        assert result.data.prompt == "Test"
        assert result.data.completion == "Result"
    
    def test_parser_with_variants(self):
        """Test parser with field name variations"""
        source_config = DataSourceConfig(
            name="test",
            path=".",
            prompt_field="prompt",
            completion_field="completion"
        )
        
        parser = JSONLParser(source_config)
        
        # Test alternative field names
        test_cases = [
            '{"input": "Q", "output": "A"}',
            '{"question": "Q", "answer": "A"}',
            '{"instruction": "Q", "response": "A"}'
        ]
        
        for test_line in test_cases:
            result = parser.parse(test_line)
            assert result.success, f"Failed to parse: {test_line}"


class TestTokenizer:
    """Test tokenization"""
    
    def test_character_tokenizer(self):
        """Test character-level tokenizer"""
        tokenizer = CharacterTokenizer(vocab_size=1000)
        
        text = "Hello World"
        tokens = tokenizer.encode(text)
        
        assert len(tokens) == len(text)
        assert all(isinstance(t, int) for t in tokens)
        
        # Test decode
        decoded = tokenizer.decode(tokens)
        assert decoded == text
    
    def test_special_tokens(self):
        """Test special tokens"""
        special_tokens = {"<PAD>": 0, "<UNK>": 1}
        tokenizer = CharacterTokenizer(special_tokens=special_tokens)
        
        assert tokenizer.get_special_tokens() == special_tokens
        assert tokenizer.get_vocab_size() == 65536


class TestTransformer:
    """Test data transformation"""
    
    def test_hrm_transformer(self, sample_data_record):
        """Test HRM data transformer"""
        config = TransformationConfig(
            max_length=100,
            vocab_size=1000,
            sequence_length=50
        )
        
        transformer = HRMDataTransformer(config)
        result = transformer.transform(sample_data_record)
        
        assert result.success
        assert result.data is not None
        assert isinstance(result.data, HRMExample)
        assert len(result.data.inputs) > 0
        assert len(result.data.targets) > 0
    
    def test_truncation_strategies(self, sample_data_record):
        """Test different truncation strategies"""
        strategies = ["end", "start", "middle"]
        
        for strategy in strategies:
            config = TransformationConfig(
                max_length=10,
                truncate_strategy=strategy,
                vocab_size=1000
            )
            
            transformer = HRMDataTransformer(config)
            result = transformer.transform(sample_data_record)
            
            assert result.success
            assert len(result.data.inputs) <= 10


class TestValidators:
    """Test data validation"""
    
    def test_length_validator(self):
        """Test length validator"""
        config = ValidationConfig(min_length=5, max_length=100)
        validator = LengthValidator(config)
        
        # Valid example
        valid_example = HRMExample(
            inputs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            targets=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            puzzle_identifiers=[1],
            example_id="test1",
            source="test",
            data_type="reasoning",
            length=10
        )
        
        result = validator.validate(valid_example)
        assert result.success
        
        # Invalid example (too short)
        invalid_example = HRMExample(
            inputs=[1, 2],
            targets=[1, 2],
            puzzle_identifiers=[1],
            example_id="test2",
            source="test",
            data_type="reasoning",
            length=2
        )
        
        result = validator.validate(invalid_example)
        assert not result.success
    
    def test_completeness_validator(self):
        """Test completeness validator"""
        config = ValidationConfig(completeness_threshold=0.8)
        validator = CompletenessValidator(config)
        
        complete_example = HRMExample(
            inputs=[1, 2, 3],
            targets=[1, 2, 3],
            puzzle_identifiers=[1],
            example_id="test",
            source="test",
            data_type="reasoning",
            length=3,
            original_prompt="test",
            original_completion="test"
        )
        
        result = validator.validate(complete_example)
        assert result.success
    
    def test_quality_validator(self):
        """Test quality validator"""
        config = ValidationConfig(quality_threshold=0.7)
        validator = QualityValidator(config)
        
        # Good quality example
        good_example = HRMExample(
            inputs=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            targets=[11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
            puzzle_identifiers=[1],
            example_id="test",
            source="test",
            data_type="reasoning",
            length=10
        )
        
        result = validator.validate(good_example)
        assert result.success


class TestHRMDataset:
    """Test HRM dataset functionality"""
    
    def test_dataset_creation(self):
        """Test dataset creation and statistics"""
        examples = [
            HRMExample(
                inputs=[1, 2, 3],
                targets=[4, 5, 6],
                puzzle_identifiers=[1],
                example_id=f"test{i}",
                source="test",
                data_type="reasoning",
                length=3
            )
            for i in range(10)
        ]
        
        dataset = HRMDataset(
            dataset_id="test_dataset",
            examples=examples,
            name="test",
            vocab_size=1000
        )
        
        stats = dataset.compute_statistics()
        
        assert dataset.total_examples == 10
        assert stats["avg_length"] == 3.0
    
    def test_dataset_splitting(self):
        """Test dataset splitting"""
        examples = [
            HRMExample(
                inputs=[1, 2, 3],
                targets=[4, 5, 6],
                puzzle_identifiers=[1],
                example_id=f"test{i}",
                source="test",
                data_type="reasoning",
                length=3
            )
            for i in range(100)
        ]
        
        dataset = HRMDataset(
            dataset_id="test_dataset",
            examples=examples,
            name="test",
            vocab_size=1000
        )
        
        splits = dataset.split({"train": 0.8, "val": 0.1, "test": 0.1})
        
        assert "train" in splits
        assert "val" in splits
        assert "test" in splits
        assert len(splits["train"].examples) == 80
        assert len(splits["val"].examples) == 10
        assert len(splits["test"].examples) == 10


class TestPipeline:
    """Test complete pipeline"""
    
    @pytest.mark.slow
    def test_full_pipeline(self, sample_config):
        """Test full pipeline execution"""
        from ..pipeline.orchestrator import HRMDataPipeline
        
        pipeline = HRMDataPipeline(sample_config)
        
        # Validate pipeline
        assert pipeline.validate_pipeline()
        
        # Execute pipeline
        result = pipeline.execute()
        
        assert result.success
        assert result.data is not None
        
        # Check outputs exist
        for split_name in ["train", "val", "test"]:
            output_file = sample_config.output_dir / f"{split_name}.jsonl"
            assert output_file.exists()


def test_integration_example(temp_data_dir):
    """
    Integration test demonstrating complete workflow
    """
    # Build configuration
    config = ConfigBuilder() \
        .set_name("integration_test") \
        .add_source("test_data", str(temp_data_dir), pattern="*.jsonl") \
        .set_output_dir(str(temp_data_dir / "output")) \
        .set_transformation(
            max_length=100,
            vocab_size=1000,
            sequence_length=50
        ) \
        .set_validation(
            min_length=1,
            max_length=200,
            quality_threshold=0.5
        ) \
        .build()
    
    # Run pipeline
    from ..pipeline.orchestrator import HRMDataPipeline
    pipeline = HRMDataPipeline(config)
    result = pipeline.execute()
    
    # Verify success
    assert result.success
    
    # Load and verify output
    train_file = config.output_dir / "train.jsonl"
    assert train_file.exists()
    
    dataset = HRMDataset.load(str(train_file))
    assert len(dataset.examples) > 0
    
    # Verify data quality
    for example in dataset.examples:
        assert len(example.inputs) > 0
        assert len(example.targets) > 0
        assert example.example_id
        assert example.source


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

