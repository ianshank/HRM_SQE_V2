# HRM Data Integration System

Modular, dynamic, and reusable data integration pipeline for Hierarchical Reasoning Model (HRM) training data preparation.

## Features

✅ **Modular Architecture** - Clean separation of concerns with well-defined interfaces  
✅ **Configuration-Driven** - No hard-coded values, all parameters externalized  
✅ **Incremental Testing** - Comprehensive unit and integration tests  
✅ **Multiple Data Sources** - Support for various JSONL file formats  
✅ **Flexible Transformations** - Configurable tokenization and sequence preparation  
✅ **Data Quality Validation** - Multi-stage validation with quality metrics  
✅ **Extensible** - Easy to add new parsers, transformers, and validators  

## Architecture

```
hrm_data_integration/
├── config/               # Configuration schemas and loaders
│   ├── schemas.py       # Pydantic models for configuration
│   ├── loader.py        # YAML configuration loader
│   └── default_config.yaml
├── core/                # Core interfaces and abstractions
│   └── interfaces.py    # Abstract base classes and protocols
├── models/              # Data models
│   └── hrm_data.py      # HRM-specific data structures
├── parsers/             # Data parsers
│   └── jsonl_parser.py  # JSONL file parser
├── transformers/        # Data transformers
│   └── hrm_transformer.py
├── validators/          # Data validators
│   └── data_validators.py
├── pipeline/            # Pipeline orchestration
│   └── orchestrator.py
├── tests/               # Unit and integration tests
│   └── test_pipeline.py
└── main.py              # CLI entry point
```

## Installation

```bash
# Install dependencies
pip install pydantic pyyaml tqdm pytest

# Run from project root
cd C:\Users\iansh\Documents\HRM
```

## Quick Start

### 1. Using Default Configuration

```bash
python -m hrm_data_integration.main --default
```

### 2. Using Custom Configuration

```bash
python -m hrm_data_integration.main --config path/to/config.yaml
```

### 3. Command-Line Configuration

```bash
python -m hrm_data_integration.main \
  --data-path "C:/path/to/data" \
  --pattern "*.jsonl" \
  --output-dir "C:/path/to/output" \
  --max-length 512 \
  --batch-size 100
```

### 4. Programmatic Usage

```python
from hrm_data_integration import ConfigBuilder, HRMDataPipeline

# Build configuration
config = (ConfigBuilder()
    .set_name("my_pipeline")
    .add_source("dataset1", "/path/to/data", pattern="*.jsonl")
    .set_output_dir("/path/to/output")
    .set_transformation(
        max_length=512,
        vocab_size=65536,
        sequence_length=512
    )
    .set_validation(
        quality_threshold=0.8,
        check_duplicates=True
    )
    .build())

# Run pipeline
pipeline = HRMDataPipeline(config)
result = pipeline.execute()

if result.success:
    print(f"Pipeline completed: {result.data.total_examples} examples")
```

## Configuration

### Data Sources

```yaml
sources:
  - name: "security_dataset"
    path: "c:/data/security"
    format: "jsonl"
    pattern: "*.jsonl"
    prompt_field: "prompt"
    completion_field: "completion"
    metadata_fields: ["agent_type", "domain"]
```

### Transformation

```yaml
transformation:
  max_length: 512
  truncate_strategy: "end"  # start, end, or middle
  tokenizer_type: "character"
  vocab_size: 65536
  special_tokens:
    "<PAD>": 0
    "<UNK>": 1
    "<BOS>": 2
    "<EOS>": 3
```

### Validation

```yaml
validation:
  min_length: 10
  max_length: 100000
  check_duplicates: true
  filter_empty: true
  quality_threshold: 0.8
  completeness_threshold: 0.9
```

## Testing

### Run All Tests

```bash
pytest hrm_data_integration/tests/ -v
```

### Run Specific Test

```bash
pytest hrm_data_integration/tests/test_pipeline.py::TestTransformer -v
```

### Run with Coverage

```bash
pytest hrm_data_integration/tests/ --cov=hrm_data_integration --cov-report=html
```

## Pipeline Stages

### 1. Parsing Stage

- Reads data from multiple sources
- Parses JSONL files
- Handles field name variations
- Collects parsing statistics

### 2. Transformation Stage

- Tokenizes text (character, word, or custom)
- Prepares sequences for HRM format
- Assigns puzzle identifiers
- Truncates to maximum length

### 3. Validation Stage

- Length validation
- Completeness checking
- Quality scoring
- Duplicate detection
- Empty filtering

### 4. Output Stage

- Creates HRM datasets
- Computes statistics
- Splits into train/val/test
- Saves to JSONL format

## Data Format

### Input Format (JSONL)

```json
{
  "prompt": "What is the capital of France?",
  "completion": "Paris",
  "agent_type": "qa",
  "domain": "geography"
}
```

### Output Format (HRM JSONL)

```json
{
  "inputs": [72, 101, 108, 108, 111],
  "targets": [87, 111, 114, 108, 100],
  "puzzle_identifiers": [1],
  "example_id": "abc123",
  "source": "dataset_name",
  "data_type": "reasoning",
  "metadata": {"agent_type": "qa"}
}
```

## Extending the Pipeline

### Add Custom Parser

```python
from hrm_data_integration.core.interfaces import IDataParser, ProcessingResult

class CustomParser(IDataParser):
    def parse(self, data):
        # Custom parsing logic
        return ProcessingResult(success=True, data=parsed_data)
    
    def can_parse(self, data):
        # Validation logic
        return True
```

### Add Custom Validator

```python
from hrm_data_integration.validators.data_validators import BaseValidator

class CustomValidator(BaseValidator):
    def validate(self, data):
        # Custom validation logic
        return ProcessingResult(success=True, data=data)
```

### Add Custom Transformer

```python
from hrm_data_integration.core.interfaces import IDataTransformer

class CustomTransformer(IDataTransformer):
    def transform(self, data):
        # Custom transformation logic
        return ProcessingResult(success=True, data=transformed_data)
```

## CLI Options

```
--config PATH              Path to configuration YAML file
--default                  Use default configuration
--output-dir PATH          Override output directory
--batch-size N             Override batch size
--max-length N             Override maximum sequence length
--log-level LEVEL          Set log level (DEBUG, INFO, WARNING, ERROR)
--data-path PATH           Path to data directory
--pattern PATTERN          File pattern to match
--validate-only            Only validate configuration
--dry-run                  Show what would be processed without running
```

## Monitoring and Metrics

The pipeline automatically collects:

- Processing time per stage
- Success/failure rates
- Data quality metrics
- Validation statistics
- Transformation statistics

Metrics are saved to:
- Console (real-time)
- Log file (if configured)
- JSON metrics file (if enabled)

## Error Handling

The pipeline supports three error handling modes:

- `continue`: Log errors and continue processing
- `stop`: Stop on first error
- `skip`: Skip problematic records

Configure in YAML:
```yaml
error_handling: "continue"
max_errors: 100
```

## Best Practices

1. **Always validate configuration** before running on large datasets
2. **Use dry-run mode** to preview what will be processed
3. **Monitor logs** for errors and warnings
4. **Review quality metrics** after each run
5. **Test incrementally** with small datasets first
6. **Version your configurations** for reproducibility

## Troubleshooting

### Common Issues

**Issue**: "No files found matching pattern"
- Check path and pattern in configuration
- Verify files exist and are readable

**Issue**: "Field not found in data"
- Check field name variations
- Verify JSONL structure matches expected format

**Issue**: "Validation failed"
- Review validation thresholds in configuration
- Check data quality metrics in logs

**Issue**: "Memory error"
- Reduce batch size
- Process in smaller chunks
- Filter data earlier in pipeline

## Performance Tips

- Use `batch_size` to control memory usage
- Enable `parallel` mode for large datasets
- Filter early to reduce processing overhead
- Use sampling for development/testing
- Monitor memory and CPU usage

## License

MIT License - See LICENSE file for details

## Contributing

Contributions welcome! Please:
1. Follow PEP 8 style guidelines
2. Add unit tests for new features
3. Update documentation
4. Use type hints
5. Write docstrings

## Support

For issues or questions:
- Check the documentation
- Review test examples
- Check logs for error details
- Create an issue with reproducible example

