"""
HRM Data Integration - Main Entry Point
========================================

Main script for running the HRM data integration pipeline.
Supports command-line arguments for flexible configuration.

Usage:
    python -m hrm_data_integration.main --config path/to/config.yaml
    python -m hrm_data_integration.main --default

Author: Data Integration Team
Date: October 2025
"""

import argparse
import sys
from pathlib import Path
from datetime import datetime

from .config.loader import YAMLConfigLoader, ConfigBuilder
from .pipeline.orchestrator import HRMDataPipeline


def parse_arguments():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(
        description="HRM Data Integration Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    # Configuration
    parser.add_argument(
        "--config",
        type=Path,
        help="Path to configuration YAML file"
    )
    
    parser.add_argument(
        "--default",
        action="store_true",
        help="Use default configuration"
    )
    
    # Override options
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Override output directory"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Override batch size"
    )
    
    parser.add_argument(
        "--max-length",
        type=int,
        help="Override maximum sequence length"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Override log level"
    )
    
    # Data source options
    parser.add_argument(
        "--data-path",
        type=Path,
        help="Path to data directory (adds all JSONL files)"
    )
    
    parser.add_argument(
        "--pattern",
        type=str,
        default="*.jsonl",
        help="File pattern to match in data directory"
    )
    
    # Execution options
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration without running pipeline"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Perform dry run (validate and show what would be processed)"
    )
    
    return parser.parse_args()


def create_config_from_args(args):
    """Create pipeline configuration from command-line arguments"""
    loader = YAMLConfigLoader()
    
    # Load base configuration
    if args.config:
        config = loader.load(args.config)
    elif args.default:
        config = loader.get_defaults()
    elif args.data_path:
        # Build configuration from data path
        builder = ConfigBuilder()
        builder.set_name(f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        builder.add_source(
            name="custom_data",
            path=str(args.data_path),
            pattern=args.pattern
        )
        builder.set_output_dir(str(args.output_dir or "output"))
        config = builder.build()
    else:
        print("Error: Must specify --config, --default, or --data-path")
        sys.exit(1)
    
    # Apply overrides
    if args.output_dir:
        config.output_dir = args.output_dir
    
    if args.batch_size:
        config.batch_size = args.batch_size
    
    if args.max_length:
        config.transformation.max_length = args.max_length
    
    if args.log_level:
        config.log_level = args.log_level
    
    return config, loader


def print_config_summary(config):
    """Print configuration summary"""
    print("\n" + "="*80)
    print("HRM DATA INTEGRATION PIPELINE")
    print("="*80)
    print(f"\nPipeline: {config.pipeline_name} (v{config.version})")
    if config.description:
        print(f"Description: {config.description}")
    
    print(f"\n📁 Data Sources ({len(config.sources)}):")
    for source in config.sources:
        print(f"  - {source.name}: {source.path} ({source.pattern})")
    
    print(f"\n⚙️  Processing:")
    print(f"  - Mode: {config.processing_mode}")
    print(f"  - Batch size: {config.batch_size}")
    print(f"  - Workers: {config.num_workers}")
    
    print(f"\n🔄 Transformation:")
    print(f"  - Tokenizer: {config.transformation.tokenizer_type}")
    print(f"  - Vocab size: {config.transformation.vocab_size}")
    print(f"  - Max length: {config.transformation.max_length}")
    print(f"  - Sequence length: {config.transformation.sequence_length}")
    
    print(f"\n✅ Validation:")
    print(f"  - Check duplicates: {config.validation.check_duplicates}")
    print(f"  - Quality threshold: {config.validation.quality_threshold}")
    print(f"  - Completeness threshold: {config.validation.completeness_threshold}")
    
    print(f"\n📤 Output:")
    print(f"  - Directory: {config.output_dir}")
    print(f"  - Format: {config.output_format}")
    print(f"  - Split ratios: {config.split_ratios}")
    
    print(f"\n📊 Monitoring:")
    print(f"  - Log level: {config.log_level}")
    if config.log_file:
        print(f"  - Log file: {config.log_file}")
    if config.enable_metrics:
        print(f"  - Metrics: {config.metrics_output}")
    
    print("\n" + "="*80 + "\n")


def main():
    """Main entry point"""
    args = parse_arguments()
    
    # Create configuration
    config, loader = create_config_from_args(args)
    
    # Print summary
    print_config_summary(config)
    
    # Validate configuration
    if not loader.validate(config):
        print("❌ Configuration validation failed!")
        sys.exit(1)
    
    print("✅ Configuration validated successfully")
    
    # Validate-only mode
    if args.validate_only:
        print("\n✓ Validation complete (--validate-only mode)")
        return 0
    
    # Dry-run mode
    if args.dry_run:
        print("\n🔍 DRY RUN MODE - No data will be processed\n")
        
        # Count records that would be processed
        total_records = 0
        for source in config.sources:
            from .parsers.jsonl_parser import JSONLDataSource
            try:
                data_source = JSONLDataSource(source)
                count = data_source.count()
                total_records += count
                print(f"  {source.name}: {count} records")
            except Exception as e:
                print(f"  {source.name}: Error - {e}")
        
        print(f"\nTotal records to process: {total_records}")
        print("\n✓ Dry run complete")
        return 0
    
    # Execute pipeline
    print("\n🚀 Starting pipeline execution...\n")
    
    try:
        pipeline = HRMDataPipeline(config)
        result = pipeline.execute()
        
        if result.success:
            print("\n" + "="*80)
            print("✅ PIPELINE COMPLETED SUCCESSFULLY")
            print("="*80)
            print(f"\nTotal processing time: {result.processing_time:.2f}s")
            
            # Print stage results
            print("\n📊 Stage Results:")
            for stage_name, stage_result in pipeline.get_stage_results().items():
                status = "✅" if stage_result.success else "❌"
                print(f"  {status} {stage_name}: {stage_result.processing_time:.2f}s")
                if stage_result.metadata:
                    for key, value in stage_result.metadata.items():
                        if key != "processing_time":
                            print(f"      {key}: {value}")
            
            print("\n✓ Pipeline execution complete")
            return 0
        else:
            print("\n" + "="*80)
            print("❌ PIPELINE FAILED")
            print("="*80)
            print(f"\nError: {result.error}")
            return 1
            
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())

