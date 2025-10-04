"""
Configuration Loader
====================

Loads and validates pipeline configurations from YAML files.
Supports environment variable substitution and defaults.

Author: Data Integration Team
Date: October 2025
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional

from .schemas import PipelineConfig
from ..core.interfaces import IConfigLoader


class YAMLConfigLoader(IConfigLoader[PipelineConfig]):
    """Loads pipeline configuration from YAML files"""
    
    def __init__(self, defaults_path: Optional[Path] = None):
        self.defaults_path = defaults_path or Path(__file__).parent / "default_config.yaml"
    
    def load(self, config_path: Path) -> PipelineConfig:
        """Load configuration from YAML file"""
        # Load config file
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
        
        # Substitute environment variables
        config_data = self._substitute_env_vars(config_data)
        
        # Create PipelineConfig with validation
        try:
            pipeline_config = PipelineConfig(**config_data)
            return pipeline_config
        except Exception as e:
            raise ValueError(f"Invalid configuration: {e}")
    
    def validate(self, config: PipelineConfig) -> bool:
        """Validate configuration"""
        try:
            # Check required fields
            if not config.sources:
                return False
            
            # Validate all sources exist
            for source in config.sources:
                if not Path(source.path).exists():
                    print(f"Warning: Source path does not exist: {source.path}")
            
            # Validate output directory
            if not config.output_dir.exists():
                print(f"Creating output directory: {config.output_dir}")
                config.output_dir.mkdir(parents=True, exist_ok=True)
            
            # Validate split ratios
            total = sum(config.split_ratios.values())
            if not (0.99 <= total <= 1.01):
                return False
            
            return True
            
        except Exception:
            return False
    
    def get_defaults(self) -> PipelineConfig:
        """Load default configuration"""
        return self.load(self.defaults_path)
    
    def _substitute_env_vars(self, data: Any) -> Any:
        """Recursively substitute environment variables in configuration"""
        if isinstance(data, dict):
            return {k: self._substitute_env_vars(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._substitute_env_vars(item) for item in data]
        elif isinstance(data, str):
            # Check for environment variable pattern: ${VAR_NAME}
            if data.startswith('${') and data.endswith('}'):
                var_name = data[2:-1]
                return os.environ.get(var_name, data)
            return data
        else:
            return data
    
    def merge_configs(self, base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
        """Merge two configurations, with override taking precedence"""
        merged = base.copy()
        
        for key, value in override.items():
            if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
                merged[key] = self.merge_configs(merged[key], value)
            else:
                merged[key] = value
        
        return merged
    
    def save(self, config: PipelineConfig, output_path: Path) -> None:
        """Save configuration to YAML file"""
        # Convert to dict
        config_dict = config.model_dump()
        
        # Convert Path objects to strings
        config_dict = self._convert_paths_to_str(config_dict)
        
        # Write to file
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
    
    def _convert_paths_to_str(self, data: Any) -> Any:
        """Convert Path objects to strings for YAML serialization"""
        if isinstance(data, Path):
            return str(data)
        elif isinstance(data, dict):
            return {k: self._convert_paths_to_str(v) for k, v in data.items()}
        elif isinstance(data, list):
            return [self._convert_paths_to_str(item) for item in data]
        else:
            return data


class ConfigBuilder:
    """Builder pattern for creating pipeline configurations programmatically"""
    
    def __init__(self):
        self._config_data = {
            "pipeline_name": "custom_pipeline",
            "version": "1.0.0",
            "sources": [],
            "output_dir": "output",
            "transformation": {},
            "validation": {},
            "split_ratios": {"train": 0.8, "val": 0.1, "test": 0.1}
        }
    
    def set_name(self, name: str) -> 'ConfigBuilder':
        """Set pipeline name"""
        self._config_data["pipeline_name"] = name
        return self
    
    def add_source(self, name: str, path: str, **kwargs) -> 'ConfigBuilder':
        """Add a data source"""
        source = {
            "name": name,
            "path": path,
            "format": kwargs.get("format", "jsonl"),
            "pattern": kwargs.get("pattern", "*.jsonl"),
            "encoding": kwargs.get("encoding", "utf-8"),
            "prompt_field": kwargs.get("prompt_field", "prompt"),
            "completion_field": kwargs.get("completion_field", "completion"),
            "metadata_fields": kwargs.get("metadata_fields", [])
        }
        self._config_data["sources"].append(source)
        return self
    
    def set_output_dir(self, path: str) -> 'ConfigBuilder':
        """Set output directory"""
        self._config_data["output_dir"] = path
        return self
    
    def set_transformation(self, **kwargs) -> 'ConfigBuilder':
        """Set transformation parameters"""
        self._config_data["transformation"] = kwargs
        return self
    
    def set_validation(self, **kwargs) -> 'ConfigBuilder':
        """Set validation parameters"""
        self._config_data["validation"] = kwargs
        return self
    
    def set_split_ratios(self, train: float, val: float, test: float) -> 'ConfigBuilder':
        """Set train/val/test split ratios"""
        self._config_data["split_ratios"] = {"train": train, "val": val, "test": test}
        return self
    
    def build(self) -> PipelineConfig:
        """Build the configuration"""
        return PipelineConfig(**self._config_data)

