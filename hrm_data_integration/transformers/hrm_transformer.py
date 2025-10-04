"""
HRM Data Transformer
====================

Transforms parsed data records into HRM training format.
Handles tokenization, sequence preparation, and puzzle ID assignment.

Author: Data Integration Team
Date: October 2025
"""

from typing import List, Dict, Any, Optional
from datetime import datetime
import hashlib

from ..core.interfaces import IDataTransformer, ITokenizer, ProcessingResult, DataRecord
from ..models.hrm_data import HRMExample, HRMDataType
from ..config.schemas import TransformationConfig


class CharacterTokenizer(ITokenizer):
    """Simple character-level tokenizer"""
    
    def __init__(self, vocab_size: int = 65536, special_tokens: Optional[Dict[str, int]] = None):
        self.vocab_size = vocab_size
        self.special_tokens = special_tokens or {
            "<PAD>": 0,
            "<UNK>": 1,
            "<BOS>": 2,
            "<EOS>": 3,
            "<SEP>": 4
        }
        
        # Reserve space for special tokens
        self.char_offset = len(self.special_tokens)
    
    def encode(self, text: str) -> List[int]:
        """Encode text to token IDs"""
        tokens = []
        for char in text:
            char_code = ord(char)
            # Check if within vocab range
            if char_code + self.char_offset < self.vocab_size:
                tokens.append(char_code + self.char_offset)
            else:
                tokens.append(self.special_tokens["<UNK>"])
        return tokens
    
    def decode(self, token_ids: List[int]) -> str:
        """Decode token IDs to text"""
        chars = []
        for token_id in token_ids:
            # Skip special tokens
            if token_id < self.char_offset:
                continue
            # Decode character
            char_code = token_id - self.char_offset
            if char_code < 1114112:  # Max Unicode code point
                chars.append(chr(char_code))
        return ''.join(chars)
    
    def get_vocab_size(self) -> int:
        return self.vocab_size
    
    def get_special_tokens(self) -> Dict[str, int]:
        return self.special_tokens.copy()


class HRMDataTransformer(IDataTransformer[DataRecord, HRMExample]):
    """Transform DataRecord to HRM training format"""
    
    def __init__(self, config: TransformationConfig, tokenizer: Optional[ITokenizer] = None):
        self.config = config
        self.tokenizer = tokenizer or CharacterTokenizer(
            vocab_size=config.vocab_size,
            special_tokens=config.special_tokens
        )
        
        # Puzzle ID mapping
        self.puzzle_id_map: Dict[str, int] = {}
        self.next_puzzle_id = 0
        
        # Statistics
        self.transform_count = 0
        self.error_count = 0
    
    def transform(self, data: DataRecord) -> ProcessingResult[HRMExample]:
        """Transform DataRecord to HRMExample"""
        start_time = datetime.now()
        
        try:
            # Validate input
            if not self.validate_input(data):
                raise ValueError(f"Invalid input data: {data.id}")
            
            # Tokenize prompt and completion
            prompt_tokens = self.tokenizer.encode(data.prompt)
            completion_tokens = self.tokenizer.encode(data.completion)
            
            # Add special tokens
            inputs = self._prepare_inputs(prompt_tokens)
            targets = self._prepare_targets(completion_tokens)
            
            # Handle sequence length
            if self.config.max_length:
                inputs = self._truncate_sequence(inputs, self.config.max_length)
                targets = self._truncate_sequence(targets, self.config.max_length)
            
            # Get or create puzzle ID
            puzzle_id = self._get_puzzle_id(data)
            
            # Determine data type
            data_type = self._infer_data_type(data)
            
            # Create HRM example
            example = HRMExample(
                inputs=inputs,
                targets=targets,
                puzzle_identifiers=[puzzle_id],
                example_id=data.id,
                source=data.source,
                data_type=data_type,
                length=len(inputs),
                original_prompt=data.prompt,
                original_completion=data.completion,
                metadata=data.metadata
            )
            
            # Validate output
            if not self.validate_output(example):
                raise ValueError(f"Invalid output for example: {data.id}")
            
            self.transform_count += 1
            
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=True,
                data=example,
                error=None,
                metadata={
                    "input_length": len(inputs),
                    "target_length": len(targets),
                    "puzzle_id": puzzle_id
                },
                processing_time=processing_time
            )
            
        except Exception as e:
            self.error_count += 1
            processing_time = (datetime.now() - start_time).total_seconds()
            
            return ProcessingResult(
                success=False,
                data=None,
                error=str(e),
                metadata={"record_id": data.id},
                processing_time=processing_time
            )
    
    def validate_input(self, data: DataRecord) -> bool:
        """Validate input data"""
        # Check non-empty
        if not data.prompt or not data.completion:
            return False
        
        return True
    
    def validate_output(self, data: HRMExample) -> bool:
        """Validate output data"""
        # Check non-empty sequences
        if not data.inputs or not data.targets:
            return False
        
        # Check vocab range
        max_token = max(data.inputs + data.targets)
        if max_token >= self.tokenizer.get_vocab_size():
            return False
        
        # Check puzzle IDs
        if not data.puzzle_identifiers or data.puzzle_identifiers[0] < 0:
            return False
        
        return True
    
    def _prepare_inputs(self, tokens: List[int]) -> List[int]:
        """Prepare input sequence with special tokens"""
        special_tokens = self.tokenizer.get_special_tokens()
        
        # Add BOS token if configured
        if "<BOS>" in special_tokens and self.config.normalize_whitespace:
            return [special_tokens["<BOS>"]] + tokens
        
        return tokens
    
    def _prepare_targets(self, tokens: List[int]) -> List[int]:
        """Prepare target sequence with special tokens"""
        special_tokens = self.tokenizer.get_special_tokens()
        
        # Add EOS token if configured
        if "<EOS>" in special_tokens and self.config.normalize_whitespace:
            return tokens + [special_tokens["<EOS>"]]
        
        return tokens
    
    def _truncate_sequence(self, tokens: List[int], max_length: int) -> List[int]:
        """Truncate sequence based on strategy"""
        if len(tokens) <= max_length:
            return tokens
        
        strategy = self.config.truncate_strategy
        
        if strategy == "end":
            return tokens[:max_length]
        elif strategy == "start":
            return tokens[-max_length:]
        elif strategy == "middle":
            # Keep start and end, remove middle
            keep_size = max_length // 2
            return tokens[:keep_size] + tokens[-keep_size:]
        else:
            return tokens[:max_length]
    
    def _get_puzzle_id(self, data: DataRecord) -> int:
        """Get or create puzzle ID for data record"""
        strategy = self.config.puzzle_id_strategy
        
        if strategy == "hash":
            # Use hash of source + metadata
            key = f"{data.source}_{data.metadata.get('agent_type', 'default')}"
            
            if key not in self.puzzle_id_map:
                self.puzzle_id_map[key] = self.next_puzzle_id
                self.next_puzzle_id += 1
            
            return self.puzzle_id_map[key]
        
        elif strategy == "index":
            # Simple incremental index
            if data.source not in self.puzzle_id_map:
                self.puzzle_id_map[data.source] = self.next_puzzle_id
                self.next_puzzle_id += 1
            
            return self.puzzle_id_map[data.source]
        
        elif strategy == "custom":
            # Use metadata if available
            custom_id = data.metadata.get('puzzle_id')
            if custom_id is not None:
                return int(custom_id)
            
            # Fallback to hash
            return hash(data.source) % self.config.vocab_size
        
        else:
            return 0
    
    def _infer_data_type(self, data: DataRecord) -> HRMDataType:
        """Infer HRM data type from record"""
        # Check metadata for explicit type
        if 'data_type' in data.metadata:
            try:
                return HRMDataType(data.metadata['data_type'])
            except ValueError:
                pass
        
        # Infer from content
        completion_lower = data.completion.lower()
        
        # Check for code
        code_indicators = ['def ', 'class ', 'import ', 'function', 'return', '{', '}']
        if any(ind in completion_lower for ind in code_indicators):
            return HRMDataType.CODE
        
        # Check for Q&A
        if data.prompt.endswith('?') or 'question' in data.metadata.get('domain', ''):
            return HRMDataType.QA
        
        # Default to reasoning
        return HRMDataType.REASONING
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get transformation statistics"""
        return {
            "transform_count": self.transform_count,
            "error_count": self.error_count,
            "success_rate": self.transform_count / (self.transform_count + self.error_count)
                if (self.transform_count + self.error_count) > 0 else 0.0,
            "num_puzzle_types": len(self.puzzle_id_map),
            "puzzle_id_map": self.puzzle_id_map.copy()
        }

