"""PyTorch Dataset for legal outcome prediction training."""

import json
import random
from pathlib import Path
from typing import List, Dict, Optional, Callable, Any, Union
from dataclasses import dataclass

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import PreTrainedTokenizer

import sys
sys.path.append(str(__file__).rsplit("/", 2)[0])
from config import (
    SPLITS_DIR,
    TRAIN_BATCH_SIZE,
    EVAL_BATCH_SIZE,
    MAX_CONTEXT_LENGTH,
)
from model.model import create_prompt, OUTCOME_TO_LABEL


@dataclass
class CaseExample:
    """Single case example for training/evaluation."""
    case_id: str
    text: str
    outcome: str  # "petitioner" or "respondent"
    similar_cases: List[Dict[str, Any]]
    
    @property
    def label(self) -> int:
        return OUTCOME_TO_LABEL.get(self.outcome, -1)


class LegalOutcomeDataset(Dataset):
    """
    Dataset for legal outcome prediction.
    
    Each example consists of:
    - Case text
    - Similar precedent cases (from graph retrieval)
    - Binary outcome label (petitioner/respondent)
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        max_length: int = MAX_CONTEXT_LENGTH,
        max_case_length: int = 8000,
        max_similar_length: int = 1500,
        num_similar_cases: int = 5,
        retriever: Optional[Callable] = None,
        include_labels_in_prompt: bool = True,
    ):
        """
        Initialize dataset.
        
        Args:
            data: List of case dictionaries with 'id', 'text', 'outcome', 'similar_cases'
            tokenizer: HuggingFace tokenizer
            max_length: Maximum sequence length for tokenization
            max_case_length: Maximum characters for main case text
            max_similar_length: Maximum characters per similar case
            num_similar_cases: Number of similar cases to include
            retriever: Optional retriever function to get similar cases dynamically
            include_labels_in_prompt: Whether to include outcome in prompt (for training)
        """
        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.max_case_length = max_case_length
        self.max_similar_length = max_similar_length
        self.num_similar_cases = num_similar_cases
        self.retriever = retriever
        self.include_labels_in_prompt = include_labels_in_prompt
        
        # Filter out cases without valid outcomes
        self.examples = [
            CaseExample(
                case_id=item.get("id", str(i)),
                text=item.get("text", ""),
                outcome=item.get("outcome", ""),
                similar_cases=item.get("similar_cases", []),
            )
            for i, item in enumerate(data)
            if item.get("outcome") in OUTCOME_TO_LABEL
        ]
        
        print(f"Loaded {len(self.examples)} examples from {len(data)} items")
    
    def __len__(self) -> int:
        return len(self.examples)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Get similar cases (from retriever or pre-computed)
        if self.retriever is not None:
            similar_cases = self.retriever(example.case_id, k=self.num_similar_cases)
        else:
            similar_cases = example.similar_cases[:self.num_similar_cases]
        
        # Create prompt
        prompt = create_prompt(
            case_text=example.text,
            similar_cases=similar_cases,
            max_case_length=self.max_case_length,
            max_similar_length=self.max_similar_length,
        )
        
        # Add outcome to prompt for training
        if self.include_labels_in_prompt:
            prompt = prompt + f" {example.outcome}</s>"
        
        # Tokenize
        encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        
        # Labels are same as input_ids for causal LM
        encodings["labels"] = encodings["input_ids"].copy()
        
        # Add metadata
        encodings["case_id"] = example.case_id
        encodings["outcome"] = example.outcome
        encodings["outcome_label"] = example.label
        
        return encodings
    
    def get_raw_example(self, idx: int) -> CaseExample:
        """Get raw example without tokenization."""
        return self.examples[idx]


class CollateFunction:
    """Collate function for DataLoader with dynamic padding."""
    
    def __init__(self, tokenizer: PreTrainedTokenizer, max_length: int = None):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.pad_token_id = tokenizer.pad_token_id
    
    def __call__(self, batch: List[Dict]) -> Dict[str, torch.Tensor]:
        # Separate metadata from tensors
        case_ids = [item.pop("case_id") for item in batch]
        outcomes = [item.pop("outcome") for item in batch]
        outcome_labels = [item.pop("outcome_label") for item in batch]
        
        # Find max length in batch
        max_len = max(len(item["input_ids"]) for item in batch)
        if self.max_length:
            max_len = min(max_len, self.max_length)
        
        input_ids = []
        attention_mask = []
        labels = []
        
        for item in batch:
            seq_len = len(item["input_ids"])
            
            if seq_len > max_len:
                # Truncate
                input_ids.append(item["input_ids"][:max_len])
                attention_mask.append([1] * max_len)
                labels.append(item["labels"][:max_len])
            else:
                # Pad
                padding_len = max_len - seq_len
                input_ids.append(item["input_ids"] + [self.pad_token_id] * padding_len)
                attention_mask.append([1] * seq_len + [0] * padding_len)
                labels.append(item["labels"] + [-100] * padding_len)
        
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "case_ids": case_ids,
            "outcomes": outcomes,
            "outcome_labels": torch.tensor(outcome_labels, dtype=torch.long),
        }


def load_split_data(split: str, data_dir: Path = SPLITS_DIR) -> List[Dict]:
    """
    Load data split from JSON file.
    
    Args:
        split: One of 'train', 'val', 'test'
        data_dir: Directory containing split files
    
    Returns:
        List of case dictionaries
    """
    file_path = data_dir / f"{split}.json"
    
    if not file_path.exists():
        raise FileNotFoundError(f"Split file not found: {file_path}")
    
    with open(file_path) as f:
        data = json.load(f)
    
    print(f"Loaded {len(data)} examples from {split} split")
    return data


def create_dataloaders(
    tokenizer: PreTrainedTokenizer,
    train_data: Optional[List[Dict]] = None,
    val_data: Optional[List[Dict]] = None,
    test_data: Optional[List[Dict]] = None,
    train_batch_size: int = TRAIN_BATCH_SIZE,
    eval_batch_size: int = EVAL_BATCH_SIZE,
    max_length: int = MAX_CONTEXT_LENGTH,
    num_workers: int = 0,
    retriever: Optional[Callable] = None,
) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train/val/test splits.
    
    Args:
        tokenizer: HuggingFace tokenizer
        train_data: Training data (or load from default path)
        val_data: Validation data (or load from default path)
        test_data: Test data (or load from default path)
        train_batch_size: Batch size for training
        eval_batch_size: Batch size for evaluation
        max_length: Maximum sequence length
        num_workers: Number of DataLoader workers
        retriever: Optional retriever for similar cases
    
    Returns:
        Dictionary with 'train', 'val', 'test' DataLoaders (if data provided)
    """
    collate_fn = CollateFunction(tokenizer, max_length)
    dataloaders = {}
    
    if train_data is not None:
        train_dataset = LegalOutcomeDataset(
            train_data,
            tokenizer,
            max_length=max_length,
            retriever=retriever,
            include_labels_in_prompt=True,
        )
        dataloaders["train"] = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    if val_data is not None:
        val_dataset = LegalOutcomeDataset(
            val_data,
            tokenizer,
            max_length=max_length,
            retriever=retriever,
            include_labels_in_prompt=True,
        )
        dataloaders["val"] = DataLoader(
            val_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    if test_data is not None:
        test_dataset = LegalOutcomeDataset(
            test_data,
            tokenizer,
            max_length=max_length,
            retriever=retriever,
            include_labels_in_prompt=False,  # Don't leak labels during testing
        )
        dataloaders["test"] = DataLoader(
            test_dataset,
            batch_size=eval_batch_size,
            shuffle=False,
            collate_fn=collate_fn,
            num_workers=num_workers,
            pin_memory=True,
        )
    
    return dataloaders


class AblationDataset(LegalOutcomeDataset):
    """
    Dataset variant for ablation studies.
    
    Supports:
    - No retrieval (empty similar_cases)
    - Random retrieval (random cases instead of graph-based)
    - Different k values
    """
    
    def __init__(
        self,
        data: List[Dict[str, Any]],
        tokenizer: PreTrainedTokenizer,
        ablation_type: str = "full",  # "full", "no_retrieval", "random_retrieval"
        num_similar_cases: int = 5,
        all_cases: Optional[List[Dict]] = None,  # For random retrieval
        **kwargs,
    ):
        super().__init__(
            data,
            tokenizer,
            num_similar_cases=num_similar_cases,
            **kwargs,
        )
        
        self.ablation_type = ablation_type
        self.all_cases = all_cases or []
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        example = self.examples[idx]
        
        # Get similar cases based on ablation type
        if self.ablation_type == "no_retrieval":
            similar_cases = []
        elif self.ablation_type == "random_retrieval" and self.all_cases:
            # Random cases (excluding current case)
            candidates = [c for c in self.all_cases if c.get("id") != example.case_id]
            similar_cases = random.sample(
                candidates,
                min(self.num_similar_cases, len(candidates))
            )
        else:
            # Full retrieval (default)
            if self.retriever is not None:
                similar_cases = self.retriever(example.case_id, k=self.num_similar_cases)
            else:
                similar_cases = example.similar_cases[:self.num_similar_cases]
        
        # Create prompt
        prompt = create_prompt(
            case_text=example.text,
            similar_cases=similar_cases,
            max_case_length=self.max_case_length,
            max_similar_length=self.max_similar_length,
        )
        
        if self.include_labels_in_prompt:
            prompt = prompt + f" {example.outcome}</s>"
        
        encodings = self.tokenizer(
            prompt,
            truncation=True,
            max_length=self.max_length,
            padding=False,
            return_tensors=None,
        )
        
        encodings["labels"] = encodings["input_ids"].copy()
        encodings["case_id"] = example.case_id
        encodings["outcome"] = example.outcome
        encodings["outcome_label"] = example.label
        
        return encodings


def create_sample_data(num_samples: int = 100) -> List[Dict]:
    """Create sample data for testing (placeholder cases)."""
    outcomes = ["petitioner", "respondent"]
    
    data = []
    for i in range(num_samples):
        outcome = random.choice(outcomes)
        data.append({
            "id": f"case_{i:04d}",
            "text": f"Sample case {i}. This is a legal case involving various issues. "
                    f"The {'petitioner' if outcome == 'petitioner' else 'respondent'} "
                    f"presents strong arguments. Legal precedent suggests... " * 10,
            "outcome": outcome,
            "similar_cases": [
                {
                    "id": f"similar_{i}_{j}",
                    "text": f"Similar case {j} with outcome: {random.choice(outcomes)}",
                    "outcome": random.choice(outcomes),
                }
                for j in range(5)
            ],
        })
    
    return data
