"""LegalGPT model architecture with QLoRA support."""

from typing import Optional, Tuple, Dict, Any
import torch
import torch.nn as nn
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
    PreTrainedModel,
)
from peft import (
    LoraConfig,
    get_peft_model,
    prepare_model_for_kbit_training,
    PeftModel,
)

import sys
sys.path.append(str(__file__).rsplit("/", 2)[0])
from config import (
    MODEL_NAME,
    MAX_CONTEXT_LENGTH,
    LORA_R,
    LORA_ALPHA,
    LORA_DROPOUT,
    LORA_TARGET_MODULES,
    HF_TOKEN,
)


class LegalGPTModel(nn.Module):
    """
    LegalGPT: Mistral-7B with QLoRA adapters for legal outcome prediction.
    
    Architecture:
    - Base: Mistral-7B-Instruct-v0.3 (32k context, Apache 2.0)
    - Quantization: 4-bit NF4 with double quantization
    - Fine-tuning: LoRA adapters on attention + MLP layers
    - Classification: Extract last token hidden state -> linear head
    """
    
    def __init__(
        self,
        model_name: str = MODEL_NAME,
        load_in_4bit: bool = True,
        use_classification_head: bool = False,
        num_labels: int = 2,
        device_map: str = "auto",
        cache_dir: Optional[str] = None,
    ):
        super().__init__()
        
        self.model_name = model_name
        self.use_classification_head = use_classification_head
        self.num_labels = num_labels
        
        # Quantization config for QLoRA
        self.bnb_config = None
        if load_in_4bit:
            self.bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16,
                bnb_4bit_use_double_quant=True,
            )
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            trust_remote_code=True,
            token=HF_TOKEN,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        # Load base model
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=self.bnb_config,
            device_map=device_map,
            cache_dir=cache_dir,
            trust_remote_code=True,
            token=HF_TOKEN,
            torch_dtype=torch.bfloat16 if not load_in_4bit else None,
        )
        
        # Optional classification head
        if use_classification_head:
            hidden_size = self.base_model.config.hidden_size
            self.classifier = nn.Sequential(
                nn.Dropout(0.1),
                nn.Linear(hidden_size, hidden_size // 2),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(hidden_size // 2, num_labels),
            )
        
        self._is_peft_model = False
    
    def prepare_for_training(self) -> "LegalGPTModel":
        """Prepare model for QLoRA training."""
        if self._is_peft_model:
            return self
        
        # Prepare for k-bit training
        self.base_model = prepare_model_for_kbit_training(self.base_model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=LORA_TARGET_MODULES,
        )
        
        # Apply LoRA
        self.base_model = get_peft_model(self.base_model, lora_config)
        self._is_peft_model = True
        
        print("Model prepared for QLoRA training:")
        self.base_model.print_trainable_parameters()
        
        return self
    
    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_dict: bool = True,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass.
        
        For language modeling: returns causal LM loss
        For classification: returns classification loss and logits
        """
        if self.use_classification_head:
            return self._forward_classification(input_ids, attention_mask, labels)
        else:
            return self._forward_lm(input_ids, attention_mask, labels)
    
    def _forward_lm(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Standard causal LM forward pass."""
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels,
        )
        
        return {
            "loss": outputs.loss,
            "logits": outputs.logits,
        }
    
    def _forward_classification(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor],
        labels: Optional[torch.Tensor],
    ) -> Dict[str, torch.Tensor]:
        """Classification forward pass using last token."""
        # Get hidden states
        outputs = self.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        
        # Extract last token hidden state
        hidden_states = outputs.hidden_states[-1]  # (batch, seq_len, hidden)
        
        # Find last non-padding token
        if attention_mask is not None:
            sequence_lengths = attention_mask.sum(dim=1) - 1
        else:
            sequence_lengths = torch.full(
                (input_ids.shape[0],),
                input_ids.shape[1] - 1,
                device=input_ids.device,
            )
        
        # Gather last token hidden states
        batch_indices = torch.arange(hidden_states.shape[0], device=hidden_states.device)
        last_hidden = hidden_states[batch_indices, sequence_lengths]  # (batch, hidden)
        
        # Classification
        logits = self.classifier(last_hidden)  # (batch, num_labels)
        
        loss = None
        if labels is not None:
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(logits, labels)
        
        return {
            "loss": loss,
            "logits": logits,
            "hidden_states": last_hidden,
        }
    
    def generate(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        max_new_tokens: int = 50,
        do_sample: bool = False,
        temperature: float = 0.7,
        top_p: float = 0.9,
        **kwargs,
    ) -> torch.Tensor:
        """Generate text from the model."""
        return self.base_model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_p=top_p if do_sample else None,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id,
            **kwargs,
        )
    
    def save_pretrained(self, output_dir: str):
        """Save LoRA adapters and tokenizer."""
        self.base_model.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)
        
        # Save classification head if present
        if self.use_classification_head:
            torch.save(
                self.classifier.state_dict(),
                f"{output_dir}/classifier_head.pt",
            )
    
    @classmethod
    def from_pretrained(
        cls,
        adapter_path: str,
        base_model_name: str = MODEL_NAME,
        load_in_4bit: bool = True,
        use_classification_head: bool = False,
        device_map: str = "auto",
        cache_dir: Optional[str] = None,
    ) -> "LegalGPTModel":
        """Load model with trained LoRA adapters."""
        # Create base model
        model = cls(
            model_name=base_model_name,
            load_in_4bit=load_in_4bit,
            use_classification_head=use_classification_head,
            device_map=device_map,
            cache_dir=cache_dir,
        )
        
        # Load LoRA adapters
        model.base_model = PeftModel.from_pretrained(
            model.base_model,
            adapter_path,
        )
        model._is_peft_model = True
        
        # Load classification head if present
        classifier_path = f"{adapter_path}/classifier_head.pt"
        if use_classification_head:
            import os
            if os.path.exists(classifier_path):
                model.classifier.load_state_dict(
                    torch.load(classifier_path, map_location="cpu")
                )
        
        return model
    
    def print_trainable_parameters(self):
        """Print trainable parameter statistics."""
        if self._is_peft_model:
            self.base_model.print_trainable_parameters()
        else:
            total = sum(p.numel() for p in self.parameters())
            trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
            print(f"Total parameters: {total:,}")
            print(f"Trainable parameters: {trainable:,} ({100 * trainable / total:.2f}%)")


def get_model_and_tokenizer(
    model_name: str = MODEL_NAME,
    load_in_4bit: bool = True,
    prepare_for_training: bool = False,
    use_classification_head: bool = False,
    device_map: str = "auto",
    cache_dir: Optional[str] = None,
) -> Tuple[LegalGPTModel, AutoTokenizer]:
    """
    Convenience function to get model and tokenizer.
    
    Args:
        model_name: HuggingFace model name
        load_in_4bit: Whether to use 4-bit quantization
        prepare_for_training: Whether to apply LoRA adapters
        use_classification_head: Whether to add classification head
        device_map: Device mapping strategy
        cache_dir: Cache directory for model files
    
    Returns:
        Tuple of (model, tokenizer)
    """
    model = LegalGPTModel(
        model_name=model_name,
        load_in_4bit=load_in_4bit,
        use_classification_head=use_classification_head,
        device_map=device_map,
        cache_dir=cache_dir,
    )
    
    if prepare_for_training:
        model.prepare_for_training()
    
    return model, model.tokenizer


def create_prompt(
    case_text: str,
    similar_cases: list = None,
    max_case_length: int = 8000,
    max_similar_length: int = 1500,
) -> str:
    """
    Create prediction prompt for the model.
    
    Args:
        case_text: Main case text to predict
        similar_cases: List of similar cases with 'text' and 'outcome' keys
        max_case_length: Maximum characters for main case
        max_similar_length: Maximum characters per similar case
    
    Returns:
        Formatted prompt string
    """
    similar_cases = similar_cases or []
    
    # Build context from similar cases
    context_parts = []
    for i, similar in enumerate(similar_cases[:5], 1):
        similar_text = similar.get("text", "")[:max_similar_length]
        similar_outcome = similar.get("outcome", "unknown")
        context_parts.append(
            f"[Similar Case {i}]\n"
            f"Outcome: {similar_outcome}\n"
            f"Text: {similar_text}\n"
        )
    
    context = "\n".join(context_parts) if context_parts else "No similar cases available."
    
    prompt = f"""<s>[INST] You are a legal outcome predictor. Based on the case text and similar precedent cases, predict whether the petitioner or respondent will win.

## Similar Precedent Cases
{context}

## Current Case
{case_text[:max_case_length]}

Based on the legal arguments, precedents, and facts presented, predict the outcome. Answer with only "petitioner" or "respondent".

Prediction: [/INST]"""
    
    return prompt


# Label mappings
OUTCOME_TO_LABEL = {"petitioner": 0, "respondent": 1}
LABEL_TO_OUTCOME = {0: "petitioner", 1: "respondent"}
