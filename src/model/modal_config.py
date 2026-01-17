"""Modal configuration for LegalGPT training and inference."""

import modal
from pathlib import Path

# Modal app definition
app = modal.App("legalgpt")

# GPU image with all dependencies
gpu_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        # Core ML
        "torch==2.1.2",
        "transformers==4.36.2",
        "accelerate==0.25.0",
        "bitsandbytes==0.41.3",
        # QLoRA / PEFT
        "peft==0.7.1",
        # Data processing
        "datasets==2.16.1",
        "pandas==2.1.4",
        "pyarrow==14.0.2",
        # Evaluation
        "scikit-learn==1.3.2",
        "scipy==1.11.4",
        # Logging
        "wandb==0.16.1",
        "tqdm==4.66.1",
        # Utils
        "einops==0.7.0",
        "sentencepiece==0.1.99",
    )
    .env({
        "HF_HOME": "/cache/huggingface",
        "TRANSFORMERS_CACHE": "/cache/huggingface",
    })
)

# Volumes for caching models and data
model_cache = modal.Volume.from_name("legalgpt-model-cache", create_if_missing=True)
data_volume = modal.Volume.from_name("legalgpt-data", create_if_missing=True)
results_volume = modal.Volume.from_name("legalgpt-results", create_if_missing=True)

CACHE_DIR = "/cache"
DATA_DIR = "/data"
RESULTS_DIR = "/results"


@app.cls(
    gpu="A100",
    image=gpu_image,
    timeout=3600 * 4,  # 4 hours max
    volumes={
        CACHE_DIR: model_cache,
        DATA_DIR: data_volume,
        RESULTS_DIR: results_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class LegalGPTTrainer:
    """Modal class for training LegalGPT model with QLoRA."""
    
    @modal.enter()
    def setup(self):
        """Initialize model and tokenizer on container start."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import prepare_model_for_kbit_training, LoraConfig, get_peft_model
        import os
        
        self.model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # QLoRA quantization config
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        print(f"Loading model: {self.model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            cache_dir=f"{CACHE_DIR}/huggingface",
            trust_remote_code=True,
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.padding_side = "right"
        
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=f"{CACHE_DIR}/huggingface",
            trust_remote_code=True,
        )
        
        # Prepare for QLoRA training
        self.model = prepare_model_for_kbit_training(self.model)
        
        # LoRA configuration
        lora_config = LoraConfig(
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
        print("Model loaded and prepared for QLoRA training")
    
    @modal.method()
    def train(
        self,
        train_data_path: str,
        val_data_path: str,
        output_dir: str = f"{RESULTS_DIR}/legalgpt-qlora",
        num_epochs: int = 3,
        batch_size: int = 4,
        learning_rate: float = 2e-4,
        gradient_accumulation_steps: int = 4,
        max_length: int = 4096,
        wandb_project: str = "legalgpt",
    ) -> dict:
        """Train the model with QLoRA."""
        import json
        import torch
        from torch.utils.data import DataLoader
        from transformers import get_linear_schedule_with_warmup
        from tqdm import tqdm
        import wandb
        import os
        
        # Load data
        with open(train_data_path) as f:
            train_data = json.load(f)
        with open(val_data_path) as f:
            val_data = json.load(f)
        
        print(f"Training samples: {len(train_data)}")
        print(f"Validation samples: {len(val_data)}")
        
        # Create datasets
        train_dataset = self._create_dataset(train_data, max_length)
        val_dataset = self._create_dataset(val_data, max_length)
        
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size * 2,
            shuffle=False,
            collate_fn=self._collate_fn,
        )
        
        # Optimizer and scheduler
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=0.01,
        )
        
        total_steps = len(train_loader) * num_epochs // gradient_accumulation_steps
        warmup_steps = int(total_steps * 0.1)
        
        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
        
        # Initialize wandb
        if os.environ.get("WANDB_API_KEY"):
            wandb.init(project=wandb_project, name="legalgpt-qlora")
        
        # Training loop
        self.model.train()
        best_val_loss = float("inf")
        training_stats = []
        
        for epoch in range(num_epochs):
            epoch_loss = 0
            self.model.train()
            
            progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
            
            for step, batch in enumerate(progress_bar):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                loss = outputs.loss / gradient_accumulation_steps
                loss.backward()
                
                epoch_loss += loss.item() * gradient_accumulation_steps
                
                if (step + 1) % gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                    optimizer.step()
                    scheduler.step()
                    optimizer.zero_grad()
                
                progress_bar.set_postfix({"loss": loss.item() * gradient_accumulation_steps})
            
            avg_train_loss = epoch_loss / len(train_loader)
            
            # Validation
            val_loss = self._evaluate(val_loader)
            
            stats = {
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": val_loss,
                "learning_rate": scheduler.get_last_lr()[0],
            }
            training_stats.append(stats)
            
            print(f"Epoch {epoch + 1}: Train Loss = {avg_train_loss:.4f}, Val Loss = {val_loss:.4f}")
            
            if os.environ.get("WANDB_API_KEY"):
                wandb.log(stats)
            
            # Save best model
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                self.model.save_pretrained(output_dir)
                self.tokenizer.save_pretrained(output_dir)
                print(f"Saved best model to {output_dir}")
        
        if os.environ.get("WANDB_API_KEY"):
            wandb.finish()
        
        # Commit volumes
        model_cache.commit()
        results_volume.commit()
        
        return {
            "best_val_loss": best_val_loss,
            "training_stats": training_stats,
            "output_dir": output_dir,
        }
    
    def _create_dataset(self, data: list, max_length: int):
        """Create tokenized dataset from case data."""
        from torch.utils.data import Dataset
        
        class LegalDataset(Dataset):
            def __init__(inner_self, data, tokenizer, max_length):
                inner_self.data = data
                inner_self.tokenizer = tokenizer
                inner_self.max_length = max_length
            
            def __len__(inner_self):
                return len(inner_self.data)
            
            def __getitem__(inner_self, idx):
                item = inner_self.data[idx]
                prompt = inner_self._format_prompt(item)
                
                encodings = inner_self.tokenizer(
                    prompt,
                    truncation=True,
                    max_length=inner_self.max_length,
                    padding=False,
                    return_tensors=None,
                )
                
                # Labels are same as input_ids for causal LM
                encodings["labels"] = encodings["input_ids"].copy()
                
                return encodings
            
            def _format_prompt(inner_self, item):
                """Format the prediction prompt."""
                case_text = item.get("text", "")[:8000]  # Truncate if too long
                similar_cases = item.get("similar_cases", [])
                outcome = item.get("outcome", "")
                
                # Build context from similar cases
                context_parts = []
                for i, similar in enumerate(similar_cases[:5], 1):
                    similar_text = similar.get("text", "")[:1500]
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
{case_text}

Based on the legal arguments, precedents, and facts presented, predict the outcome. Answer with only "petitioner" or "respondent".

Prediction: [/INST] {outcome}</s>"""
                
                return prompt
        
        return LegalDataset(data, self.tokenizer, max_length)
    
    def _collate_fn(self, batch):
        """Collate function for DataLoader."""
        import torch
        
        max_len = max(len(item["input_ids"]) for item in batch)
        
        input_ids = []
        attention_mask = []
        labels = []
        
        for item in batch:
            padding_len = max_len - len(item["input_ids"])
            
            input_ids.append(item["input_ids"] + [self.tokenizer.pad_token_id] * padding_len)
            attention_mask.append([1] * len(item["input_ids"]) + [0] * padding_len)
            labels.append(item["labels"] + [-100] * padding_len)
        
        return {
            "input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels),
        }
    
    def _evaluate(self, dataloader) -> float:
        """Evaluate model on validation set."""
        import torch
        
        self.model.eval()
        total_loss = 0
        
        with torch.no_grad():
            for batch in dataloader:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                
                total_loss += outputs.loss.item()
        
        return total_loss / len(dataloader)
    
    @modal.method()
    def predict(self, case_text: str, similar_cases: list = None, max_new_tokens: int = 10) -> dict:
        """Generate prediction for a single case."""
        import torch
        
        similar_cases = similar_cases or []
        
        # Build context
        context_parts = []
        for i, similar in enumerate(similar_cases[:5], 1):
            similar_text = similar.get("text", "")[:1500]
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
{case_text[:8000]}

Based on the legal arguments, precedents, and facts presented, predict the outcome. Answer with only "petitioner" or "respondent".

Prediction: [/INST]"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.device)
        
        self.model.eval()
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        prediction = response.strip().lower()
        
        # Normalize prediction
        if "petitioner" in prediction:
            prediction = "petitioner"
        elif "respondent" in prediction:
            prediction = "respondent"
        else:
            prediction = "unknown"
        
        return {
            "prediction": prediction,
            "raw_response": response,
        }


@app.cls(
    gpu="A100",
    image=gpu_image,
    timeout=3600,
    volumes={
        CACHE_DIR: model_cache,
        DATA_DIR: data_volume,
        RESULTS_DIR: results_volume,
    },
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
class LegalGPTInference:
    """Modal class for inference with trained model."""
    
    @modal.enter()
    def setup(self):
        """Load trained model."""
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from peft import PeftModel
        
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        base_model_name = "mistralai/Mistral-7B-Instruct-v0.3"
        adapter_path = f"{RESULTS_DIR}/legalgpt-qlora"
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            base_model_name,
            cache_dir=f"{CACHE_DIR}/huggingface",
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base_model = AutoModelForCausalLM.from_pretrained(
            base_model_name,
            quantization_config=bnb_config,
            device_map="auto",
            cache_dir=f"{CACHE_DIR}/huggingface",
        )
        
        # Load LoRA adapter
        self.model = PeftModel.from_pretrained(base_model, adapter_path)
        self.model.eval()
        print("Inference model loaded")
    
    @modal.method()
    def batch_predict(self, cases: list, batch_size: int = 8) -> list:
        """Run batch prediction on multiple cases."""
        import torch
        from tqdm import tqdm
        
        results = []
        
        for i in tqdm(range(0, len(cases), batch_size), desc="Predicting"):
            batch = cases[i:i + batch_size]
            
            for case in batch:
                result = self._predict_single(case)
                results.append(result)
        
        return results
    
    def _predict_single(self, case: dict) -> dict:
        """Predict single case."""
        import torch
        
        case_text = case.get("text", "")[:8000]
        similar_cases = case.get("similar_cases", [])
        
        context_parts = []
        for i, similar in enumerate(similar_cases[:5], 1):
            similar_text = similar.get("text", "")[:1500]
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
{case_text}

Based on the legal arguments, precedents, and facts presented, predict the outcome. Answer with only "petitioner" or "respondent".

Prediction: [/INST]"""
        
        inputs = self.tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096).to(self.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=10,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        
        response = self.tokenizer.decode(outputs[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        prediction = response.strip().lower()
        
        if "petitioner" in prediction:
            prediction = "petitioner"
        elif "respondent" in prediction:
            prediction = "respondent"
        else:
            prediction = "unknown"
        
        return {
            "case_id": case.get("id", ""),
            "prediction": prediction,
            "ground_truth": case.get("outcome", ""),
            "raw_response": response,
        }


# Local entrypoint for testing
@app.local_entrypoint()
def main():
    """Test Modal setup."""
    print("Modal app 'legalgpt' configured successfully!")
    print("Available classes: LegalGPTTrainer, LegalGPTInference")
    print("\nTo train: modal run src/model/modal_config.py::LegalGPTTrainer.train")
    print("To deploy: modal deploy src/model/modal_config.py")
