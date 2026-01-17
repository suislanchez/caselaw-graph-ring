"""Training loop for LegalGPT with QLoRA."""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from tqdm import tqdm

import sys
sys.path.append(str(__file__).rsplit("/", 2)[0])
from config import (
    MODELS_DIR,
    RESULTS_DIR,
    LEARNING_RATE,
    NUM_EPOCHS,
    WARMUP_RATIO,
    GRADIENT_ACCUMULATION_STEPS,
    TRAIN_BATCH_SIZE,
)


@dataclass
class TrainingConfig:
    """Configuration for training."""
    output_dir: str = str(MODELS_DIR / "legalgpt-qlora")
    num_epochs: int = NUM_EPOCHS
    learning_rate: float = LEARNING_RATE
    warmup_ratio: float = WARMUP_RATIO
    gradient_accumulation_steps: int = GRADIENT_ACCUMULATION_STEPS
    max_grad_norm: float = 1.0
    weight_decay: float = 0.01
    
    # Mixed precision
    use_amp: bool = True
    
    # Logging
    logging_steps: int = 10
    eval_steps: int = 100
    save_steps: int = 500
    
    # Early stopping
    early_stopping_patience: int = 3
    early_stopping_threshold: float = 0.001
    
    # Wandb
    use_wandb: bool = False
    wandb_project: str = "legalgpt"
    wandb_run_name: Optional[str] = None
    
    # Checkpointing
    save_total_limit: int = 3
    load_best_model_at_end: bool = True


@dataclass
class TrainingState:
    """State tracking during training."""
    global_step: int = 0
    epoch: int = 0
    best_val_loss: float = float("inf")
    patience_counter: int = 0
    training_loss: float = 0.0
    
    history: List[Dict[str, float]] = field(default_factory=list)


class LegalGPTTrainer:
    """
    Trainer for LegalGPT with QLoRA fine-tuning.
    
    Features:
    - Gradient accumulation
    - Mixed precision training (AMP)
    - Learning rate scheduling with warmup
    - Early stopping
    - Checkpoint management
    - Wandb logging
    """
    
    def __init__(
        self,
        model: nn.Module,
        train_dataloader: DataLoader,
        val_dataloader: Optional[DataLoader] = None,
        config: Optional[TrainingConfig] = None,
        eval_fn: Optional[Callable] = None,
    ):
        """
        Initialize trainer.
        
        Args:
            model: LegalGPT model (with LoRA applied)
            train_dataloader: Training DataLoader
            val_dataloader: Validation DataLoader
            config: Training configuration
            eval_fn: Optional evaluation function(model, dataloader) -> dict
        """
        self.model = model
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.config = config or TrainingConfig()
        self.eval_fn = eval_fn
        
        self.device = next(model.parameters()).device
        self.state = TrainingState()
        
        # Setup optimizer
        self.optimizer = self._create_optimizer()
        
        # Setup scheduler
        self.scheduler = self._create_scheduler()
        
        # Mixed precision
        self.scaler = GradScaler() if self.config.use_amp else None
        
        # Create output directory
        Path(self.config.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Wandb
        if self.config.use_wandb:
            self._init_wandb()
    
    def _create_optimizer(self) -> AdamW:
        """Create AdamW optimizer with weight decay."""
        # Don't apply weight decay to bias and LayerNorm
        no_decay = ["bias", "LayerNorm.weight", "layer_norm.weight"]
        
        optimizer_grouped_parameters = [
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if p.requires_grad and not any(nd in n for nd in no_decay)
                ],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [
                    p for n, p in self.model.named_parameters()
                    if p.requires_grad and any(nd in n for nd in no_decay)
                ],
                "weight_decay": 0.0,
            },
        ]
        
        return AdamW(
            optimizer_grouped_parameters,
            lr=self.config.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-8,
        )
    
    def _create_scheduler(self):
        """Create learning rate scheduler with warmup."""
        total_steps = (
            len(self.train_dataloader)
            * self.config.num_epochs
            // self.config.gradient_accumulation_steps
        )
        warmup_steps = int(total_steps * self.config.warmup_ratio)
        
        return get_linear_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_steps,
        )
    
    def _init_wandb(self):
        """Initialize Weights & Biases logging."""
        try:
            import wandb
            
            run_name = self.config.wandb_run_name or f"legalgpt-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            
            wandb.init(
                project=self.config.wandb_project,
                name=run_name,
                config={
                    "learning_rate": self.config.learning_rate,
                    "epochs": self.config.num_epochs,
                    "batch_size": TRAIN_BATCH_SIZE,
                    "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                    "warmup_ratio": self.config.warmup_ratio,
                },
            )
            self.wandb = wandb
        except ImportError:
            print("wandb not installed, disabling logging")
            self.config.use_wandb = False
    
    def train(self) -> Dict[str, Any]:
        """
        Run training loop.
        
        Returns:
            Dictionary with training results
        """
        print(f"Starting training for {self.config.num_epochs} epochs")
        print(f"Total training steps: {len(self.train_dataloader) * self.config.num_epochs}")
        print(f"Effective batch size: {TRAIN_BATCH_SIZE * self.config.gradient_accumulation_steps}")
        
        for epoch in range(self.config.num_epochs):
            self.state.epoch = epoch + 1
            
            # Train epoch
            train_metrics = self._train_epoch()
            
            # Evaluate
            val_metrics = {}
            if self.val_dataloader is not None:
                val_metrics = self._evaluate()
            
            # Log epoch results
            epoch_metrics = {
                "epoch": epoch + 1,
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            self.state.history.append(epoch_metrics)
            
            print(f"\nEpoch {epoch + 1}/{self.config.num_epochs}")
            print(f"  Train Loss: {train_metrics.get('loss', 0):.4f}")
            if val_metrics:
                print(f"  Val Loss: {val_metrics.get('loss', 0):.4f}")
            
            if self.config.use_wandb:
                self.wandb.log(epoch_metrics)
            
            # Check for improvement
            val_loss = val_metrics.get("loss", float("inf"))
            if val_loss < self.state.best_val_loss - self.config.early_stopping_threshold:
                self.state.best_val_loss = val_loss
                self.state.patience_counter = 0
                
                # Save best model
                self._save_checkpoint("best")
                print(f"  New best model saved (val_loss: {val_loss:.4f})")
            else:
                self.state.patience_counter += 1
            
            # Early stopping
            if self.state.patience_counter >= self.config.early_stopping_patience:
                print(f"\nEarly stopping triggered after {epoch + 1} epochs")
                break
            
            # Regular checkpoint
            if (epoch + 1) % (self.config.save_steps // len(self.train_dataloader) or 1) == 0:
                self._save_checkpoint(f"epoch-{epoch + 1}")
        
        # Load best model if requested
        if self.config.load_best_model_at_end:
            self._load_checkpoint("best")
        
        # Final save
        self._save_checkpoint("final")
        
        # Save training history
        self._save_history()
        
        if self.config.use_wandb:
            self.wandb.finish()
        
        return {
            "best_val_loss": self.state.best_val_loss,
            "total_epochs": self.state.epoch,
            "total_steps": self.state.global_step,
            "history": self.state.history,
        }
    
    def _train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        
        total_loss = 0.0
        num_batches = 0
        
        progress_bar = tqdm(
            self.train_dataloader,
            desc=f"Epoch {self.state.epoch}",
            leave=True,
        )
        
        self.optimizer.zero_grad()
        
        for step, batch in enumerate(progress_bar):
            # Move batch to device
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)
            
            # Forward pass with mixed precision
            if self.config.use_amp:
                with autocast(dtype=torch.bfloat16):
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                    loss = outputs["loss"] / self.config.gradient_accumulation_steps
                
                self.scaler.scale(loss).backward()
            else:
                outputs = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    labels=labels,
                )
                loss = outputs["loss"] / self.config.gradient_accumulation_steps
                loss.backward()
            
            total_loss += loss.item() * self.config.gradient_accumulation_steps
            num_batches += 1
            
            # Gradient accumulation step
            if (step + 1) % self.config.gradient_accumulation_steps == 0:
                if self.config.use_amp:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config.max_grad_norm,
                    )
                    self.optimizer.step()
                
                self.scheduler.step()
                self.optimizer.zero_grad()
                self.state.global_step += 1
            
            # Update progress bar
            progress_bar.set_postfix({
                "loss": loss.item() * self.config.gradient_accumulation_steps,
                "lr": self.scheduler.get_last_lr()[0],
            })
            
            # Periodic logging
            if self.config.use_wandb and step % self.config.logging_steps == 0:
                self.wandb.log({
                    "train/loss": loss.item() * self.config.gradient_accumulation_steps,
                    "train/learning_rate": self.scheduler.get_last_lr()[0],
                    "train/global_step": self.state.global_step,
                })
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        self.state.training_loss = avg_loss
        
        return {"loss": avg_loss}
    
    def _evaluate(self) -> Dict[str, float]:
        """Evaluate on validation set."""
        self.model.eval()
        
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_dataloader, desc="Evaluating", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)
                
                if self.config.use_amp:
                    with autocast(dtype=torch.bfloat16):
                        outputs = self.model(
                            input_ids=input_ids,
                            attention_mask=attention_mask,
                            labels=labels,
                        )
                else:
                    outputs = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        labels=labels,
                    )
                
                total_loss += outputs["loss"].item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        
        metrics = {"loss": avg_loss}
        
        # Run custom evaluation function if provided
        if self.eval_fn is not None:
            custom_metrics = self.eval_fn(self.model, self.val_dataloader)
            metrics.update(custom_metrics)
        
        return metrics
    
    def _save_checkpoint(self, name: str):
        """Save model checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / name
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model (LoRA adapters)
        self.model.save_pretrained(str(checkpoint_dir))
        
        # Save training state
        state_dict = {
            "global_step": self.state.global_step,
            "epoch": self.state.epoch,
            "best_val_loss": self.state.best_val_loss,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
        }
        
        torch.save(state_dict, checkpoint_dir / "training_state.pt")
        
        # Manage checkpoint count
        self._cleanup_checkpoints()
    
    def _load_checkpoint(self, name: str):
        """Load model checkpoint."""
        checkpoint_dir = Path(self.config.output_dir) / name
        
        if not checkpoint_dir.exists():
            print(f"Checkpoint {name} not found")
            return
        
        # Load training state
        state_path = checkpoint_dir / "training_state.pt"
        if state_path.exists():
            state_dict = torch.load(state_path, map_location="cpu")
            self.state.global_step = state_dict.get("global_step", 0)
            self.state.epoch = state_dict.get("epoch", 0)
            self.state.best_val_loss = state_dict.get("best_val_loss", float("inf"))
        
        print(f"Loaded checkpoint from {checkpoint_dir}")
    
    def _cleanup_checkpoints(self):
        """Remove old checkpoints to maintain save_total_limit."""
        output_dir = Path(self.config.output_dir)
        
        # Get all checkpoint directories (excluding 'best' and 'final')
        checkpoints = [
            d for d in output_dir.iterdir()
            if d.is_dir() and d.name not in ["best", "final"]
        ]
        
        # Sort by modification time
        checkpoints.sort(key=lambda x: x.stat().st_mtime)
        
        # Remove oldest if over limit
        while len(checkpoints) > self.config.save_total_limit:
            oldest = checkpoints.pop(0)
            import shutil
            shutil.rmtree(oldest)
            print(f"Removed old checkpoint: {oldest}")
    
    def _save_history(self):
        """Save training history to JSON."""
        history_path = Path(self.config.output_dir) / "training_history.json"
        
        with open(history_path, "w") as f:
            json.dump({
                "config": {
                    "num_epochs": self.config.num_epochs,
                    "learning_rate": self.config.learning_rate,
                    "warmup_ratio": self.config.warmup_ratio,
                    "gradient_accumulation_steps": self.config.gradient_accumulation_steps,
                },
                "best_val_loss": self.state.best_val_loss,
                "total_epochs": self.state.epoch,
                "total_steps": self.state.global_step,
                "history": self.state.history,
            }, f, indent=2)
        
        print(f"Saved training history to {history_path}")


def train_legalgpt(
    model,
    train_data: List[Dict],
    val_data: Optional[List[Dict]] = None,
    tokenizer=None,
    config: Optional[TrainingConfig] = None,
    **kwargs,
) -> Dict[str, Any]:
    """
    Convenience function to train LegalGPT.
    
    Args:
        model: LegalGPT model
        train_data: Training data list
        val_data: Validation data list
        tokenizer: Tokenizer (uses model's if not provided)
        config: Training configuration
        **kwargs: Additional arguments for create_dataloaders
    
    Returns:
        Training results dictionary
    """
    from model.dataset import create_dataloaders
    
    if tokenizer is None:
        tokenizer = model.tokenizer
    
    # Create dataloaders
    dataloaders = create_dataloaders(
        tokenizer=tokenizer,
        train_data=train_data,
        val_data=val_data,
        **kwargs,
    )
    
    # Create trainer
    trainer = LegalGPTTrainer(
        model=model,
        train_dataloader=dataloaders["train"],
        val_dataloader=dataloaders.get("val"),
        config=config,
    )
    
    # Train
    return trainer.train()
