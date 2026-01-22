"""Model & Training Pipeline for LegalGPT."""

from .model import (
    LegalGPTModel,
    get_model_and_tokenizer,
    create_prompt,
    OUTCOME_TO_LABEL,
    LABEL_TO_OUTCOME,
)
from .dataset import (
    LegalOutcomeDataset,
    AblationDataset,
    CollateFunction,
    CaseExample,
    create_dataloaders,
    load_split_data,
    create_sample_data,
)
from .trainer import LegalGPTTrainer, TrainingConfig, train_legalgpt
from .inference import LegalGPTPredictor, PredictionResult
from .evaluate import (
    EvaluationMetrics,
    evaluate_model,
    compute_metrics,
    save_metrics,
    compare_models,
)
from .ablations import (
    AblationStudyRunner,
    AblationResult,
    AblationConfig,
    STANDARD_ABLATIONS,
    run_ablation_study,
)

__all__ = [
    # Model
    "LegalGPTModel",
    "get_model_and_tokenizer",
    "create_prompt",
    "OUTCOME_TO_LABEL",
    "LABEL_TO_OUTCOME",
    # Dataset
    "LegalOutcomeDataset",
    "AblationDataset",
    "CollateFunction",
    "CaseExample",
    "create_dataloaders",
    "load_split_data",
    "create_sample_data",
    # Training
    "LegalGPTTrainer",
    "TrainingConfig",
    "train_legalgpt",
    # Inference
    "LegalGPTPredictor",
    "PredictionResult",
    # Evaluation
    "EvaluationMetrics",
    "evaluate_model",
    "compute_metrics",
    "save_metrics",
    "compare_models",
    # Ablations
    "AblationStudyRunner",
    "AblationResult",
    "AblationConfig",
    "STANDARD_ABLATIONS",
    "run_ablation_study",
]
