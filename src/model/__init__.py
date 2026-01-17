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
from .trainer import LegalGPTTrainerLocal
from .inference import (
    LegalGPTInference,
    LegalGPTPredictor,
    PredictionResult,
)
from .evaluate import (
    Evaluator,
    EvalResults,
    evaluate_model,
    compute_metrics,
)
from .ablations import (
    AblationRunner,
    AblationResult,
    AblationConfig,
    RandomRetriever,
    NoRetriever,
    run_ablation_study,
)
from .modal_config import (
    app,
    gpu_image,
    model_cache,
    data_volume,
    results_volume,
    CACHE_DIR,
    DATA_DIR,
    RESULTS_DIR,
    LegalGPTTrainer,
    LegalGPTInference as ModalLegalGPTInference,
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
    "LegalGPTTrainerLocal",
    # Inference
    "LegalGPTInference",
    "LegalGPTPredictor",
    "PredictionResult",
    # Evaluation
    "Evaluator",
    "EvalResults",
    "evaluate_model",
    "compute_metrics",
    # Ablations
    "AblationRunner",
    "AblationResult",
    "AblationConfig",
    "RandomRetriever",
    "NoRetriever",
    "run_ablation_study",
    # Modal
    "app",
    "gpu_image",
    "model_cache",
    "data_volume",
    "results_volume",
    "CACHE_DIR",
    "DATA_DIR",
    "RESULTS_DIR",
    "LegalGPTTrainer",
    "ModalLegalGPTInference",
]
