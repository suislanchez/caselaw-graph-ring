"""Evaluation metrics for LegalGPT: AUROC, F1, Precision, Recall, ECE."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Tuple
import numpy as np

import sys
sys.path.append(str(__file__).rsplit("/", 2)[0])
from config import RESULTS_DIR
from model.model import OUTCOME_TO_LABEL, LABEL_TO_OUTCOME


@dataclass
class EvaluationMetrics:
    """Container for evaluation metrics."""
    # Classification metrics
    accuracy: float
    precision: float
    recall: float
    f1: float
    auroc: float
    
    # Calibration
    ece: float  # Expected Calibration Error
    mce: float  # Maximum Calibration Error
    
    # Per-class metrics
    precision_per_class: Dict[str, float]
    recall_per_class: Dict[str, float]
    f1_per_class: Dict[str, float]
    
    # Counts
    total_samples: int
    correct_predictions: int
    confusion_matrix: Dict[str, Dict[str, int]]
    
    # Additional
    unknown_predictions: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1": self.f1,
            "auroc": self.auroc,
            "ece": self.ece,
            "mce": self.mce,
            "precision_per_class": self.precision_per_class,
            "recall_per_class": self.recall_per_class,
            "f1_per_class": self.f1_per_class,
            "total_samples": self.total_samples,
            "correct_predictions": self.correct_predictions,
            "confusion_matrix": self.confusion_matrix,
            "unknown_predictions": self.unknown_predictions,
        }
    
    def __str__(self) -> str:
        lines = [
            "=" * 50,
            "Evaluation Results",
            "=" * 50,
            f"Accuracy:  {self.accuracy:.4f}",
            f"Precision: {self.precision:.4f}",
            f"Recall:    {self.recall:.4f}",
            f"F1 Score:  {self.f1:.4f}",
            f"AUROC:     {self.auroc:.4f}",
            "-" * 50,
            "Calibration:",
            f"  ECE: {self.ece:.4f}",
            f"  MCE: {self.mce:.4f}",
            "-" * 50,
            "Per-class Metrics:",
        ]
        
        for cls in ["petitioner", "respondent"]:
            lines.append(f"  {cls}:")
            lines.append(f"    Precision: {self.precision_per_class.get(cls, 0):.4f}")
            lines.append(f"    Recall:    {self.recall_per_class.get(cls, 0):.4f}")
            lines.append(f"    F1:        {self.f1_per_class.get(cls, 0):.4f}")
        
        lines.extend([
            "-" * 50,
            f"Total samples: {self.total_samples}",
            f"Correct: {self.correct_predictions}",
            f"Unknown predictions: {self.unknown_predictions}",
            "=" * 50,
        ])
        
        return "\n".join(lines)


def compute_metrics(
    predictions: List[str],
    ground_truths: List[str],
    confidences: Optional[List[float]] = None,
    num_bins: int = 10,
) -> EvaluationMetrics:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions: List of predicted outcomes ("petitioner" or "respondent")
        ground_truths: List of true outcomes
        confidences: Optional list of prediction confidences for calibration
        num_bins: Number of bins for ECE calculation
    
    Returns:
        EvaluationMetrics object with all computed metrics
    """
    if len(predictions) != len(ground_truths):
        raise ValueError("Predictions and ground_truths must have same length")
    
    # Filter out unknown predictions for main metrics
    valid_mask = [p in OUTCOME_TO_LABEL and g in OUTCOME_TO_LABEL for p, g in zip(predictions, ground_truths)]
    valid_preds = [p for p, v in zip(predictions, valid_mask) if v]
    valid_truths = [g for g, v in zip(ground_truths, valid_mask) if v]
    valid_confs = None
    if confidences:
        valid_confs = [c for c, v in zip(confidences, valid_mask) if v]
    
    unknown_count = len(predictions) - len(valid_preds)
    
    if not valid_preds:
        # Return zero metrics if no valid predictions
        return EvaluationMetrics(
            accuracy=0.0,
            precision=0.0,
            recall=0.0,
            f1=0.0,
            auroc=0.5,
            ece=1.0,
            mce=1.0,
            precision_per_class={},
            recall_per_class={},
            f1_per_class={},
            total_samples=len(predictions),
            correct_predictions=0,
            confusion_matrix={},
            unknown_predictions=unknown_count,
        )
    
    # Convert to binary labels
    y_pred = np.array([OUTCOME_TO_LABEL[p] for p in valid_preds])
    y_true = np.array([OUTCOME_TO_LABEL[g] for g in valid_truths])
    
    # Basic metrics
    accuracy = np.mean(y_pred == y_true)
    
    # Confusion matrix
    cm = _compute_confusion_matrix(y_pred, y_true, num_classes=2)
    
    # Convert confusion matrix to labeled dict
    confusion_dict = {
        "petitioner": {
            "petitioner": int(cm[0, 0]),
            "respondent": int(cm[0, 1]),
        },
        "respondent": {
            "petitioner": int(cm[1, 0]),
            "respondent": int(cm[1, 1]),
        },
    }
    
    # Per-class metrics
    precision_per_class = {}
    recall_per_class = {}
    f1_per_class = {}
    
    for label, name in LABEL_TO_OUTCOME.items():
        tp = cm[label, label]
        fp = cm[:, label].sum() - tp
        fn = cm[label, :].sum() - tp
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
        
        precision_per_class[name] = precision
        recall_per_class[name] = recall
        f1_per_class[name] = f1
    
    # Macro-averaged metrics
    macro_precision = np.mean(list(precision_per_class.values()))
    macro_recall = np.mean(list(recall_per_class.values()))
    macro_f1 = np.mean(list(f1_per_class.values()))
    
    # AUROC (using confidences if available)
    if valid_confs is not None:
        auroc = _compute_auroc(y_true, np.array(valid_confs), y_pred)
    else:
        auroc = _compute_auroc_from_predictions(y_true, y_pred)
    
    # Calibration metrics
    if valid_confs is not None:
        ece, mce = _compute_calibration(y_pred, y_true, np.array(valid_confs), num_bins)
    else:
        ece, mce = 0.0, 0.0  # Cannot compute without confidences
    
    return EvaluationMetrics(
        accuracy=float(accuracy),
        precision=float(macro_precision),
        recall=float(macro_recall),
        f1=float(macro_f1),
        auroc=float(auroc),
        ece=float(ece),
        mce=float(mce),
        precision_per_class=precision_per_class,
        recall_per_class=recall_per_class,
        f1_per_class=f1_per_class,
        total_samples=len(predictions),
        correct_predictions=int(np.sum(y_pred == y_true)),
        confusion_matrix=confusion_dict,
        unknown_predictions=unknown_count,
    )


def _compute_confusion_matrix(y_pred: np.ndarray, y_true: np.ndarray, num_classes: int = 2) -> np.ndarray:
    """Compute confusion matrix."""
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    
    for pred, true in zip(y_pred, y_true):
        cm[true, pred] += 1
    
    return cm


def _compute_auroc(y_true: np.ndarray, confidences: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Compute AUROC using prediction confidences.
    
    For binary classification, use confidence as score for positive class.
    """
    try:
        from sklearn.metrics import roc_auc_score
        
        # Convert confidence to probability of respondent (class 1)
        # If prediction is respondent, use confidence; else use 1 - confidence
        probs = np.where(y_pred == 1, confidences, 1 - confidences)
        
        return roc_auc_score(y_true, probs)
    except ImportError:
        return _compute_auroc_from_predictions(y_true, y_pred)
    except ValueError:
        # Handle edge cases (e.g., only one class in y_true)
        return 0.5


def _compute_auroc_from_predictions(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Compute AUROC approximation from hard predictions (Wilcoxon-Mann-Whitney)."""
    # Count pairs
    n_pos = np.sum(y_true == 1)
    n_neg = np.sum(y_true == 0)
    
    if n_pos == 0 or n_neg == 0:
        return 0.5
    
    # Count correctly ordered pairs
    correct = 0
    ties = 0
    
    pos_preds = y_pred[y_true == 1]
    neg_preds = y_pred[y_true == 0]
    
    for pp in pos_preds:
        for np_ in neg_preds:
            if pp > np_:
                correct += 1
            elif pp == np_:
                ties += 0.5
    
    auroc = (correct + ties) / (n_pos * n_neg)
    return auroc


def _compute_calibration(
    y_pred: np.ndarray,
    y_true: np.ndarray,
    confidences: np.ndarray,
    num_bins: int = 10,
) -> Tuple[float, float]:
    """
    Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE).
    
    ECE: Weighted average of |accuracy - confidence| across bins
    MCE: Maximum |accuracy - confidence| across bins
    """
    bin_boundaries = np.linspace(0, 1, num_bins + 1)
    
    ece = 0.0
    mce = 0.0
    
    correct = (y_pred == y_true).astype(np.float32)
    
    for i in range(num_bins):
        bin_lower = bin_boundaries[i]
        bin_upper = bin_boundaries[i + 1]
        
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        bin_size = np.sum(in_bin)
        
        if bin_size > 0:
            bin_accuracy = np.mean(correct[in_bin])
            bin_confidence = np.mean(confidences[in_bin])
            
            bin_error = np.abs(bin_accuracy - bin_confidence)
            
            ece += (bin_size / len(confidences)) * bin_error
            mce = max(mce, bin_error)
    
    return ece, mce


def evaluate_model(
    model,
    dataloader,
    device: str = "cuda",
) -> EvaluationMetrics:
    """
    Evaluate model on a dataloader.
    
    Args:
        model: LegalGPT model
        dataloader: DataLoader with test data
        device: Device to run evaluation on
    
    Returns:
        EvaluationMetrics
    """
    from model.inference import LegalGPTPredictor
    
    predictor = LegalGPTPredictor(model=model, device=device)
    results = predictor.predict_from_dataloader(dataloader)
    
    predictions = [r.prediction for r in results]
    ground_truths = [r.ground_truth for r in results]
    confidences = [r.confidence for r in results]
    
    return compute_metrics(predictions, ground_truths, confidences)


def evaluate_from_predictions(
    predictions_path: Union[str, Path],
) -> EvaluationMetrics:
    """
    Evaluate from saved predictions file.
    
    Args:
        predictions_path: Path to predictions JSON file
    
    Returns:
        EvaluationMetrics
    """
    from model.inference import LegalGPTPredictor
    
    results = LegalGPTPredictor.load_predictions(predictions_path)
    
    predictions = [r.prediction for r in results]
    ground_truths = [r.ground_truth for r in results]
    confidences = [r.confidence for r in results]
    
    return compute_metrics(predictions, ground_truths, confidences)


def save_metrics(
    metrics: EvaluationMetrics,
    output_path: Union[str, Path],
    include_summary: bool = True,
):
    """Save metrics to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = metrics.to_dict()
    
    if include_summary:
        data["summary"] = str(metrics)
    
    with open(output_path, "w") as f:
        json.dump(data, f, indent=2)
    
    print(f"Saved metrics to {output_path}")


def compare_models(
    results: Dict[str, EvaluationMetrics],
    output_path: Optional[Union[str, Path]] = None,
) -> str:
    """
    Compare metrics across multiple models/experiments.
    
    Args:
        results: Dict mapping model name to EvaluationMetrics
        output_path: Optional path to save comparison
    
    Returns:
        Formatted comparison table
    """
    headers = ["Model", "Accuracy", "Precision", "Recall", "F1", "AUROC", "ECE"]
    
    rows = []
    for name, metrics in results.items():
        rows.append([
            name,
            f"{metrics.accuracy:.4f}",
            f"{metrics.precision:.4f}",
            f"{metrics.recall:.4f}",
            f"{metrics.f1:.4f}",
            f"{metrics.auroc:.4f}",
            f"{metrics.ece:.4f}",
        ])
    
    # Format as table
    col_widths = [max(len(row[i]) for row in [headers] + rows) for i in range(len(headers))]
    
    lines = []
    
    # Header
    header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
    lines.append(header_line)
    lines.append("-" * len(header_line))
    
    # Rows
    for row in rows:
        lines.append(" | ".join(c.ljust(w) for c, w in zip(row, col_widths)))
    
    table = "\n".join(lines)
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            f.write(table)
            f.write("\n\n")
            f.write("Full metrics:\n")
            f.write(json.dumps({k: v.to_dict() for k, v in results.items()}, indent=2))
        
        print(f"Saved comparison to {output_path}")
    
    return table


# Latex table generation for papers
def generate_latex_table(
    results: Dict[str, EvaluationMetrics],
    caption: str = "Model comparison",
    label: str = "tab:results",
) -> str:
    """Generate LaTeX table for paper."""
    lines = [
        r"\begin{table}[h]",
        r"\centering",
        r"\begin{tabular}{lcccccc}",
        r"\toprule",
        r"Model & Accuracy & Precision & Recall & F1 & AUROC & ECE \\",
        r"\midrule",
    ]
    
    for name, metrics in results.items():
        row = f"{name} & {metrics.accuracy:.3f} & {metrics.precision:.3f} & {metrics.recall:.3f} & {metrics.f1:.3f} & {metrics.auroc:.3f} & {metrics.ece:.3f} \\\\"
        lines.append(row)
    
    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        r"\end{table}",
    ])
    
    return "\n".join(lines)
