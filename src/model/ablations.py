"""Ablation study runner for LegalGPT experiments."""

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict, Optional, Any, Callable
from datetime import datetime
import random

import sys
sys.path.append(str(__file__).rsplit("/", 2)[0])
from config import RESULTS_DIR, SPLITS_DIR


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""
    name: str
    description: str
    
    # Retrieval settings
    use_retrieval: bool = True
    retrieval_type: str = "graph"  # "graph", "random", "none"
    num_similar_cases: int = 5
    
    # Model settings
    use_fine_tuning: bool = True
    adapter_path: Optional[str] = None
    
    # Context settings
    max_context_length: int = 4096
    max_case_length: int = 8000
    max_similar_length: int = 1500
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "use_retrieval": self.use_retrieval,
            "retrieval_type": self.retrieval_type,
            "num_similar_cases": self.num_similar_cases,
            "use_fine_tuning": self.use_fine_tuning,
            "adapter_path": self.adapter_path,
            "max_context_length": self.max_context_length,
            "max_case_length": self.max_case_length,
            "max_similar_length": self.max_similar_length,
        }


@dataclass
class AblationResult:
    """Result of a single ablation experiment."""
    config: AblationConfig
    metrics: Dict[str, float]
    predictions_path: Optional[str] = None
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "metrics": self.metrics,
            "predictions_path": self.predictions_path,
            "duration_seconds": self.duration_seconds,
            "timestamp": self.timestamp,
        }


# Predefined ablation configurations
STANDARD_ABLATIONS = [
    AblationConfig(
        name="full_model",
        description="Full model with graph retrieval (k=5)",
        use_retrieval=True,
        retrieval_type="graph",
        num_similar_cases=5,
    ),
    AblationConfig(
        name="no_retrieval",
        description="No retrieval - case text only",
        use_retrieval=False,
        retrieval_type="none",
        num_similar_cases=0,
    ),
    AblationConfig(
        name="random_retrieval",
        description="Random case retrieval instead of graph-based",
        use_retrieval=True,
        retrieval_type="random",
        num_similar_cases=5,
    ),
    AblationConfig(
        name="k1",
        description="Graph retrieval with k=1",
        use_retrieval=True,
        retrieval_type="graph",
        num_similar_cases=1,
    ),
    AblationConfig(
        name="k10",
        description="Graph retrieval with k=10",
        use_retrieval=True,
        retrieval_type="graph",
        num_similar_cases=10,
    ),
    AblationConfig(
        name="k20",
        description="Graph retrieval with k=20",
        use_retrieval=True,
        retrieval_type="graph",
        num_similar_cases=20,
    ),
    AblationConfig(
        name="short_context",
        description="Shorter context window (2048 tokens)",
        use_retrieval=True,
        retrieval_type="graph",
        num_similar_cases=5,
        max_context_length=2048,
    ),
    AblationConfig(
        name="no_finetuning",
        description="Base model without fine-tuning",
        use_retrieval=True,
        retrieval_type="graph",
        num_similar_cases=5,
        use_fine_tuning=False,
    ),
]


class AblationStudyRunner:
    """
    Runner for ablation studies.
    
    Executes multiple experimental configurations and compares results.
    """
    
    def __init__(
        self,
        test_data: List[Dict[str, Any]],
        all_data: Optional[List[Dict[str, Any]]] = None,
        graph_retriever: Optional[Callable] = None,
        base_adapter_path: Optional[str] = None,
        output_dir: Optional[Path] = None,
    ):
        """
        Initialize ablation runner.
        
        Args:
            test_data: Test cases for evaluation
            all_data: All cases (for random retrieval baseline)
            graph_retriever: Function(case_id, k) -> List[Dict] for graph retrieval
            base_adapter_path: Path to fine-tuned adapter
            output_dir: Directory for saving results
        """
        self.test_data = test_data
        self.all_data = all_data or test_data
        self.graph_retriever = graph_retriever
        self.base_adapter_path = base_adapter_path
        self.output_dir = Path(output_dir or RESULTS_DIR / "ablations")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.results: List[AblationResult] = []
    
    def run_ablation(self, config: AblationConfig) -> AblationResult:
        """
        Run a single ablation experiment.
        
        Args:
            config: Ablation configuration
        
        Returns:
            AblationResult with metrics
        """
        import time
        from model.inference import LegalGPTPredictor
        from model.evaluate import compute_metrics
        from model.dataset import AblationDataset, CollateFunction
        from model.model import get_model_and_tokenizer
        
        print(f"\n{'='*60}")
        print(f"Running ablation: {config.name}")
        print(f"Description: {config.description}")
        print(f"{'='*60}")
        
        start_time = time.time()
        
        # Determine adapter path
        adapter_path = config.adapter_path or self.base_adapter_path
        
        # Setup retriever based on config
        retriever = None
        if config.use_retrieval:
            if config.retrieval_type == "graph" and self.graph_retriever:
                retriever = self.graph_retriever
            elif config.retrieval_type == "random":
                retriever = self._create_random_retriever(config.num_similar_cases)
        
        # Prepare test data with appropriate retrieval
        prepared_data = self._prepare_data(config, retriever)
        
        # Load model
        if config.use_fine_tuning and adapter_path:
            predictor = LegalGPTPredictor(
                adapter_path=adapter_path,
                num_similar_cases=config.num_similar_cases,
            )
        else:
            # Load base model without fine-tuning
            model, tokenizer = get_model_and_tokenizer(
                prepare_for_training=False,
            )
            predictor = LegalGPTPredictor(
                model=model,
                num_similar_cases=config.num_similar_cases,
            )
        
        # Run predictions
        results = predictor.predict_batch(prepared_data, show_progress=True)
        
        # Save predictions
        predictions_path = self.output_dir / f"{config.name}_predictions.json"
        predictor.save_predictions(results, predictions_path)
        
        # Compute metrics
        predictions = [r.prediction for r in results]
        ground_truths = [r.ground_truth for r in results]
        confidences = [r.confidence for r in results]
        
        metrics = compute_metrics(predictions, ground_truths, confidences)
        
        duration = time.time() - start_time
        
        # Create result
        result = AblationResult(
            config=config,
            metrics=metrics.to_dict(),
            predictions_path=str(predictions_path),
            duration_seconds=duration,
        )
        
        self.results.append(result)
        
        print(f"\nResults for {config.name}:")
        print(f"  Accuracy: {metrics.accuracy:.4f}")
        print(f"  F1 Score: {metrics.f1:.4f}")
        print(f"  AUROC:    {metrics.auroc:.4f}")
        print(f"  Duration: {duration:.1f}s")
        
        return result
    
    def _prepare_data(
        self,
        config: AblationConfig,
        retriever: Optional[Callable],
    ) -> List[Dict[str, Any]]:
        """Prepare test data with retrieval based on config."""
        prepared = []
        
        for case in self.test_data:
            item = {
                "id": case.get("id", ""),
                "text": case.get("text", "")[:config.max_case_length],
                "outcome": case.get("outcome"),
            }
            
            # Get similar cases
            if config.use_retrieval and retriever:
                similar = retriever(case.get("id", ""), config.num_similar_cases)
                # Truncate similar case text
                for s in similar:
                    s["text"] = s.get("text", "")[:config.max_similar_length]
                item["similar_cases"] = similar
            elif config.use_retrieval and "similar_cases" in case:
                # Use pre-computed similar cases
                similar = case["similar_cases"][:config.num_similar_cases]
                for s in similar:
                    s["text"] = s.get("text", "")[:config.max_similar_length]
                item["similar_cases"] = similar
            else:
                item["similar_cases"] = []
            
            prepared.append(item)
        
        return prepared
    
    def _create_random_retriever(self, k: int) -> Callable:
        """Create a random retriever for baseline comparison."""
        all_cases = self.all_data
        
        def random_retriever(case_id: str, num_cases: int) -> List[Dict]:
            # Exclude current case
            candidates = [c for c in all_cases if c.get("id") != case_id]
            selected = random.sample(candidates, min(num_cases, len(candidates)))
            return selected
        
        return random_retriever
    
    def run_all(
        self,
        configs: Optional[List[AblationConfig]] = None,
    ) -> List[AblationResult]:
        """
        Run all ablation experiments.
        
        Args:
            configs: List of configurations (uses STANDARD_ABLATIONS if None)
        
        Returns:
            List of AblationResults
        """
        configs = configs or STANDARD_ABLATIONS
        
        print(f"\nRunning {len(configs)} ablation experiments...")
        
        for config in configs:
            try:
                self.run_ablation(config)
            except Exception as e:
                print(f"Error in ablation {config.name}: {e}")
                continue
        
        # Save summary
        self._save_summary()
        
        return self.results
    
    def _save_summary(self):
        """Save ablation study summary."""
        summary_path = self.output_dir / "ablation_summary.json"
        
        summary = {
            "total_experiments": len(self.results),
            "timestamp": datetime.now().isoformat(),
            "results": [r.to_dict() for r in self.results],
        }
        
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        
        print(f"\nSaved ablation summary to {summary_path}")
        
        # Generate comparison table
        self._print_comparison_table()
    
    def _print_comparison_table(self):
        """Print comparison table of all results."""
        if not self.results:
            return
        
        print("\n" + "=" * 80)
        print("ABLATION STUDY RESULTS")
        print("=" * 80)
        
        headers = ["Experiment", "Accuracy", "F1", "AUROC", "ECE"]
        
        # Find column widths
        rows = []
        for r in self.results:
            m = r.metrics
            rows.append([
                r.config.name[:20],
                f"{m.get('accuracy', 0):.4f}",
                f"{m.get('f1', 0):.4f}",
                f"{m.get('auroc', 0):.4f}",
                f"{m.get('ece', 0):.4f}",
            ])
        
        col_widths = [max(len(str(row[i])) for row in [headers] + rows) for i in range(len(headers))]
        
        # Print header
        header_line = " | ".join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(header_line)
        print("-" * len(header_line))
        
        # Print rows (sorted by F1 descending)
        sorted_rows = sorted(rows, key=lambda x: float(x[2]), reverse=True)
        for row in sorted_rows:
            print(" | ".join(c.ljust(w) for c, w in zip(row, col_widths)))
        
        print("=" * 80)
    
    def generate_latex_table(self) -> str:
        """Generate LaTeX table for paper."""
        from model.evaluate import generate_latex_table, EvaluationMetrics
        
        # Convert results to EvaluationMetrics format
        metrics_dict = {}
        for r in self.results:
            # Create a mock EvaluationMetrics from dict
            m = r.metrics
            metrics = EvaluationMetrics(
                accuracy=m.get("accuracy", 0),
                precision=m.get("precision", 0),
                recall=m.get("recall", 0),
                f1=m.get("f1", 0),
                auroc=m.get("auroc", 0),
                ece=m.get("ece", 0),
                mce=m.get("mce", 0),
                precision_per_class=m.get("precision_per_class", {}),
                recall_per_class=m.get("recall_per_class", {}),
                f1_per_class=m.get("f1_per_class", {}),
                total_samples=m.get("total_samples", 0),
                correct_predictions=m.get("correct_predictions", 0),
                confusion_matrix=m.get("confusion_matrix", {}),
            )
            metrics_dict[r.config.name] = metrics
        
        return generate_latex_table(
            metrics_dict,
            caption="Ablation study results comparing different retrieval and model configurations.",
            label="tab:ablations",
        )


def run_ablation_study(
    test_data: List[Dict[str, Any]],
    all_data: Optional[List[Dict[str, Any]]] = None,
    graph_retriever: Optional[Callable] = None,
    adapter_path: Optional[str] = None,
    output_dir: Optional[str] = None,
    configs: Optional[List[AblationConfig]] = None,
) -> List[AblationResult]:
    """
    Convenience function to run ablation study.
    
    Args:
        test_data: Test cases
        all_data: All cases for random baseline
        graph_retriever: Graph-based retriever function
        adapter_path: Path to fine-tuned model
        output_dir: Output directory
        configs: Ablation configurations
    
    Returns:
        List of AblationResults
    """
    runner = AblationStudyRunner(
        test_data=test_data,
        all_data=all_data,
        graph_retriever=graph_retriever,
        base_adapter_path=adapter_path,
        output_dir=output_dir,
    )
    
    return runner.run_all(configs)


# Quick ablation for k-value sensitivity
def run_k_sensitivity_study(
    test_data: List[Dict[str, Any]],
    graph_retriever: Callable,
    adapter_path: str,
    k_values: List[int] = [1, 2, 5, 10, 15, 20],
    output_dir: Optional[str] = None,
) -> List[AblationResult]:
    """Run sensitivity study for retrieval k parameter."""
    configs = [
        AblationConfig(
            name=f"k{k}",
            description=f"Graph retrieval with k={k}",
            use_retrieval=True,
            retrieval_type="graph",
            num_similar_cases=k,
        )
        for k in k_values
    ]
    
    return run_ablation_study(
        test_data=test_data,
        graph_retriever=graph_retriever,
        adapter_path=adapter_path,
        output_dir=output_dir or str(RESULTS_DIR / "k_sensitivity"),
        configs=configs,
    )
