#!/usr/bin/env python3
"""
Run full evaluation pipeline for LegalGPT.

Usage:
    python scripts/run_evaluation.py --adapter-path models/legalgpt-qlora/best
    python scripts/run_evaluation.py --run-ablations
    python scripts/run_evaluation.py --generate-paper
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import MODELS_DIR, RESULTS_DIR, SPLITS_DIR, DATA_DIR


def load_test_data():
    """Load test data."""
    test_path = SPLITS_DIR / "test.json"
    
    if not test_path.exists():
        print(f"Test data not found at {test_path}")
        print("Creating sample test data...")
        
        # Create sample data for testing
        from src.model.dataset import create_sample_data
        test_data = create_sample_data(num_samples=50)
        
        test_path.parent.mkdir(parents=True, exist_ok=True)
        with open(test_path, "w") as f:
            json.dump(test_data, f, indent=2)
        
        return test_data
    
    with open(test_path) as f:
        return json.load(f)


def get_retriever():
    """Get graph retriever if available."""
    try:
        from src.graph.retriever import get_similar_cases
        return get_similar_cases
    except ImportError:
        print("Graph retriever not available, using None")
        return None


def run_evaluation(
    adapter_path: str,
    output_dir: Path,
    test_data: list,
    retriever=None,
):
    """Run model evaluation."""
    from src.model.inference import LegalGPTPredictor
    from src.model.evaluate import compute_metrics, save_metrics
    
    print(f"\nLoading model from {adapter_path}...")
    predictor = LegalGPTPredictor(
        adapter_path=adapter_path,
        retriever=retriever,
    )
    
    print(f"Running predictions on {len(test_data)} test cases...")
    results = predictor.predict_batch(test_data, show_progress=True)
    
    # Save predictions
    predictions_path = output_dir / "predictions.json"
    predictor.save_predictions(results, predictions_path)
    
    # Compute metrics
    predictions = [r.prediction for r in results]
    ground_truths = [r.ground_truth for r in results]
    confidences = [r.confidence for r in results]
    
    metrics = compute_metrics(predictions, ground_truths, confidences)
    
    # Save metrics
    metrics_path = output_dir / "metrics.json"
    save_metrics(metrics, metrics_path)
    
    # Print results
    print("\n" + str(metrics))
    
    return metrics


def run_ablation_studies(
    adapter_path: str,
    output_dir: Path,
    test_data: list,
    all_data: list = None,
    retriever=None,
):
    """Run ablation studies."""
    from src.model.ablations import AblationStudyRunner, STANDARD_ABLATIONS
    
    print("\nRunning ablation studies...")
    
    runner = AblationStudyRunner(
        test_data=test_data,
        all_data=all_data or test_data,
        graph_retriever=retriever,
        base_adapter_path=adapter_path,
        output_dir=output_dir / "ablations",
    )
    
    results = runner.run_all()
    
    # Generate LaTeX table
    latex = runner.generate_latex_table()
    latex_path = output_dir / "ablations" / "results_table.tex"
    with open(latex_path, "w") as f:
        f.write(latex)
    print(f"\nSaved LaTeX table to {latex_path}")
    
    return results


def generate_paper_results(output_dir: Path):
    """Generate all results needed for the paper."""
    from src.graph.visualize import generate_all_visualizations
    
    print("\nGenerating paper results...")
    
    paper_dir = output_dir / "paper"
    paper_dir.mkdir(parents=True, exist_ok=True)
    
    # Load metrics if available
    metrics_path = output_dir / "metrics.json"
    ablations_path = output_dir / "ablations" / "ablation_summary.json"
    
    paper_data = {
        "generated_at": datetime.now().isoformat(),
        "metrics": None,
        "ablations": None,
        "figures": [],
    }
    
    if metrics_path.exists():
        with open(metrics_path) as f:
            paper_data["metrics"] = json.load(f)
    
    if ablations_path.exists():
        with open(ablations_path) as f:
            paper_data["ablations"] = json.load(f)
    
    # Generate visualizations
    print("Generating figures...")
    try:
        fig_paths = generate_all_visualizations(output_dir=paper_dir / "figures")
        paper_data["figures"] = [str(p) for p in fig_paths.values()]
    except Exception as e:
        print(f"Figure generation failed: {e}")
    
    # Generate results summary
    summary_path = paper_dir / "results_summary.json"
    with open(summary_path, "w") as f:
        json.dump(paper_data, f, indent=2)
    
    # Generate markdown summary
    md_content = generate_markdown_summary(paper_data)
    md_path = paper_dir / "results_summary.md"
    with open(md_path, "w") as f:
        f.write(md_content)
    
    print(f"\nPaper results saved to {paper_dir}")
    return paper_data


def generate_markdown_summary(data: dict) -> str:
    """Generate markdown summary for paper."""
    lines = [
        "# LegalGPT Evaluation Results",
        "",
        f"Generated: {data.get('generated_at', 'N/A')}",
        "",
        "## Main Results",
        "",
    ]
    
    if data.get("metrics"):
        m = data["metrics"]
        lines.extend([
            "| Metric | Value |",
            "|--------|-------|",
            f"| Accuracy | {m.get('accuracy', 'N/A'):.4f} |" if isinstance(m.get('accuracy'), (int, float)) else "| Accuracy | N/A |",
            f"| Precision | {m.get('precision', 'N/A'):.4f} |" if isinstance(m.get('precision'), (int, float)) else "| Precision | N/A |",
            f"| Recall | {m.get('recall', 'N/A'):.4f} |" if isinstance(m.get('recall'), (int, float)) else "| Recall | N/A |",
            f"| F1 Score | {m.get('f1', 'N/A'):.4f} |" if isinstance(m.get('f1'), (int, float)) else "| F1 Score | N/A |",
            f"| AUROC | {m.get('auroc', 'N/A'):.4f} |" if isinstance(m.get('auroc'), (int, float)) else "| AUROC | N/A |",
            f"| ECE | {m.get('ece', 'N/A'):.4f} |" if isinstance(m.get('ece'), (int, float)) else "| ECE | N/A |",
            "",
        ])
    
    lines.append("## Ablation Studies")
    lines.append("")
    
    if data.get("ablations") and data["ablations"].get("results"):
        lines.extend([
            "| Experiment | Accuracy | F1 | AUROC |",
            "|------------|----------|-----|-------|",
        ])
        
        for r in data["ablations"]["results"]:
            config = r.get("config", {})
            metrics = r.get("metrics", {})
            lines.append(
                f"| {config.get('name', 'N/A')} | "
                f"{metrics.get('accuracy', 0):.4f} | "
                f"{metrics.get('f1', 0):.4f} | "
                f"{metrics.get('auroc', 0):.4f} |"
            )
        lines.append("")
    
    lines.extend([
        "## Figures",
        "",
    ])
    
    for fig in data.get("figures", []):
        fig_name = Path(fig).name
        lines.append(f"- [{fig_name}]({fig})")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(description="Run LegalGPT evaluation")
    parser.add_argument(
        "--adapter-path",
        type=str,
        default=str(MODELS_DIR / "legalgpt-qlora" / "best"),
        help="Path to trained model adapter",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(RESULTS_DIR),
        help="Output directory for results",
    )
    parser.add_argument(
        "--run-ablations",
        action="store_true",
        help="Run ablation studies",
    )
    parser.add_argument(
        "--generate-paper",
        action="store_true",
        help="Generate paper results and figures",
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip main evaluation (use existing results)",
    )
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print("LegalGPT Evaluation Pipeline")
    print("=" * 60)
    
    # Load data
    print("\nLoading test data...")
    test_data = load_test_data()
    print(f"Loaded {len(test_data)} test cases")
    
    # Get retriever
    retriever = get_retriever()
    
    # Run main evaluation
    if not args.skip_eval:
        if Path(args.adapter_path).exists():
            run_evaluation(
                adapter_path=args.adapter_path,
                output_dir=output_dir,
                test_data=test_data,
                retriever=retriever,
            )
        else:
            print(f"\nAdapter not found at {args.adapter_path}")
            print("Skipping evaluation. Train the model first.")
    
    # Run ablations
    if args.run_ablations:
        if Path(args.adapter_path).exists():
            run_ablation_studies(
                adapter_path=args.adapter_path,
                output_dir=output_dir,
                test_data=test_data,
                retriever=retriever,
            )
        else:
            print("\nSkipping ablations - model not found")
    
    # Generate paper results
    if args.generate_paper:
        generate_paper_results(output_dir)
    
    print("\n" + "=" * 60)
    print("Evaluation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
