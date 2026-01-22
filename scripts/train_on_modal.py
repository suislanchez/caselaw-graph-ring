#!/usr/bin/env python3
"""
Train LegalGPT on Modal with A100 GPU.

This script:
1. Prepares training data with similar cases from graph retriever
2. Uploads data to Modal volumes
3. Runs QLoRA training on Modal A100
4. Downloads trained model and runs evaluation

Usage:
    # Full training pipeline
    python scripts/train_on_modal.py train

    # Just prepare data
    python scripts/train_on_modal.py prepare

    # Run inference on test set
    python scripts/train_on_modal.py evaluate

    # Run ablation studies
    python scripts/train_on_modal.py ablations
"""

import argparse
import json
import os
import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.config import (
    SPLITS_DIR,
    RESULTS_DIR,
    MODELS_DIR,
    DATA_DIR,
    DEFAULT_TOP_K,
)


def prepare_data_with_retrieval(
    use_graph_retrieval: bool = True,
    num_similar_cases: int = DEFAULT_TOP_K,
):
    """
    Prepare training data with similar cases from graph retriever.

    This enriches each case with similar precedent cases for RAG-style training.
    """
    print("=" * 60)
    print("Preparing Training Data with Graph Retrieval")
    print("=" * 60)

    # Load split data
    train_path = SPLITS_DIR / "train.json"
    val_path = SPLITS_DIR / "val.json"
    test_path = SPLITS_DIR / "test.json"

    if not train_path.exists():
        print(f"Error: Training data not found at {train_path}")
        print("Run Agent 1 (Data Pipeline) first to create data splits.")
        return False

    with open(train_path) as f:
        train_data = json.load(f)
    with open(val_path) as f:
        val_data = json.load(f)
    with open(test_path) as f:
        test_data = json.load(f)

    print(f"Loaded: {len(train_data)} train, {len(val_data)} val, {len(test_data)} test")

    # Initialize retriever if using graph
    retriever = None
    if use_graph_retrieval:
        try:
            from src.graph import CaseRetriever, check_health

            if not check_health():
                print("Warning: Neo4j not running. Using pre-computed similar cases or none.")
            else:
                retriever = CaseRetriever()
                print(f"Graph retriever initialized (k={num_similar_cases})")
        except Exception as e:
            print(f"Warning: Could not initialize graph retriever: {e}")

    def enrich_with_similar_cases(data, retriever, k):
        """Add similar cases to each example."""
        enriched = []

        for i, case in enumerate(data):
            case_id = case.get("id", case.get("cap_id", str(i)))

            # Get similar cases
            if retriever:
                try:
                    similar = retriever.get_similar_cases(
                        case_id, k=k, method="hybrid", include_text=True
                    )
                    case["similar_cases"] = [
                        {
                            "id": s.case_id,
                            "name": s.name,
                            "text": s.text[:2000],  # Truncate for efficiency
                            "outcome": s.outcome,
                            "similarity_score": s.similarity_score,
                        }
                        for s in similar
                    ]
                except Exception as e:
                    case["similar_cases"] = case.get("similar_cases", [])
            else:
                case["similar_cases"] = case.get("similar_cases", [])

            enriched.append(case)

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(data)} cases")

        return enriched

    # Enrich data
    print("\nEnriching training data with similar cases...")
    train_enriched = enrich_with_similar_cases(train_data, retriever, num_similar_cases)

    print("\nEnriching validation data...")
    val_enriched = enrich_with_similar_cases(val_data, retriever, num_similar_cases)

    print("\nEnriching test data...")
    test_enriched = enrich_with_similar_cases(test_data, retriever, num_similar_cases)

    if retriever:
        retriever.close()

    # Save enriched data for Modal
    modal_data_dir = DATA_DIR / "modal"
    modal_data_dir.mkdir(parents=True, exist_ok=True)

    with open(modal_data_dir / "train.json", "w") as f:
        json.dump(train_enriched, f)
    with open(modal_data_dir / "val.json", "w") as f:
        json.dump(val_enriched, f)
    with open(modal_data_dir / "test.json", "w") as f:
        json.dump(test_enriched, f)

    print(f"\nSaved enriched data to {modal_data_dir}")
    print(f"  train.json: {len(train_enriched)} examples")
    print(f"  val.json: {len(val_enriched)} examples")
    print(f"  test.json: {len(test_enriched)} examples")

    return True


def upload_data_to_modal():
    """Upload prepared data to Modal volume."""
    print("\n" + "=" * 60)
    print("Uploading Data to Modal Volume")
    print("=" * 60)

    import subprocess

    modal_data_dir = DATA_DIR / "modal"

    if not modal_data_dir.exists():
        print(f"Error: Prepared data not found at {modal_data_dir}")
        print("Run 'prepare' command first.")
        return False

    # Use Modal CLI to upload
    for filename in ["train.json", "val.json", "test.json"]:
        filepath = modal_data_dir / filename
        if filepath.exists():
            print(f"Uploading {filename}...")
            cmd = [
                "modal", "volume", "put",
                "legalgpt-data",
                str(filepath),
                f"/{filename}",
            ]
            result = subprocess.run(cmd, capture_output=True, text=True)
            if result.returncode != 0:
                print(f"  Error: {result.stderr}")
            else:
                print(f"  Uploaded {filename}")

    return True


def run_training_on_modal(
    num_epochs: int = 3,
    batch_size: int = 4,
    learning_rate: float = 2e-4,
    max_length: int = 4096,
):
    """Run training on Modal A100."""
    print("\n" + "=" * 60)
    print("Starting Training on Modal A100")
    print("=" * 60)

    import modal

    # Import the Modal app
    from src.model.modal_config import LegalGPTTrainer, DATA_DIR as MODAL_DATA_DIR, RESULTS_DIR as MODAL_RESULTS_DIR

    print(f"Configuration:")
    print(f"  Epochs: {num_epochs}")
    print(f"  Batch size: {batch_size}")
    print(f"  Learning rate: {learning_rate}")
    print(f"  Max length: {max_length}")

    # Create trainer and run
    with modal.enable_output():
        trainer = LegalGPTTrainer()

        result = trainer.train.remote(
            train_data_path=f"{MODAL_DATA_DIR}/train.json",
            val_data_path=f"{MODAL_DATA_DIR}/val.json",
            output_dir=f"{MODAL_RESULTS_DIR}/legalgpt-qlora",
            num_epochs=num_epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            max_length=max_length,
        )

    print("\n" + "=" * 60)
    print("Training Complete!")
    print("=" * 60)
    print(f"Best validation loss: {result['best_val_loss']:.4f}")
    print(f"Output directory: {result['output_dir']}")

    # Save training stats locally
    stats_path = RESULTS_DIR / "training_stats.json"
    with open(stats_path, "w") as f:
        json.dump(result, f, indent=2)
    print(f"Saved training stats to {stats_path}")

    return result


def run_evaluation():
    """Run evaluation on test set using Modal inference."""
    print("\n" + "=" * 60)
    print("Running Evaluation on Test Set")
    print("=" * 60)

    import modal
    from src.model.modal_config import LegalGPTInference, DATA_DIR as MODAL_DATA_DIR
    from src.model.evaluate import compute_metrics, save_metrics

    # Load test data
    test_path = DATA_DIR / "modal" / "test.json"
    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}")
        return None

    with open(test_path) as f:
        test_data = json.load(f)

    print(f"Evaluating on {len(test_data)} test cases...")

    # Run inference on Modal
    with modal.enable_output():
        inference = LegalGPTInference()
        results = inference.batch_predict.remote(test_data, batch_size=8)

    # Compute metrics
    predictions = [r["prediction"] for r in results]
    ground_truths = [r["ground_truth"] for r in results]

    metrics = compute_metrics(predictions, ground_truths)

    print("\n" + str(metrics))

    # Save metrics
    metrics_path = RESULTS_DIR / "test_metrics.json"
    save_metrics(metrics, metrics_path)

    # Save predictions
    predictions_path = RESULTS_DIR / "test_predictions.json"
    with open(predictions_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved predictions to {predictions_path}")

    return metrics


def run_ablations():
    """Run ablation studies."""
    print("\n" + "=" * 60)
    print("Running Ablation Studies")
    print("=" * 60)

    from src.model.ablations import run_ablation_study, STANDARD_ABLATIONS

    # Load test data
    test_path = DATA_DIR / "modal" / "test.json"
    if not test_path.exists():
        print(f"Error: Test data not found at {test_path}")
        return None

    with open(test_path) as f:
        test_data = json.load(f)

    # Load all data for random baseline
    train_path = DATA_DIR / "modal" / "train.json"
    with open(train_path) as f:
        train_data = json.load(f)
    all_data = train_data + test_data

    # Setup graph retriever if available
    retriever = None
    try:
        from src.graph import CaseRetriever, check_health
        if check_health():
            retriever_obj = CaseRetriever()
            def retriever(case_id, k):
                results = retriever_obj.get_similar_cases(case_id, k=k, method="hybrid")
                return [{"id": r.case_id, "text": r.text, "outcome": r.outcome} for r in results]
    except Exception as e:
        print(f"Graph retriever not available: {e}")

    # Run ablations
    results = run_ablation_study(
        test_data=test_data,
        all_data=all_data,
        graph_retriever=retriever,
        adapter_path=str(MODELS_DIR / "legalgpt-qlora" / "best"),
        output_dir=str(RESULTS_DIR / "ablations"),
    )

    return results


def run_baselines():
    """Run baseline comparisons (majority class, random, etc.)."""
    print("\n" + "=" * 60)
    print("Running Baselines")
    print("=" * 60)

    import random
    from collections import Counter
    from src.model.evaluate import compute_metrics

    # Load test data
    test_path = DATA_DIR / "modal" / "test.json"
    if not test_path.exists():
        test_path = SPLITS_DIR / "test.json"

    with open(test_path) as f:
        test_data = json.load(f)

    ground_truths = [case["outcome"] for case in test_data if case.get("outcome")]

    # Load training data for majority class baseline
    train_path = DATA_DIR / "modal" / "train.json"
    if not train_path.exists():
        train_path = SPLITS_DIR / "train.json"

    with open(train_path) as f:
        train_data = json.load(f)

    train_outcomes = [case["outcome"] for case in train_data if case.get("outcome")]
    outcome_counts = Counter(train_outcomes)
    majority_class = outcome_counts.most_common(1)[0][0]

    results = {}

    # 1. Majority Class Baseline
    print("\n1. Majority Class Baseline")
    majority_preds = [majority_class] * len(ground_truths)
    majority_metrics = compute_metrics(majority_preds, ground_truths)
    results["majority_class"] = majority_metrics
    print(f"   Accuracy: {majority_metrics.accuracy:.4f}")
    print(f"   F1: {majority_metrics.f1:.4f}")

    # 2. Random Baseline
    print("\n2. Random Baseline")
    random_preds = [random.choice(["petitioner", "respondent"]) for _ in ground_truths]
    random_metrics = compute_metrics(random_preds, ground_truths)
    results["random"] = random_metrics
    print(f"   Accuracy: {random_metrics.accuracy:.4f}")
    print(f"   F1: {random_metrics.f1:.4f}")

    # 3. Weighted Random (based on training distribution)
    print("\n3. Weighted Random Baseline")
    weights = [outcome_counts[o] / len(train_outcomes) for o in ["petitioner", "respondent"]]
    weighted_preds = random.choices(["petitioner", "respondent"], weights=weights, k=len(ground_truths))
    weighted_metrics = compute_metrics(weighted_preds, ground_truths)
    results["weighted_random"] = weighted_metrics
    print(f"   Accuracy: {weighted_metrics.accuracy:.4f}")
    print(f"   F1: {weighted_metrics.f1:.4f}")

    # Save baseline results
    baseline_results = {
        name: metrics.to_dict() for name, metrics in results.items()
    }
    baseline_path = RESULTS_DIR / "baseline_results.json"
    with open(baseline_path, "w") as f:
        json.dump(baseline_results, f, indent=2)
    print(f"\nSaved baseline results to {baseline_path}")

    return results


def main():
    parser = argparse.ArgumentParser(description="Train LegalGPT on Modal")
    parser.add_argument(
        "command",
        choices=["prepare", "upload", "train", "evaluate", "ablations", "baselines", "full"],
        help="Command to run",
    )
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size")
    parser.add_argument("--lr", type=float, default=2e-4, help="Learning rate")
    parser.add_argument("--max-length", type=int, default=4096, help="Max sequence length")
    parser.add_argument("--k", type=int, default=5, help="Number of similar cases for retrieval")
    parser.add_argument("--no-retrieval", action="store_true", help="Disable graph retrieval")

    args = parser.parse_args()

    if args.command == "prepare":
        prepare_data_with_retrieval(
            use_graph_retrieval=not args.no_retrieval,
            num_similar_cases=args.k,
        )

    elif args.command == "upload":
        upload_data_to_modal()

    elif args.command == "train":
        run_training_on_modal(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_length=args.max_length,
        )

    elif args.command == "evaluate":
        run_evaluation()

    elif args.command == "ablations":
        run_ablations()

    elif args.command == "baselines":
        run_baselines()

    elif args.command == "full":
        # Full pipeline
        print("Running full training pipeline...")

        # 1. Prepare data
        if not prepare_data_with_retrieval(
            use_graph_retrieval=not args.no_retrieval,
            num_similar_cases=args.k,
        ):
            return

        # 2. Upload to Modal
        upload_data_to_modal()

        # 3. Train
        run_training_on_modal(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            max_length=args.max_length,
        )

        # 4. Evaluate
        run_evaluation()

        # 5. Run baselines for comparison
        run_baselines()

        print("\n" + "=" * 60)
        print("Full Pipeline Complete!")
        print("=" * 60)


if __name__ == "__main__":
    main()
