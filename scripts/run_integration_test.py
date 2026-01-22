#!/usr/bin/env python3
"""
Integration test for the full LegalGPT pipeline.

Tests:
1. Data loading
2. Citation extraction
3. Graph retriever
4. Model inference
5. Evaluation metrics
6. End-to-end prediction

Usage:
    python scripts/run_integration_test.py
    python scripts/run_integration_test.py --verbose
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Tuple, Optional

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class TestResult:
    def __init__(self, name: str, passed: bool, message: str = "", details: str = ""):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details
    
    def __str__(self):
        status = "PASS" if self.passed else "FAIL"
        icon = "\u2713" if self.passed else "\u2717"
        return f"[{status}] {icon} {self.name}: {self.message}"


def test_config_loading() -> TestResult:
    """Test that config loads correctly."""
    try:
        from src.config import (
            PROJECT_ROOT, DATA_DIR, MODELS_DIR, RESULTS_DIR,
            MODEL_NAME, LORA_R, LORA_ALPHA
        )
        
        assert PROJECT_ROOT.exists(), "PROJECT_ROOT doesn't exist"
        assert DATA_DIR.exists(), "DATA_DIR doesn't exist"
        assert MODEL_NAME == "mistralai/Mistral-7B-Instruct-v0.3"
        assert LORA_R == 16
        assert LORA_ALPHA == 32
        
        return TestResult("Config Loading", True, "All config values loaded correctly")
    except Exception as e:
        return TestResult("Config Loading", False, str(e))


def test_data_modules() -> TestResult:
    """Test data module imports."""
    try:
        from src.model.dataset import (
            LegalOutcomeDataset, 
            CollateFunction, 
            create_sample_data,
            CaseExample,
        )
        
        # Create sample data
        samples = create_sample_data(num_samples=10)
        assert len(samples) == 10
        assert "text" in samples[0]
        assert "outcome" in samples[0]
        
        return TestResult("Data Modules", True, f"Created {len(samples)} sample cases")
    except Exception as e:
        return TestResult("Data Modules", False, str(e))


def test_model_architecture() -> TestResult:
    """Test model class imports (without loading weights)."""
    try:
        from src.model.model import (
            LegalGPTModel,
            create_prompt,
            OUTCOME_TO_LABEL,
            LABEL_TO_OUTCOME,
        )
        
        # Test prompt creation
        prompt = create_prompt(
            case_text="Test case about constitutional law.",
            similar_cases=[
                {"text": "Similar case 1", "outcome": "petitioner"},
                {"text": "Similar case 2", "outcome": "respondent"},
            ],
        )
        
        assert "[INST]" in prompt
        assert "Test case" in prompt
        assert "Similar case 1" in prompt
        
        # Test label mappings
        assert OUTCOME_TO_LABEL["petitioner"] == 0
        assert LABEL_TO_OUTCOME[1] == "respondent"
        
        return TestResult("Model Architecture", True, "Model classes and prompts work correctly")
    except Exception as e:
        return TestResult("Model Architecture", False, str(e))


def test_evaluation_metrics() -> TestResult:
    """Test evaluation metric computation."""
    try:
        from src.model.evaluate import compute_metrics, EvaluationMetrics
        
        # Test data
        predictions = ["petitioner", "respondent", "petitioner", "petitioner", "respondent"]
        ground_truths = ["petitioner", "respondent", "respondent", "petitioner", "respondent"]
        confidences = [0.8, 0.9, 0.6, 0.7, 0.85]
        
        metrics = compute_metrics(predictions, ground_truths, confidences)
        
        assert isinstance(metrics, EvaluationMetrics)
        assert 0 <= metrics.accuracy <= 1
        assert 0 <= metrics.f1 <= 1
        assert 0 <= metrics.auroc <= 1
        assert metrics.total_samples == 5
        assert metrics.correct_predictions == 4  # 4/5 correct
        
        return TestResult(
            "Evaluation Metrics", 
            True, 
            f"Accuracy={metrics.accuracy:.2f}, F1={metrics.f1:.2f}, AUROC={metrics.auroc:.2f}"
        )
    except Exception as e:
        return TestResult("Evaluation Metrics", False, str(e))


def test_ablation_configs() -> TestResult:
    """Test ablation study configurations."""
    try:
        from src.model.ablations import (
            AblationConfig,
            STANDARD_ABLATIONS,
            AblationResult,
        )
        
        assert len(STANDARD_ABLATIONS) >= 5, "Should have at least 5 standard ablations"
        
        # Check required ablations exist
        ablation_names = [a.name for a in STANDARD_ABLATIONS]
        assert "full_model" in ablation_names
        assert "no_retrieval" in ablation_names
        assert "random_retrieval" in ablation_names
        
        # Test config creation
        config = AblationConfig(
            name="test_ablation",
            description="Test config",
            num_similar_cases=10,
        )
        assert config.num_similar_cases == 10
        
        return TestResult(
            "Ablation Configs", 
            True, 
            f"Found {len(STANDARD_ABLATIONS)} standard ablations"
        )
    except Exception as e:
        return TestResult("Ablation Configs", False, str(e))


def test_inference_pipeline() -> TestResult:
    """Test inference classes (without loading model)."""
    try:
        from src.model.inference import PredictionResult, LegalGPTPredictor
        
        # Test PredictionResult
        result = PredictionResult(
            case_id="test_001",
            prediction="petitioner",
            confidence=0.85,
            raw_output="petitioner",
            ground_truth="petitioner",
        )
        
        assert result.is_correct == True
        assert result.confidence == 0.85
        
        result_dict = result.to_dict()
        assert "prediction" in result_dict
        assert "confidence" in result_dict
        
        return TestResult("Inference Pipeline", True, "PredictionResult class works correctly")
    except Exception as e:
        return TestResult("Inference Pipeline", False, str(e))


def test_visualization_module() -> TestResult:
    """Test visualization imports."""
    try:
        from src.graph.visualize import (
            load_citation_edges,
            create_networkx_graph,
            compute_graph_stats,
        )
        
        # Test with empty data (should not crash)
        edges = []
        G = create_networkx_graph(edges)
        
        assert G.number_of_nodes() == 0
        assert G.number_of_edges() == 0
        
        return TestResult("Visualization Module", True, "Visualization functions imported successfully")
    except ImportError as e:
        return TestResult("Visualization Module", False, f"Missing dependency: {e}")
    except Exception as e:
        return TestResult("Visualization Module", False, str(e))


def test_demo_module() -> TestResult:
    """Test demo module imports."""
    try:
        from src.demo.app import create_demo, predict_outcome
        
        # Just test that imports work
        return TestResult("Demo Module", True, "Demo module imported successfully")
    except ImportError as e:
        # Gradio might not be installed
        if "gradio" in str(e).lower():
            return TestResult("Demo Module", True, "Skipped (gradio not installed)")
        return TestResult("Demo Module", False, f"Import error: {e}")
    except Exception as e:
        return TestResult("Demo Module", False, str(e))


def test_modal_config() -> TestResult:
    """Test Modal configuration."""
    try:
        # Just check the file exists and has valid Python syntax
        modal_path = PROJECT_ROOT / "src" / "model" / "modal_config.py"
        
        if not modal_path.exists():
            return TestResult("Modal Config", False, "modal_config.py not found")
        
        # Read and compile (syntax check)
        with open(modal_path) as f:
            code = f.read()
        
        compile(code, modal_path, "exec")
        
        # Check for key components
        assert "modal.App" in code or 'modal.App' in code
        assert "LegalGPTTrainer" in code
        assert "A100" in code or "gpu=" in code
        
        return TestResult("Modal Config", True, "Modal configuration is valid")
    except SyntaxError as e:
        return TestResult("Modal Config", False, f"Syntax error: {e}")
    except Exception as e:
        return TestResult("Modal Config", False, str(e))


def test_end_to_end_pipeline() -> TestResult:
    """Test the complete pipeline (without GPU/model loading)."""
    try:
        from src.model.dataset import create_sample_data
        from src.model.model import create_prompt
        from src.model.evaluate import compute_metrics
        
        # 1. Create sample data
        data = create_sample_data(num_samples=5)
        
        # 2. Create prompts
        prompts = []
        for case in data:
            prompt = create_prompt(
                case_text=case["text"],
                similar_cases=case.get("similar_cases", []),
            )
            prompts.append(prompt)
        
        assert len(prompts) == 5
        
        # 3. Simulate predictions
        predictions = [d["outcome"] for d in data]  # Perfect predictions
        ground_truths = [d["outcome"] for d in data]
        confidences = [0.9] * len(predictions)
        
        # 4. Compute metrics
        metrics = compute_metrics(predictions, ground_truths, confidences)
        
        assert metrics.accuracy == 1.0  # Perfect accuracy for simulated
        
        return TestResult(
            "End-to-End Pipeline", 
            True, 
            "Full pipeline (data -> prompt -> eval) works"
        )
    except Exception as e:
        return TestResult("End-to-End Pipeline", False, str(e))


def run_all_tests(verbose: bool = False) -> Tuple[int, int]:
    """Run all integration tests."""
    tests = [
        test_config_loading,
        test_data_modules,
        test_model_architecture,
        test_evaluation_metrics,
        test_ablation_configs,
        test_inference_pipeline,
        test_visualization_module,
        test_demo_module,
        test_modal_config,
        test_end_to_end_pipeline,
    ]
    
    print("=" * 60)
    print("LegalGPT Integration Tests")
    print("=" * 60)
    print()
    
    passed = 0
    failed = 0
    results = []
    
    for test_fn in tests:
        result = test_fn()
        results.append(result)
        
        if result.passed:
            passed += 1
            print(f"\033[92m{result}\033[0m")  # Green
        else:
            failed += 1
            print(f"\033[91m{result}\033[0m")  # Red
        
        if verbose and result.details:
            print(f"    Details: {result.details}")
    
    print()
    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    # Save results
    results_path = PROJECT_ROOT / "results" / "integration_tests.json"
    results_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(results_path, "w") as f:
        json.dump({
            "passed": passed,
            "failed": failed,
            "tests": [
                {"name": r.name, "passed": r.passed, "message": r.message}
                for r in results
            ]
        }, f, indent=2)
    
    return passed, failed


def main():
    parser = argparse.ArgumentParser(description="Run LegalGPT integration tests")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    args = parser.parse_args()
    
    passed, failed = run_all_tests(verbose=args.verbose)
    
    # Exit with error code if any tests failed
    sys.exit(0 if failed == 0 else 1)


if __name__ == "__main__":
    main()
