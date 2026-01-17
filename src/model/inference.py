"""Inference pipeline for LegalGPT predictions."""

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional, Any, Union, Callable
import re

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import sys
sys.path.append(str(__file__).rsplit("/", 2)[0])
from config import MODELS_DIR, MODEL_NAME
from model.model import LegalGPTModel, create_prompt, LABEL_TO_OUTCOME


@dataclass
class PredictionResult:
    """Result of a single prediction."""
    case_id: str
    prediction: str  # "petitioner" or "respondent"
    confidence: float  # 0.0 to 1.0
    raw_output: str
    ground_truth: Optional[str] = None
    
    @property
    def is_correct(self) -> Optional[bool]:
        if self.ground_truth is None:
            return None
        return self.prediction == self.ground_truth
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "case_id": self.case_id,
            "prediction": self.prediction,
            "confidence": self.confidence,
            "raw_output": self.raw_output,
            "ground_truth": self.ground_truth,
            "is_correct": self.is_correct,
        }


class LegalGPTPredictor:
    """
    Inference engine for LegalGPT.
    
    Supports:
    - Single case prediction
    - Batch prediction
    - Integration with graph retriever
    - Confidence estimation
    """
    
    def __init__(
        self,
        model: Optional[LegalGPTModel] = None,
        adapter_path: Optional[str] = None,
        retriever: Optional[Callable] = None,
        device: Optional[str] = None,
        max_new_tokens: int = 20,
        num_similar_cases: int = 5,
    ):
        """
        Initialize predictor.
        
        Args:
            model: Pre-loaded LegalGPT model
            adapter_path: Path to trained LoRA adapters (if model not provided)
            retriever: Function to retrieve similar cases: (case_id, k) -> List[Dict]
            device: Device to run inference on
            max_new_tokens: Maximum tokens to generate
            num_similar_cases: Number of similar cases for context
        """
        self.retriever = retriever
        self.max_new_tokens = max_new_tokens
        self.num_similar_cases = num_similar_cases
        
        # Determine device
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = device
        
        # Load model
        if model is not None:
            self.model = model
            self.tokenizer = model.tokenizer
        elif adapter_path is not None:
            self.model = LegalGPTModel.from_pretrained(
                adapter_path=adapter_path,
                device_map="auto" if device == "cuda" else device,
            )
            self.tokenizer = self.model.tokenizer
        else:
            raise ValueError("Either model or adapter_path must be provided")
        
        self.model.eval()
    
    def predict(
        self,
        case_text: str,
        case_id: Optional[str] = None,
        similar_cases: Optional[List[Dict]] = None,
        ground_truth: Optional[str] = None,
    ) -> PredictionResult:
        """
        Predict outcome for a single case.
        
        Args:
            case_text: Text of the case to predict
            case_id: Optional case identifier
            similar_cases: Pre-computed similar cases (if retriever not used)
            ground_truth: Optional ground truth for evaluation
        
        Returns:
            PredictionResult with prediction and confidence
        """
        case_id = case_id or "unknown"
        
        # Get similar cases from retriever if available
        if similar_cases is None and self.retriever is not None:
            similar_cases = self.retriever(case_id, k=self.num_similar_cases)
        similar_cases = similar_cases or []
        
        # Create prompt
        prompt = create_prompt(
            case_text=case_text,
            similar_cases=similar_cases,
        )
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=4096,
        ).to(self.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                input_ids=inputs["input_ids"],
                attention_mask=inputs["attention_mask"],
                max_new_tokens=self.max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
                return_dict_in_generate=True,
                output_scores=True,
            )
        
        # Decode output
        generated_ids = outputs.sequences[0][inputs["input_ids"].shape[1]:]
        raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
        
        # Parse prediction
        prediction, confidence = self._parse_output(raw_output, outputs)
        
        return PredictionResult(
            case_id=case_id,
            prediction=prediction,
            confidence=confidence,
            raw_output=raw_output,
            ground_truth=ground_truth,
        )
    
    def predict_batch(
        self,
        cases: List[Dict[str, Any]],
        batch_size: int = 8,
        show_progress: bool = True,
    ) -> List[PredictionResult]:
        """
        Predict outcomes for multiple cases.
        
        Args:
            cases: List of case dictionaries with 'id', 'text', optional 'outcome'
            batch_size: Number of cases to process at once
            show_progress: Whether to show progress bar
        
        Returns:
            List of PredictionResults
        """
        results = []
        
        iterator = range(0, len(cases), batch_size)
        if show_progress:
            iterator = tqdm(iterator, desc="Predicting", total=(len(cases) + batch_size - 1) // batch_size)
        
        for i in iterator:
            batch = cases[i:i + batch_size]
            
            for case in batch:
                result = self.predict(
                    case_text=case.get("text", ""),
                    case_id=case.get("id", f"case_{i}"),
                    similar_cases=case.get("similar_cases"),
                    ground_truth=case.get("outcome"),
                )
                results.append(result)
        
        return results
    
    def predict_from_dataloader(
        self,
        dataloader: DataLoader,
        show_progress: bool = True,
    ) -> List[PredictionResult]:
        """
        Predict from a DataLoader (useful for test set).
        
        Args:
            dataloader: DataLoader with case data
            show_progress: Whether to show progress bar
        
        Returns:
            List of PredictionResults
        """
        results = []
        
        iterator = dataloader
        if show_progress:
            iterator = tqdm(iterator, desc="Predicting")
        
        for batch in iterator:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            case_ids = batch.get("case_ids", ["unknown"] * len(input_ids))
            ground_truths = batch.get("outcomes", [None] * len(input_ids))
            
            # Generate for each item in batch
            with torch.no_grad():
                for idx in range(len(input_ids)):
                    outputs = self.model.generate(
                        input_ids=input_ids[idx:idx+1],
                        attention_mask=attention_mask[idx:idx+1],
                        max_new_tokens=self.max_new_tokens,
                        do_sample=False,
                        pad_token_id=self.tokenizer.pad_token_id,
                        return_dict_in_generate=True,
                        output_scores=True,
                    )
                    
                    # Find where generation starts (after input)
                    input_len = attention_mask[idx].sum().item()
                    generated_ids = outputs.sequences[0][input_len:]
                    raw_output = self.tokenizer.decode(generated_ids, skip_special_tokens=True)
                    
                    prediction, confidence = self._parse_output(raw_output, outputs)
                    
                    results.append(PredictionResult(
                        case_id=case_ids[idx],
                        prediction=prediction,
                        confidence=confidence,
                        raw_output=raw_output,
                        ground_truth=ground_truths[idx] if ground_truths else None,
                    ))
        
        return results
    
    def _parse_output(
        self,
        raw_output: str,
        generation_output: Any = None,
    ) -> tuple[str, float]:
        """
        Parse model output to extract prediction and confidence.
        
        Args:
            raw_output: Raw text output from model
            generation_output: Full generation output with scores
        
        Returns:
            Tuple of (prediction, confidence)
        """
        # Clean and lowercase
        text = raw_output.strip().lower()
        
        # Look for petitioner/respondent in output
        if "petitioner" in text:
            prediction = "petitioner"
        elif "respondent" in text:
            prediction = "respondent"
        else:
            # Try to extract from first word
            first_word = text.split()[0] if text.split() else ""
            if first_word.startswith("pet"):
                prediction = "petitioner"
            elif first_word.startswith("res"):
                prediction = "respondent"
            else:
                prediction = "unknown"
        
        # Estimate confidence from generation scores
        confidence = 0.5  # Default
        
        if generation_output is not None and hasattr(generation_output, "scores") and generation_output.scores:
            # Get probability of first token
            first_scores = generation_output.scores[0][0]  # First token, first batch item
            probs = torch.softmax(first_scores, dim=-1)
            
            # Get max probability as confidence proxy
            confidence = probs.max().item()
            
            # Normalize to reasonable range (usually very high with greedy)
            confidence = min(0.99, max(0.5, confidence))
        
        return prediction, confidence
    
    def save_predictions(
        self,
        results: List[PredictionResult],
        output_path: Union[str, Path],
    ):
        """Save predictions to JSON file."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w") as f:
            json.dump(
                [r.to_dict() for r in results],
                f,
                indent=2,
            )
        
        print(f"Saved {len(results)} predictions to {output_path}")
    
    @classmethod
    def load_predictions(cls, path: Union[str, Path]) -> List[PredictionResult]:
        """Load predictions from JSON file."""
        with open(path) as f:
            data = json.load(f)
        
        return [
            PredictionResult(
                case_id=item["case_id"],
                prediction=item["prediction"],
                confidence=item["confidence"],
                raw_output=item["raw_output"],
                ground_truth=item.get("ground_truth"),
            )
            for item in data
        ]


def run_inference(
    adapter_path: str,
    test_data: List[Dict],
    retriever: Optional[Callable] = None,
    output_path: Optional[str] = None,
    **kwargs,
) -> List[PredictionResult]:
    """
    Convenience function to run inference.
    
    Args:
        adapter_path: Path to trained model
        test_data: Test cases
        retriever: Similar case retriever
        output_path: Path to save predictions
        **kwargs: Additional arguments for predictor
    
    Returns:
        List of prediction results
    """
    predictor = LegalGPTPredictor(
        adapter_path=adapter_path,
        retriever=retriever,
        **kwargs,
    )
    
    results = predictor.predict_batch(test_data)
    
    if output_path:
        predictor.save_predictions(results, output_path)
    
    return results


# Demo function for quick testing
def demo_predict(case_text: str, adapter_path: Optional[str] = None):
    """Quick demo prediction."""
    if adapter_path is None:
        adapter_path = str(MODELS_DIR / "legalgpt-qlora" / "best")
    
    predictor = LegalGPTPredictor(adapter_path=adapter_path)
    result = predictor.predict(case_text)
    
    print(f"Prediction: {result.prediction}")
    print(f"Confidence: {result.confidence:.2%}")
    print(f"Raw output: {result.raw_output}")
    
    return result
