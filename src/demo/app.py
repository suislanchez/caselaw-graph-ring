"""Gradio demo for LegalGPT legal outcome prediction."""

import json
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import MODELS_DIR, RESULTS_DIR, DATA_DIR


# Lazy imports for faster startup
def get_gradio():
    import gradio as gr
    return gr


def get_predictor(adapter_path: Optional[str] = None):
    """Lazy load predictor to avoid startup delay."""
    from src.model.inference import LegalGPTPredictor
    
    if adapter_path is None:
        adapter_path = str(MODELS_DIR / "legalgpt-qlora" / "best")
    
    return LegalGPTPredictor(adapter_path=adapter_path)


def get_retriever():
    """Lazy load graph retriever."""
    try:
        from src.graph.retriever import get_similar_cases
        return get_similar_cases
    except ImportError:
        return None


# Global predictor (lazy loaded)
_predictor = None
_retriever = None


def load_models(adapter_path: Optional[str] = None):
    """Load models on first use."""
    global _predictor, _retriever
    
    if _predictor is None:
        _predictor = get_predictor(adapter_path)
    
    if _retriever is None:
        _retriever = get_retriever()
    
    return _predictor, _retriever


def predict_outcome(
    case_text: str,
    use_retrieval: bool = True,
    num_similar: int = 5,
    show_similar: bool = True,
) -> Tuple[str, str, str]:
    """
    Predict case outcome.
    
    Returns:
        Tuple of (prediction_html, confidence_html, similar_cases_html)
    """
    if not case_text.strip():
        return (
            "<p style='color: red;'>Please enter case text.</p>",
            "",
            "",
        )
    
    predictor, retriever = load_models()
    
    # Get similar cases if retrieval enabled
    similar_cases = []
    if use_retrieval and retriever:
        try:
            similar_cases = retriever(case_text[:500], k=num_similar)
        except Exception as e:
            print(f"Retrieval error: {e}")
    
    # Make prediction
    result = predictor.predict(
        case_text=case_text,
        similar_cases=similar_cases,
    )
    
    # Format prediction
    prediction = result.prediction
    confidence = result.confidence
    
    if prediction == "petitioner":
        color = "#28a745"  # Green
        icon = "✓"
    elif prediction == "respondent":
        color = "#dc3545"  # Red
        icon = "✗"
    else:
        color = "#6c757d"  # Gray
        icon = "?"
    
    prediction_html = f"""
    <div style="text-align: center; padding: 20px;">
        <h2 style="color: {color}; margin: 0;">
            {icon} {prediction.upper()}
        </h2>
        <p style="color: #666; margin-top: 10px;">
            Predicted winner of the case
        </p>
    </div>
    """
    
    # Confidence bar
    conf_pct = confidence * 100
    confidence_html = f"""
    <div style="padding: 10px;">
        <p style="margin-bottom: 5px;"><strong>Confidence: {conf_pct:.1f}%</strong></p>
        <div style="background: #e9ecef; border-radius: 4px; height: 20px; overflow: hidden;">
            <div style="background: {color}; width: {conf_pct}%; height: 100%;"></div>
        </div>
    </div>
    """
    
    # Similar cases
    similar_html = ""
    if show_similar and similar_cases:
        similar_html = "<div style='padding: 10px;'><h4>Similar Precedent Cases:</h4>"
        for i, case in enumerate(similar_cases[:5], 1):
            case_name = case.get("name", f"Case {i}")[:50]
            case_outcome = case.get("outcome", "unknown")
            outcome_color = "#28a745" if case_outcome == "petitioner" else "#dc3545"
            
            similar_html += f"""
            <div style="border: 1px solid #ddd; border-radius: 4px; padding: 10px; margin: 5px 0;">
                <strong>{i}. {case_name}</strong>
                <span style="float: right; color: {outcome_color};">{case_outcome}</span>
                <p style="font-size: 0.9em; color: #666; margin-top: 5px;">
                    {case.get('text', '')[:200]}...
                </p>
            </div>
            """
        similar_html += "</div>"
    elif show_similar:
        similar_html = "<p style='color: #666;'>No similar cases retrieved. Enable graph retrieval for precedent-based prediction.</p>"
    
    return prediction_html, confidence_html, similar_html


def load_sample_cases() -> List[Dict[str, str]]:
    """Load sample cases for demo."""
    samples_path = DATA_DIR / "splits" / "test.json"
    
    if samples_path.exists():
        with open(samples_path) as f:
            data = json.load(f)
        return data[:10]  # First 10 for demo
    
    # Fallback samples
    return [
        {
            "name": "Sample Case 1",
            "text": "The petitioner argues that the lower court erred in its interpretation of the Commerce Clause. The respondent contends that the regulation falls within the state's police powers...",
        },
        {
            "name": "Sample Case 2", 
            "text": "This case involves a challenge to the constitutionality of a federal statute under the First Amendment. The petitioner claims the statute imposes an unconstitutional prior restraint...",
        },
    ]


def create_demo(
    adapter_path: Optional[str] = None,
    share: bool = False,
) -> Any:
    """
    Create Gradio demo interface.
    
    Args:
        adapter_path: Path to trained model adapter
        share: Whether to create public link
    
    Returns:
        Gradio Blocks interface
    """
    gr = get_gradio()
    
    # Custom CSS
    css = """
    .main-title {
        text-align: center;
        margin-bottom: 20px;
    }
    .prediction-box {
        border: 2px solid #ddd;
        border-radius: 8px;
        padding: 15px;
        margin: 10px 0;
    }
    """
    
    with gr.Blocks(css=css, title="LegalGPT Demo") as demo:
        gr.Markdown("""
        # LegalGPT: Legal Outcome Prediction
        
        Predict Supreme Court case outcomes using citation graph-enhanced LLM.
        
        Enter case text below or select a sample case.
        """)
        
        with gr.Row():
            with gr.Column(scale=2):
                case_input = gr.Textbox(
                    label="Case Text",
                    placeholder="Enter the case text here...",
                    lines=10,
                    max_lines=20,
                )
                
                with gr.Row():
                    use_retrieval = gr.Checkbox(
                        label="Use Graph Retrieval",
                        value=True,
                        info="Retrieve similar precedent cases",
                    )
                    num_similar = gr.Slider(
                        minimum=1,
                        maximum=20,
                        value=5,
                        step=1,
                        label="Number of Similar Cases",
                    )
                
                with gr.Row():
                    clear_btn = gr.Button("Clear", variant="secondary")
                    predict_btn = gr.Button("Predict Outcome", variant="primary")
            
            with gr.Column(scale=1):
                prediction_output = gr.HTML(label="Prediction")
                confidence_output = gr.HTML(label="Confidence")
        
        similar_output = gr.HTML(label="Similar Cases")
        
        # Sample cases
        gr.Markdown("### Sample Cases")
        samples = load_sample_cases()
        
        with gr.Row():
            for i, sample in enumerate(samples[:3]):
                btn = gr.Button(sample.get("name", f"Sample {i+1}")[:20], size="sm")
                btn.click(
                    fn=lambda s=sample: s.get("text", ""),
                    outputs=case_input,
                )
        
        # Event handlers
        predict_btn.click(
            fn=predict_outcome,
            inputs=[case_input, use_retrieval, num_similar, gr.State(True)],
            outputs=[prediction_output, confidence_output, similar_output],
        )
        
        clear_btn.click(
            fn=lambda: ("", "", "", ""),
            outputs=[case_input, prediction_output, confidence_output, similar_output],
        )
        
        # Footer
        gr.Markdown("""
        ---
        **Model:** Mistral-7B-Instruct with QLoRA fine-tuning  
        **Retrieval:** GraphSAGE on citation network  
        **Data:** Supreme Court Database (SCDB)
        """)
    
    return demo


def launch_demo(
    adapter_path: Optional[str] = None,
    share: bool = False,
    server_port: int = 7860,
    **kwargs,
):
    """Launch the Gradio demo."""
    demo = create_demo(adapter_path=adapter_path, share=share)
    demo.launch(
        share=share,
        server_port=server_port,
        **kwargs,
    )


# CLI entrypoint
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch LegalGPT demo")
    parser.add_argument("--adapter-path", type=str, help="Path to model adapter")
    parser.add_argument("--share", action="store_true", help="Create public link")
    parser.add_argument("--port", type=int, default=7860, help="Server port")
    
    args = parser.parse_args()
    
    launch_demo(
        adapter_path=args.adapter_path,
        share=args.share,
        server_port=args.port,
    )
