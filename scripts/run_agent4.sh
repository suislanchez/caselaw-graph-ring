#!/bin/bash
# Agent 4: Model & Training Pipeline Runner

set -e

echo "=========================================="
echo "LegalGPT - Agent 4: Model & Training"
echo "=========================================="

# Check environment
if [ -z "$HF_TOKEN" ]; then
    echo "Warning: HF_TOKEN not set. Set it for Mistral model access."
fi

# Navigate to project root
cd "$(dirname "$0")/.."

# Install dependencies
echo "Installing dependencies..."
pip install -q -r requirements.txt

# Create necessary directories
mkdir -p data/{raw,processed,citations,embeddings,splits}
mkdir -p models results

# Run Modal setup check
echo ""
echo "Checking Modal configuration..."
modal setup --check 2>/dev/null || {
    echo "Modal not configured. Run 'modal setup' to authenticate."
    exit 1
}

echo ""
echo "Agent 4 setup complete!"
echo ""
echo "Available commands:"
echo "  1. Train model on Modal:"
echo "     modal run src/model/modal_config.py"
echo ""
echo "  2. Deploy inference endpoint:"
echo "     modal deploy src/model/modal_config.py"
echo ""
echo "  3. Run local training (requires GPU):"
echo "     python -c 'from src.model import train_legalgpt; ...'"
echo ""
echo "  4. Run ablation studies:"
echo "     python -c 'from src.model.ablations import run_ablation_study; ...'"
echo ""
