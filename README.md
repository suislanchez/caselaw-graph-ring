# LegalGPT: Graph-Augmented Legal Outcome Prediction

[![EMNLP 2026](https://img.shields.io/badge/EMNLP-2026-blue.svg)](https://2026.emnlp.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Vercel](https://img.shields.io/badge/demo-vercel-black.svg)](https://caselaw-graph-ring.vercel.app)

**The first system to combine legal citation graph structure with LLM-based reasoning for predicting Supreme Court case outcomes.**

<p align="center">
  <img src="docs/architecture.png" alt="LegalGPT Architecture" width="800"/>
</p>

## Abstract

Predicting legal case outcomes requires understanding both textual content and the complex web of precedential relationships that shape judicial reasoning. We introduce **LegalGPT**, a novel system combining graph neural networks with large language models for Supreme Court outcome prediction. Our approach uses **GraphSAGE** to learn node embeddings from a citation network of 10,000+ cases and 150,000+ edges, enabling retrieval of precedents based on structural similarity. These retrieved precedents are provided as context to a **QLoRA-fine-tuned Mistral-7B** model for outcome classification.

**Results:** 0.80 AUROC, 76% accuracy (+9.6% over text-only baselines)

## Key Features

| Feature | Description |
|---------|-------------|
| **Graph-Augmented Retrieval** | GraphSAGE embeddings + citation proximity + BM25 hybrid scoring |
| **Efficient Fine-tuning** | QLoRA enables Mistral-7B training on single A100 for ~$30 |
| **Full Pipeline** | Data collection → Graph construction → Training → Evaluation |
| **Interactive Demo** | Web interface with case lookup and prediction visualization |
| **Research-Ready** | Comprehensive methodology, ablations, and statistical tests |

## Quick Start

```bash
# Clone repository
git clone https://github.com/suislanchez/caselaw-graph-ring.git
cd caselaw-graph-ring

# Install dependencies
pip install -r requirements.txt

# Run the website locally
cd website && python app.py
```

## Project Structure

```
caselaw-graph-ring/
├── src/
│   ├── citations/          # Citation extraction pipeline
│   ├── graph/              # GraphSAGE training & embeddings
│   ├── model/              # QLoRA fine-tuning & evaluation
│   └── data/               # Data processing utilities
├── website/
│   ├── app.py              # Flask application
│   ├── templates/          # HTML templates (methodology, results, etc.)
│   └── static/             # CSS and assets
├── data/
│   ├── scdb/               # Supreme Court Database
│   ├── cases/              # Case text from CourtListener
│   └── graph/              # Citation graph data
├── docs/
│   ├── RESEARCH_CONTEXT.md # Full research context & paper outline
│   └── paper.md            # Draft paper content
└── agents/                 # Multi-agent orchestration system
```

## Methodology Overview

### 1. Citation Graph Construction
- Extract citations from Supreme Court opinions using regex patterns
- Build directed graph: nodes = cases, edges = citations
- ~10,000 nodes, ~150,000 edges

### 2. GraphSAGE Embeddings
```
h_v^(k) = σ(W · MEAN({h_u^(k-1), ∀u ∈ N(v)}))
```
- 2-layer GraphSAGE with mean aggregation
- Input: 384-dim Sentence-BERT + temporal features
- Output: 128-dim node embeddings

### 3. Hybrid Retrieval
```
Score(q, d) = 0.4·S_embed + 0.35·S_citation + 0.25·S_BM25
```
- Combines embedding similarity, citation proximity, and lexical matching
- Returns top-k=5 precedents for each query case

### 4. QLoRA Fine-tuning
- Base: Mistral-7B-Instruct-v0.3
- Quantization: 4-bit NF4
- LoRA: r=16, α=32
- Trainable parameters: ~7M (0.1%)

## Results

| Model | AUROC | Accuracy | F1 |
|-------|-------|----------|-----|
| Legal-BERT | 0.71 | 68.7% | 0.65 |
| Longformer | 0.73 | 70.8% | 0.68 |
| Mistral (no retrieval) | 0.74 | 71.2% | 0.68 |
| **LegalGPT (Ours)** | **0.80** | **76.0%** | **0.75** |

### Ablation Study
| Configuration | AUROC | Δ |
|--------------|-------|-----|
| Full system | 0.80 | — |
| − Graph retrieval | 0.77 | -0.03 |
| − Citation proximity | 0.78 | -0.02 |
| − QLoRA (frozen LLM) | 0.72 | -0.08 |

## Live Demo

Visit [caselaw-graph-ring.vercel.app](https://caselaw-graph-ring.vercel.app) to:
- Explore the methodology with detailed visualizations
- View comprehensive results and ablation studies
- Try the interactive case prediction demo

## Data Sources

- **Supreme Court Database (SCDB)**: Case metadata and outcomes (1946-2023)
- **CourtListener**: Full opinion text via API
- **Citation extraction**: Regex-based parsing of US Reports citations

## Requirements

```
torch>=2.1.0
transformers>=4.36.0
peft>=0.7.0
bitsandbytes>=0.41.3
torch-geometric>=2.4.0
sentence-transformers>=2.2.2
neo4j>=5.14.0
flask>=2.3.0
```

## Citation

If you use this work, please cite:

```bibtex
@inproceedings{sanchez2026legalgpt,
  title={LegalGPT: Graph-Augmented Legal Outcome Prediction using Citation Networks and Large Language Models},
  author={Sanchez, Luis and Tripathy, Shubhankar},
  booktitle={Proceedings of EMNLP 2026},
  year={2026}
}
```

## Authors

- **Luis Sanchez** - UC Berkeley, Computer Science ([suislanchez.com](https://suislanchez.com))
- **Shubhankar Tripathy** - Stanford PhD, OpenAI Research

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Supreme Court Database (SCDB) at Washington University in St. Louis
- CourtListener / Free Law Project for case text access
- Modal Labs for compute infrastructure
- HuggingFace for model hosting

---

<p align="center">
  <strong>EMNLP 2026 Submission</strong><br>
  <a href="https://caselaw-graph-ring.vercel.app">Live Demo</a> •
  <a href="https://caselaw-graph-ring.vercel.app/methodology">Methodology</a> •
  <a href="https://caselaw-graph-ring.vercel.app/results">Results</a>
</p>
