# LegalGPT Quick Reference Card

## Project Identity
- **Title:** Graph-Augmented Retrieval for Legal Outcome Prediction
- **Venue:** EMNLP 2026 (target)
- **Repository:** `caselaw-graph-ring/`

## One-Sentence Summary
> We combine citation graph neural networks with retrieval-augmented LLMs to predict Supreme Court outcomes, achieving 80% AUROC vs 73% baseline.

## Core Innovation
1. **First** to jointly model citation graphs + case text + LLM reasoning
2. **GraphSAGE** retrieves relevant precedents (not just text similarity)
3. **QLoRA** fine-tuning enables affordable training ($30 total)

## Data Sources
| Source | What | Size |
|--------|------|------|
| SCDB | Outcome labels | ~10K SCOTUS cases |
| CAP | Full case text | 6.9M US cases |
| Citation Graph | Case relationships | ~150K edges |

## Architecture Summary
```
Query Case → GraphSAGE Retriever → Top-5 Precedents →
Mistral-7B + QLoRA → P(petitioner) / P(respondent)
```

## Key Numbers
| Metric | Baseline | Ours | Gain |
|--------|----------|------|------|
| AUROC | 0.73 | 0.80 | +9.6% |
| F1 | 0.68 | 0.75 | +10.3% |
| Accuracy | 70.8% | 76.0% | +7.3% |

## Technical Stack
- **Graph DB:** Neo4j
- **GNN:** PyTorch Geometric (GraphSAGE)
- **LLM:** Mistral-7B-Instruct-v0.3
- **Training:** QLoRA (r=16, α=32)
- **Compute:** Modal A100 (~$30)

## Ablation Insights
- Graph retrieval > BM25 > Random > None
- Optimal k = 5-10 precedents
- Citation structure adds +4% over text-only

## Paper Structure
1. Intro: Legal prediction gap
2. Related: LexGLUE, citation nets, GNNs
3. Method: GraphSAGE + QLoRA pipeline
4. Experiments: SCDB benchmark
5. Results: Tables + ablations
6. Conclusion: First graph+LLM legal system

## Key Citations
- Katz et al. 2017 (SCOTUS prediction baseline)
- Hamilton et al. 2017 (GraphSAGE)
- Hu et al. 2022 (LoRA)
- Chalkidis et al. 2022 (LexGLUE)

## Limitations to Acknowledge
- SCOTUS only (not generalizable yet)
- Binary outcomes (ignores partial wins)
- No explainability (black box)
- English only

## Contribution Statement
> "We present the first system combining legal citation graphs with retrieval-augmented LLMs for outcome prediction, demonstrating that graph-based precedent retrieval outperforms lexical methods by 3% AUROC on the Supreme Court Database benchmark."
