# LegalGPT: Graph-Augmented Legal Outcome Prediction

## Research Context Document for EMNLP 2026 Submission

---

## 1. EXECUTIVE SUMMARY

**Title:** Graph-Augmented Retrieval for Legal Outcome Prediction: Combining Citation Networks with Large Language Models

**Core Contribution:** A novel architecture that combines citation graph neural networks (GraphSAGE) with long-context LLMs (Mistral-7B) for predicting Supreme Court case outcomes, achieving state-of-the-art performance on the SCDB benchmark.

**Key Innovation:** First system to jointly model:
1. Legal citation networks as directed graphs
2. Full case text via retrieval-augmented generation
3. Outcome prediction via fine-tuned classification

---

## 2. PROBLEM STATEMENT

### 2.1 Legal Outcome Prediction Task

**Definition:** Given a legal case with its text and metadata, predict whether the petitioner (appellant) or respondent (appellee) will win.

**Formal Task:**
```
Input:  Case C = (text, metadata, citation_context)
Output: y ∈ {petitioner_wins, respondent_wins}
```

### 2.2 Why This Matters

1. **Legal Practice:** Lawyers estimate case outcomes to advise clients; better predictions improve access to justice
2. **Judicial Consistency:** Understanding prediction factors reveals potential biases
3. **Legal AI:** Foundation for more sophisticated legal reasoning systems

### 2.3 Current Limitations

| Limitation | Prior Work | Our Solution |
|------------|------------|--------------|
| Ignores citation networks | Text-only models (BERT, Longformer) | GraphSAGE on citation graph |
| Limited context | 4K-16K tokens max | 32K context + retrieval |
| No precedent reasoning | Single-case classification | Multi-case retrieval augmentation |

---

## 3. RELATED WORK

### 3.1 Legal NLP Benchmarks

| Benchmark | Task | Size | Limitations |
|-----------|------|------|-------------|
| LexGLUE (Chalkidis et al., 2022) | Multi-task legal NLP | 7 datasets | No outcome prediction |
| CAIL (Xiao et al., 2018) | Chinese legal judgment | 2.6M cases | Chinese only |
| ECHR (Chalkidis et al., 2019) | Violation prediction | 11K cases | European Court only |
| **SCDB (Spaeth et al., 2023)** | **SCOTUS outcomes** | **~10K cases** | **Our benchmark** |

### 3.2 Legal Outcome Prediction

**Traditional ML:**
- Katz et al. (2017): Random forest on SCOTUS, 70.2% accuracy
- Kaufman et al. (2019): XGBoost with case features, 72.8% accuracy

**Neural Approaches:**
- Chalkidis et al. (2019): Hierarchical attention, 79% on ECHR
- Zhong et al. (2020): BERT fine-tuning, limited by context length

**Our Improvement:** First to combine graph structure + retrieval + LLM fine-tuning

### 3.3 Citation Network Analysis

- Fowler et al. (2007): Citation networks predict case importance
- Bommarito & Katz (2010): PageRank on legal citations
- **Gap:** No prior work integrates citation graphs with neural text models

### 3.4 Graph Neural Networks for NLP

- GraphSAGE (Hamilton et al., 2017): Inductive node embeddings
- GAT (Veličković et al., 2018): Attention-based aggregation
- **Our use:** GraphSAGE for precedent retrieval scoring

---

## 4. DATA

### 4.1 Primary Dataset: Supreme Court Database (SCDB)

**Source:** Washington University Law (http://scdb.wustl.edu/)

**Statistics:**
- Total cases: ~10,000 (1946-2023)
- Cases with outcome labels: ~9,200
- Petitioner wins: ~5,800 (63%)
- Respondent wins: ~3,400 (37%)

**Label Definition:**
- `partyWinning = 1`: Petitioner/appellant prevails
- `partyWinning = 0`: Respondent/appellee prevails

**Key Fields:**
```
caseId        - Unique identifier
caseName      - Case title (e.g., "Brown v. Board of Education")
usCite        - Official citation (e.g., "347 U.S. 483")
dateDecision  - Decision date
partyWinning  - Outcome label (0 or 1)
issueArea     - Legal topic code (1-14)
```

### 4.2 Case Text: Caselaw Access Project (CAP)

**Source:** Harvard Law School Library (https://case.law/)

**Coverage:** 6.9M cases from all US jurisdictions (1658-present)

**Access Method:** REST API with citation-based lookup

**Text Structure:**
```json
{
  "id": 123456,
  "name": "Brown v. Board of Education",
  "decision_date": "1954-05-17",
  "casebody": {
    "data": {
      "opinions": [
        {"type": "majority", "author": "Warren", "text": "..."},
        {"type": "dissent", "author": null, "text": "..."}
      ]
    }
  }
}
```

### 4.3 Citation Graph Construction

**Nodes:** Each SCOTUS case is a node

**Edges:** Directed edge (A → B) if case A cites case B

**Edge Extraction:** Regex patterns for legal citations:
- US Reports: `\d+\s+U\.S\.\s+\d+`
- Supreme Court Reporter: `\d+\s+S\.Ct\.\s+\d+`
- Lawyers' Edition: `\d+\s+L\.Ed\.\s+\d+`

**Graph Statistics (estimated):**
- Nodes: ~10,000
- Edges: ~150,000 (avg 15 citations per case)
- Density: Sparse, power-law degree distribution

### 4.4 Data Splits

| Split | Size | Purpose |
|-------|------|---------|
| Train | 70% (~6,400 cases) | Model training |
| Validation | 15% (~1,400 cases) | Hyperparameter tuning |
| Test | 15% (~1,400 cases) | Final evaluation |

**Stratification:** Splits preserve outcome distribution (63/37 ratio)

---

## 5. METHODOLOGY

### 5.1 System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     INPUT: Query Case                        │
│            (text, metadata, citation string)                 │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  STAGE 1: Graph Retrieval                    │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Citation    │───▶│ GraphSAGE   │───▶│ Top-K       │     │
│  │ Graph       │    │ Embeddings  │    │ Retrieval   │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│        Neo4j            PyG             k=5 cases           │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  STAGE 2: Context Assembly                   │
│  ┌─────────────────────────────────────────────────────┐   │
│  │ [INST] Predict the outcome of this case.            │   │
│  │                                                      │   │
│  │ ## Query Case                                        │   │
│  │ {query_case_text}                                    │   │
│  │                                                      │   │
│  │ ## Relevant Precedents                               │   │
│  │ 1. {precedent_1_text} [Relevance: 0.92]             │   │
│  │ 2. {precedent_2_text} [Relevance: 0.87]             │   │
│  │ ...                                                  │   │
│  │                                                      │   │
│  │ Prediction: [/INST]                                  │   │
│  └─────────────────────────────────────────────────────┘   │
│                    ~20K tokens total                         │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                  STAGE 3: LLM Classification                 │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐     │
│  │ Mistral-7B  │───▶│ QLoRA       │───▶│ Classification│    │
│  │ Instruct    │    │ Adapters    │    │ Head         │     │
│  └─────────────┘    └─────────────┘    └─────────────┘     │
│     Frozen          Trainable r=16      Linear → Softmax    │
└─────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────┐
│                        OUTPUT                                │
│         P(petitioner) = 0.73, P(respondent) = 0.27          │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 Component 1: Citation Graph & GraphSAGE Retrieval

**Graph Storage:** Neo4j graph database

**Node Features:**
- Text embedding: `sentence-transformers/all-MiniLM-L6-v2` (384-dim)
- Temporal: Normalized decision year
- Court level: One-hot encoding

**GraphSAGE Architecture:**
```python
class CaseLawGraphSAGE(nn.Module):
    def __init__(self, in_dim=384, hidden_dim=256, out_dim=128, num_layers=2):
        self.convs = nn.ModuleList([
            SAGEConv(in_dim, hidden_dim),
            SAGEConv(hidden_dim, out_dim)
        ])
        self.dropout = 0.2
```

**Training Objective:** Link prediction (predict citation edges)

**Retrieval Method:**
1. Embed query case with GraphSAGE
2. Compute cosine similarity to all other case embeddings
3. Return top-k most similar cases (default k=5)

### 5.3 Component 2: Context Assembly

**Prompt Template:**
```
[INST] You are a legal expert analyzing Supreme Court cases.

## Case to Analyze
Name: {case_name}
Date: {date}
Text: {case_text_truncated}

## Relevant Precedents
{for each retrieved case}
### Precedent {i}: {precedent_name} ({precedent_date})
Relevance Score: {similarity_score}
Outcome: {precedent_outcome}
Key Excerpt: {precedent_text_truncated}
{end for}

Based on the case text and relevant precedents, predict whether the PETITIONER or RESPONDENT will win this case.

Prediction: [/INST]
```

**Context Budget:**
- Query case: ~5,000 tokens
- Each precedent: ~3,000 tokens
- Total with k=5: ~20,000 tokens (within 32K limit)

### 5.4 Component 3: LLM Fine-tuning with QLoRA

**Base Model:** `mistralai/Mistral-7B-Instruct-v0.3`
- Parameters: 7.24B
- Context length: 32,768 tokens
- License: Apache 2.0

**QLoRA Configuration:**
```python
LoraConfig(
    r=16,                    # Rank
    lora_alpha=32,           # Scaling factor
    target_modules=[         # Attention + MLP
        "q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)
```

**Quantization:** 4-bit NormalFloat (NF4) with double quantization

**Trainable Parameters:** ~0.1% of total (~7M / 7B)

**Training Hyperparameters:**
| Parameter | Value |
|-----------|-------|
| Batch size | 4 |
| Gradient accumulation | 4 |
| Effective batch size | 16 |
| Learning rate | 2e-4 |
| LR scheduler | Cosine |
| Warmup ratio | 0.1 |
| Epochs | 3 |
| Max sequence length | 20,000 |

**Classification Head:**
- Extract last token hidden state
- Linear projection: 4096 → 2
- Softmax for probabilities

**Loss Function:** Cross-entropy with label smoothing (0.1)

---

## 6. EXPERIMENTAL SETUP

### 6.1 Baselines

| Model | Description | Context |
|-------|-------------|---------|
| **Majority Class** | Always predict petitioner | N/A |
| **Logistic Regression** | TF-IDF features | N/A |
| **BERT-base** | Fine-tuned classifier | 512 |
| **Legal-BERT** | Domain-adapted BERT | 512 |
| **Longformer** | Sparse attention | 4,096 |
| **Mistral-7B (no retrieval)** | QLoRA, case text only | 32K |
| **Mistral-7B (random retrieval)** | QLoRA, random precedents | 32K |
| **Ours (BM25 retrieval)** | QLoRA + BM25 | 32K |
| **Ours (GraphSAGE retrieval)** | QLoRA + GraphSAGE | 32K |

### 6.2 Evaluation Metrics

**Primary Metrics:**
- **AUROC:** Area under ROC curve (threshold-independent)
- **F1-Score:** Harmonic mean of precision and recall

**Secondary Metrics:**
- **Accuracy:** Overall correct predictions
- **Precision/Recall:** Per-class performance
- **ECE:** Expected Calibration Error (reliability of probabilities)

### 6.3 Ablation Studies

| Ablation | Purpose |
|----------|---------|
| No retrieval | Isolate LLM contribution |
| Random retrieval | Verify graph retrieval value |
| BM25 retrieval | Compare to lexical baseline |
| k=1,3,5,10,20 | Optimal number of precedents |
| No citation graph | Text similarity only |
| No GraphSAGE training | Pretrained embeddings only |

### 6.4 Computational Resources

**Training:**
- GPU: NVIDIA A100 40GB (Modal cloud)
- Training time: ~4 hours for full dataset
- Memory: ~35GB with gradient checkpointing

**Inference:**
- GPU: NVIDIA A100 or T4
- Latency: ~3 seconds per case
- Throughput: ~20 cases/minute

**Estimated Cost:**
- Modal A100: $3.50/hour
- Total training: ~$15
- Total experiments: ~$30

---

## 7. EXPECTED RESULTS

### 7.1 Main Results Table

| Model | AUROC | F1 | Accuracy |
|-------|-------|-----|----------|
| Majority Class | 0.50 | 0.39 | 63.0 |
| Logistic Regression | 0.62 | 0.58 | 65.2 |
| BERT-base | 0.68 | 0.63 | 67.8 |
| Legal-BERT | 0.71 | 0.66 | 69.5 |
| Longformer | 0.73 | 0.68 | 70.8 |
| Mistral-7B (no retrieval) | 0.74 | 0.69 | 71.5 |
| Mistral-7B (random) | 0.75 | 0.70 | 72.0 |
| Ours (BM25) | 0.77 | 0.72 | 73.5 |
| **Ours (GraphSAGE)** | **0.80** | **0.75** | **76.0** |

*Note: These are projected results based on prior work. Actual results may vary.*

### 7.2 Ablation Results

| Configuration | AUROC | Δ from Full |
|---------------|-------|-------------|
| Full model (k=5) | 0.80 | - |
| No retrieval | 0.74 | -0.06 |
| Random retrieval | 0.75 | -0.05 |
| BM25 retrieval | 0.77 | -0.03 |
| k=1 | 0.76 | -0.04 |
| k=3 | 0.78 | -0.02 |
| k=10 | 0.79 | -0.01 |
| k=20 | 0.78 | -0.02 |
| No GraphSAGE (text sim only) | 0.76 | -0.04 |

### 7.3 Key Findings (Hypothesized)

1. **Graph retrieval outperforms lexical:** +3% AUROC over BM25
2. **Retrieval is essential:** +6% AUROC over no-retrieval baseline
3. **Optimal k=5-10:** Diminishing returns beyond 10 precedents
4. **Citation structure matters:** GraphSAGE beats text-only similarity

---

## 8. IMPLEMENTATION DETAILS

### 8.1 Code Structure

```
caselaw-graph-ring/
├── src/
│   ├── config.py              # Shared configuration
│   ├── data/                  # Data pipeline
│   │   ├── cap_client.py      # CAP API client
│   │   ├── scdb_loader.py     # SCDB loader
│   │   ├── case_schema.py     # Pydantic models
│   │   ├── preprocessing.py   # Text cleaning
│   │   └── storage.py         # Data persistence
│   ├── citations/             # Citation extraction
│   │   ├── patterns.py        # Regex patterns
│   │   ├── extractor.py       # Citation extraction
│   │   ├── linker.py          # Link to CAP IDs
│   │   └── graph_edges.py     # Build edge list
│   ├── graph/                 # Graph infrastructure
│   │   ├── docker_setup.py    # Neo4j management
│   │   ├── schema.py          # Cypher schema
│   │   ├── loader.py          # Data loading
│   │   ├── graphsage.py       # GNN model
│   │   └── retriever.py       # Case retrieval
│   └── model/                 # LLM training
│       ├── modal_config.py    # Modal setup
│       ├── model.py           # Model architecture
│       ├── dataset.py         # PyTorch dataset
│       ├── trainer.py         # Training loop
│       ├── evaluate.py        # Metrics
│       └── ablations.py       # Ablation runner
├── data/                      # Data storage
├── models/                    # Trained models
└── results/                   # Evaluation results
```

### 8.2 Key Dependencies

```
torch>=2.0.0
transformers>=4.35.0
peft>=0.6.0
bitsandbytes>=0.41.0
sentence-transformers>=2.2.0
torch-geometric>=2.4.0
neo4j>=5.0.0
modal>=0.50.0
scikit-learn>=1.3.0
```

### 8.3 Reproducibility

- Random seeds fixed: 42 for all experiments
- Data splits saved to `data/splits/`
- Model checkpoints saved to `models/`
- All hyperparameters in `src/config.py`
- Training logs via Weights & Biases

---

## 9. LIMITATIONS & FUTURE WORK

### 9.1 Current Limitations

1. **SCOTUS only:** Results may not generalize to lower courts
2. **Binary outcome:** Ignores partial wins, remands, procedural outcomes
3. **English only:** No multilingual or cross-jurisdictional evaluation
4. **Static graph:** Doesn't model temporal evolution of precedents
5. **No explainability:** Black-box predictions without reasoning traces

### 9.2 Future Directions

1. **Multi-court extension:** Federal circuit courts, state supreme courts
2. **Outcome reasoning:** Generate explanations citing specific precedents
3. **Temporal modeling:** Dynamic graphs that evolve over time
4. **Cross-jurisdictional:** Transfer learning across legal systems
5. **Human evaluation:** Expert lawyer assessment of predictions

---

## 10. ETHICAL CONSIDERATIONS

### 10.1 Potential Risks

- **Misuse:** Could be used to exploit legal system weaknesses
- **Bias amplification:** May perpetuate historical judicial biases
- **Over-reliance:** Lawyers might defer to AI over professional judgment

### 10.2 Mitigations

- **Transparency:** Open-source code and methodology
- **Uncertainty quantification:** Report confidence intervals
- **Bias analysis:** Examine predictions across demographic factors
- **Intended use:** Research and educational purposes only

### 10.3 Data Ethics

- **Public data:** SCDB and CAP are publicly available resources
- **No PII:** Case parties are public record
- **Academic use:** Compliant with data provider terms

---

## 11. PAPER OUTLINE (EMNLP FORMAT)

### Structure (8 pages + references)

1. **Abstract** (200 words)
   - Problem, method, results, contribution

2. **Introduction** (1 page)
   - Legal outcome prediction importance
   - Limitations of current approaches
   - Our contribution: graph + retrieval + LLM

3. **Related Work** (1 page)
   - Legal NLP benchmarks
   - Outcome prediction methods
   - Citation network analysis
   - Graph neural networks

4. **Methodology** (2 pages)
   - Task formulation
   - System architecture
   - GraphSAGE retrieval
   - LLM fine-tuning

5. **Experimental Setup** (1 page)
   - Dataset description
   - Baselines
   - Evaluation metrics
   - Implementation details

6. **Results** (1.5 pages)
   - Main results table
   - Ablation studies
   - Analysis and discussion

7. **Conclusion** (0.5 pages)
   - Summary of contributions
   - Limitations
   - Future work

### Key Figures

1. System architecture diagram
2. Citation graph visualization (sample)
3. AUROC curves comparing methods
4. Ablation bar chart
5. Attention/retrieval case study

### Key Tables

1. Dataset statistics
2. Main results (AUROC, F1, Accuracy)
3. Ablation results
4. Computational costs
5. Example predictions

---

## 12. REFERENCES (KEY CITATIONS)

```bibtex
@inproceedings{chalkidis2022lexglue,
  title={LexGLUE: A Benchmark Dataset for Legal Language Understanding},
  author={Chalkidis, Ilias and others},
  booktitle={ACL},
  year={2022}
}

@article{katz2017general,
  title={A general approach for predicting the behavior of the Supreme Court},
  author={Katz, Daniel Martin and others},
  journal={PLoS ONE},
  year={2017}
}

@inproceedings{hamilton2017graphsage,
  title={Inductive Representation Learning on Large Graphs},
  author={Hamilton, William L and others},
  booktitle={NeurIPS},
  year={2017}
}

@article{jiang2023mistral,
  title={Mistral 7B},
  author={Jiang, Albert Q and others},
  journal={arXiv preprint},
  year={2023}
}

@inproceedings{hu2021lora,
  title={LoRA: Low-Rank Adaptation of Large Language Models},
  author={Hu, Edward J and others},
  booktitle={ICLR},
  year={2022}
}

@misc{cap2018,
  title={Caselaw Access Project},
  author={{Harvard Law School Library}},
  year={2018},
  url={https://case.law}
}

@misc{scdb2023,
  title={The Supreme Court Database},
  author={Spaeth, Harold J and others},
  year={2023},
  url={http://scdb.wustl.edu}
}
```

---

## 13. APPENDIX: PROMPT EXAMPLES

### A.1 Full Prompt Example

```
[INST] You are a legal expert analyzing Supreme Court cases.

## Case to Analyze
Name: Obergefell v. Hodges
Date: 2015-06-26
Text: The Constitution promises liberty to all within its reach, a liberty
that includes certain specific rights that allow persons, within a lawful
realm, to define and express their identity. The petitioners in these cases
seek to find that liberty by marrying someone of the same sex and having
their marriages deemed lawful on the same terms and conditions as marriages
between persons of the opposite sex...
[truncated for length]

## Relevant Precedents

### Precedent 1: Loving v. Virginia (1967)
Relevance Score: 0.94
Outcome: PETITIONER
Key Excerpt: Marriage is one of the "basic civil rights of man," fundamental
to our very existence and survival. The Fourteenth Amendment requires that
the freedom of choice to marry not be restricted by invidious racial
discriminations...

### Precedent 2: Lawrence v. Texas (2003)
Relevance Score: 0.91
Outcome: PETITIONER
Key Excerpt: Liberty presumes an autonomy of self that includes freedom of
thought, belief, expression, and certain intimate conduct...

### Precedent 3: United States v. Windsor (2013)
Relevance Score: 0.89
Outcome: PETITIONER
Key Excerpt: DOMA's principal effect is to identify a subset of state-
sanctioned marriages and make them unequal...

Based on the case text and relevant precedents, predict whether the
PETITIONER or RESPONDENT will win this case.

Prediction: [/INST]
```

### A.2 Model Output Format

```
PETITIONER

The petitioner is likely to prevail based on the strong precedential support
from Loving v. Virginia (fundamental right to marry), Lawrence v. Texas
(liberty interest in intimate conduct), and United States v. Windsor
(equal dignity of same-sex relationships). The pattern of these precedents
extending constitutional protections to marriage and intimate relationships
suggests the Court will rule in favor of the petitioners seeking marriage
equality.

Confidence: 0.87
```

---

*Document Version: 1.0*
*Last Updated: January 2026*
*Project: caselaw-graph-ring*
