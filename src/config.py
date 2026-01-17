"""Shared configuration for LegalGPT project."""

import os
from pathlib import Path

# Base directories
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# Data subdirectories
RAW_DATA_DIR = DATA_DIR / "raw"
RAW_DIR = RAW_DATA_DIR  # Alias for backward compatibility
PROCESSED_DATA_DIR = DATA_DIR / "processed"
PROCESSED_DIR = PROCESSED_DATA_DIR  # Alias for backward compatibility
CITATIONS_DATA_DIR = DATA_DIR / "citations"
EMBEDDINGS_DIR = DATA_DIR / "embeddings"
SPLITS_DIR = DATA_DIR / "splits"

# Cases directory
CASES_DIR = RAW_DATA_DIR / "cases"

# Neo4j configuration
NEO4J_URI = os.environ.get("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.environ.get("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.environ.get("NEO4J_PASSWORD", "legalgpt123")

# Caselaw Access Project (CAP) API configuration
CAP_API_BASE = "https://api.case.law/v1"
CAP_API_KEY = os.environ.get("CAP_API_KEY")

# Supreme Court Database (SCDB) configuration
SCDB_URL = "http://scdb.wustl.edu/_brickFiles/2023_01/SCDB_2023_01_caseCentered_Citation.csv.zip"

# Hugging Face configuration
HF_TOKEN = os.environ.get("HF_TOKEN")

# Model configuration
MODEL_NAME = "mistralai/Mistral-7B-Instruct-v0.3"
MAX_CONTEXT_LENGTH = 32768
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_DIM = 384

# Training configuration
TRAIN_BATCH_SIZE = 4
EVAL_BATCH_SIZE = 8
LEARNING_RATE = 2e-4
NUM_EPOCHS = 3
WARMUP_RATIO = 0.1
GRADIENT_ACCUMULATION_STEPS = 4

# QLoRA configuration
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]

# Retrieval configuration
DEFAULT_RETRIEVAL_K = 5
DEFAULT_TOP_K = DEFAULT_RETRIEVAL_K  # Alias
TOP_K_RETRIEVAL = DEFAULT_RETRIEVAL_K  # Alias
MAX_RETRIEVAL_K = 20

# GraphSAGE configuration
GRAPHSAGE_HIDDEN_DIM = 256
GRAPHSAGE_NUM_LAYERS = 2

# Data splits
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15

# Aliases for backward compatibility
TRAIN_SPLIT = TRAIN_RATIO
VAL_SPLIT = VAL_RATIO
TEST_SPLIT = TEST_RATIO
CITATIONS_DIR = CITATIONS_DATA_DIR

# Ensure directories exist
for dir_path in [DATA_DIR, MODELS_DIR, RESULTS_DIR, RAW_DATA_DIR,
                 PROCESSED_DATA_DIR, CITATIONS_DATA_DIR, EMBEDDINGS_DIR, SPLITS_DIR, CASES_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)
