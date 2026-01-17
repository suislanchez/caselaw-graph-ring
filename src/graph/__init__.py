"""Graph infrastructure module for LegalGPT citation network.

This module provides:
- Neo4j Docker container management
- Graph schema and data loading
- GraphSAGE model for learning citation embeddings
- Case retrieval API combining embeddings and graph proximity
- Visualization utilities

Main interface for other agents:
    from src.graph.retriever import get_similar_cases, CaseRetriever, RetrievedCase

Output:
- Neo4j running on localhost:7474/7687
- data/embeddings/graphsage_embeddings.pt
- retriever.get_similar_cases() function ready for Agent 4
"""

from src.graph.docker_setup import (
    start_neo4j,
    stop_neo4j,
    is_running,
    is_container_running,
    wait_for_ready,
    wait_for_neo4j,
    check_health,
    get_connection,
    get_stats,
    get_container_logs,
)
from src.graph.schema import (
    create_schema,
    drop_schema,
    clear_all_data,
    get_schema_info,
    SCHEMA_STATEMENTS,
    SCHEMA_CYPHER,
)
from src.graph.loader import (
    load_cases_to_neo4j,
    load_edges_to_neo4j,
    load_all,
    update_embeddings,
    generate_embeddings,
)
from src.graph.graphsage import (
    GraphSAGE,
    GraphSAGETrainer,
    LinkPredictor,
    train_graphsage,
    load_graph_from_neo4j,
    save_embeddings,
    load_embeddings,
    build_graph_data,
)
from src.graph.retriever import (
    CaseRetriever,
    RetrievedCase,
    get_similar_cases,
)
from src.graph.visualize import (
    load_subgraph_from_neo4j,
    visualize_subgraph,
    visualize_citation_graph_matplotlib,
    visualize_citation_graph_plotly,
    visualize_embedding_space,
    export_to_gephi,
    get_graph_statistics,
)

__all__ = [
    # Docker management
    "start_neo4j",
    "stop_neo4j",
    "is_running",
    "is_container_running",
    "wait_for_ready",
    "wait_for_neo4j",
    "check_health",
    "get_connection",
    "get_stats",
    "get_container_logs",
    # Schema
    "create_schema",
    "drop_schema",
    "clear_all_data",
    "get_schema_info",
    "SCHEMA_STATEMENTS",
    "SCHEMA_CYPHER",
    # Loading
    "load_cases_to_neo4j",
    "load_edges_to_neo4j",
    "load_all",
    "update_embeddings",
    "generate_embeddings",
    # GraphSAGE
    "GraphSAGE",
    "GraphSAGETrainer",
    "LinkPredictor",
    "train_graphsage",
    "load_graph_from_neo4j",
    "save_embeddings",
    "load_embeddings",
    "build_graph_data",
    # Retrieval (main interface for Agent 4)
    "CaseRetriever",
    "RetrievedCase",
    "get_similar_cases",
    # Visualization
    "load_subgraph_from_neo4j",
    "visualize_subgraph",
    "visualize_citation_graph_matplotlib",
    "visualize_citation_graph_plotly",
    "visualize_embedding_space",
    "export_to_gephi",
    "get_graph_statistics",
]
