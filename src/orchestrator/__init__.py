"""
LegalGPT 4-Agent Orchestration System.

Coordinates parallel execution of:
- Agent A: Citation extraction and linking
- Agent B: Graph infrastructure (Neo4j + GraphSAGE)
- Agent C: Model training (Modal A100)
- Agent D: Evaluation and ablations
"""

from .orchestrator import PipelineOrchestrator
from .status import StatusManager, AgentStatus, PipelineStatus

__all__ = [
    "PipelineOrchestrator",
    "StatusManager",
    "AgentStatus",
    "PipelineStatus",
]
