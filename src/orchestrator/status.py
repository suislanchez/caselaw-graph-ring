"""
Status management for pipeline orchestration.

Provides thread-safe JSON-based status tracking with file locking
for real-time website updates via SSE.
"""

import json
import fcntl
import asyncio
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Any, Optional, Callable
import threading


class AgentState(str, Enum):
    """Agent execution states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class PipelineState(str, Enum):
    """Pipeline execution states."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    PAUSED = "paused"


@dataclass
class StepStatus:
    """Status of a single step within an agent."""
    name: str
    status: AgentState = AgentState.PENDING
    progress: int = 0
    message: str = ""
    details: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[str] = None
    completed_at: Optional[str] = None


@dataclass
class AgentStatus:
    """Status of a single agent."""
    id: str
    name: str
    description: str
    status: AgentState = AgentState.PENDING
    progress: int = 0
    current_step: str = ""
    steps: Dict[str, StepStatus] = field(default_factory=dict)
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    logs: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "status": self.status.value if isinstance(self.status, AgentState) else self.status,
            "progress": self.progress,
            "current_step": self.current_step,
            "steps": {
                k: {
                    "name": v.name,
                    "status": v.status.value if isinstance(v.status, AgentState) else v.status,
                    "progress": v.progress,
                    "message": v.message,
                    "details": v.details,
                    "started_at": v.started_at,
                    "completed_at": v.completed_at,
                } for k, v in self.steps.items()
            },
            "metrics": self.metrics,
            "error": self.error,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "logs": self.logs[-50:],  # Keep last 50 logs
        }


@dataclass
class PipelineStatus:
    """Overall pipeline status."""
    id: str
    status: PipelineState = PipelineState.IDLE
    started_at: Optional[str] = None
    completed_at: Optional[str] = None
    error: Optional[str] = None
    agents: Dict[str, AgentStatus] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "pipeline": {
                "id": self.id,
                "status": self.status.value if isinstance(self.status, PipelineState) else self.status,
                "started_at": self.started_at,
                "completed_at": self.completed_at,
                "error": self.error,
            },
            "agents": {k: v.to_dict() for k, v in self.agents.items()},
            "last_updated": datetime.now().isoformat(),
        }


# Default agent definitions
DEFAULT_AGENTS = [
    AgentStatus(
        id="citations",
        name="Citation Extraction",
        description="Extract citations from case text, link to CourtListener, build graph edges",
        steps={
            "loading_cases": StepStatus(name="Loading cases from storage"),
            "extracting_citations": StepStatus(name="Extracting citations from text"),
            "linking_citations": StepStatus(name="Linking citations to case IDs"),
            "building_edges": StepStatus(name="Building citation graph edges"),
        }
    ),
    AgentStatus(
        id="graph",
        name="Graph Infrastructure",
        description="Load data to Neo4j, train GraphSAGE embeddings, setup retriever",
        steps={
            "loading_neo4j": StepStatus(name="Loading cases to Neo4j"),
            "generating_embeddings": StepStatus(name="Generating text embeddings"),
            "training_graphsage": StepStatus(name="Training GraphSAGE model"),
            "exporting_embeddings": StepStatus(name="Exporting node embeddings"),
        }
    ),
    AgentStatus(
        id="model",
        name="Model Training",
        description="Train Mistral-7B with QLoRA on Modal A100",
        steps={
            "preparing_data": StepStatus(name="Preparing training data with retrieval"),
            "uploading_modal": StepStatus(name="Uploading data to Modal"),
            "training_qlora": StepStatus(name="Training QLoRA adapters"),
            "downloading_model": StepStatus(name="Downloading trained model"),
        }
    ),
    AgentStatus(
        id="evaluation",
        name="Evaluation & Results",
        description="Compute metrics, run ablations, generate paper results",
        steps={
            "running_predictions": StepStatus(name="Running model predictions"),
            "computing_metrics": StepStatus(name="Computing evaluation metrics"),
            "running_ablations": StepStatus(name="Running ablation studies"),
            "generating_results": StepStatus(name="Generating paper results"),
        }
    ),
]


class StatusManager:
    """
    Thread-safe status manager with file-based persistence.

    Uses file locking for concurrent access and provides
    callbacks for real-time status updates to website.
    """

    def __init__(self, status_path: Path, auto_save: bool = True):
        """
        Initialize status manager.

        Args:
            status_path: Path to status JSON file
            auto_save: Whether to auto-save on updates
        """
        self.status_path = Path(status_path)
        self.auto_save = auto_save
        self._lock = threading.RLock()
        self._subscribers: List[Callable[[Dict], None]] = []
        self._status: Optional[PipelineStatus] = None

        # Ensure directory exists
        self.status_path.parent.mkdir(parents=True, exist_ok=True)

        # Load or initialize status
        self._load_or_initialize()

    def _load_or_initialize(self):
        """Load existing status or initialize new one."""
        if self.status_path.exists():
            try:
                self._status = self._load_from_file()
            except (json.JSONDecodeError, KeyError):
                self._status = self._create_initial_status()
        else:
            self._status = self._create_initial_status()
            self._save_to_file()

    def _create_initial_status(self) -> PipelineStatus:
        """Create initial pipeline status with default agents."""
        pipeline_id = f"pipeline-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        return PipelineStatus(
            id=pipeline_id,
            agents={agent.id: agent for agent in DEFAULT_AGENTS}
        )

    def _load_from_file(self) -> PipelineStatus:
        """Load status from JSON file with file locking."""
        with open(self.status_path, 'r') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_SH)
            try:
                data = json.load(f)
                return self._dict_to_status(data)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _save_to_file(self):
        """Save status to JSON file with file locking."""
        with open(self.status_path, 'w') as f:
            fcntl.flock(f.fileno(), fcntl.LOCK_EX)
            try:
                json.dump(self._status.to_dict(), f, indent=2)
            finally:
                fcntl.flock(f.fileno(), fcntl.LOCK_UN)

    def _dict_to_status(self, data: Dict) -> PipelineStatus:
        """Convert dictionary to PipelineStatus object."""
        pipeline_data = data.get("pipeline", {})
        agents_data = data.get("agents", {})

        agents = {}
        for agent_id, agent_data in agents_data.items():
            steps = {}
            for step_id, step_data in agent_data.get("steps", {}).items():
                steps[step_id] = StepStatus(
                    name=step_data.get("name", step_id),
                    status=AgentState(step_data.get("status", "pending")),
                    progress=step_data.get("progress", 0),
                    message=step_data.get("message", ""),
                    details=step_data.get("details", {}),
                    started_at=step_data.get("started_at"),
                    completed_at=step_data.get("completed_at"),
                )

            agents[agent_id] = AgentStatus(
                id=agent_id,
                name=agent_data.get("name", agent_id),
                description=agent_data.get("description", ""),
                status=AgentState(agent_data.get("status", "pending")),
                progress=agent_data.get("progress", 0),
                current_step=agent_data.get("current_step", ""),
                steps=steps,
                metrics=agent_data.get("metrics", {}),
                error=agent_data.get("error"),
                started_at=agent_data.get("started_at"),
                completed_at=agent_data.get("completed_at"),
                logs=agent_data.get("logs", []),
            )

        return PipelineStatus(
            id=pipeline_data.get("id", "unknown"),
            status=PipelineState(pipeline_data.get("status", "idle")),
            started_at=pipeline_data.get("started_at"),
            completed_at=pipeline_data.get("completed_at"),
            error=pipeline_data.get("error"),
            agents=agents,
        )

    def _notify_subscribers(self):
        """Notify all subscribers of status update."""
        status_dict = self._status.to_dict()
        for callback in self._subscribers:
            try:
                callback(status_dict)
            except Exception:
                pass  # Don't let subscriber errors affect status updates

    def subscribe(self, callback: Callable[[Dict], None]):
        """Subscribe to status updates."""
        self._subscribers.append(callback)

    def unsubscribe(self, callback: Callable[[Dict], None]):
        """Unsubscribe from status updates."""
        if callback in self._subscribers:
            self._subscribers.remove(callback)

    def get_status(self) -> Dict[str, Any]:
        """Get current status as dictionary."""
        with self._lock:
            return self._status.to_dict()

    def get_agent_status(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a specific agent."""
        with self._lock:
            agent = self._status.agents.get(agent_id)
            return agent.to_dict() if agent else None

    def start_pipeline(self):
        """Mark pipeline as started."""
        with self._lock:
            self._status.status = PipelineState.RUNNING
            self._status.started_at = datetime.now().isoformat()
            self._status.completed_at = None
            self._status.error = None
            if self.auto_save:
                self._save_to_file()
            self._notify_subscribers()

    def complete_pipeline(self):
        """Mark pipeline as completed."""
        with self._lock:
            self._status.status = PipelineState.COMPLETED
            self._status.completed_at = datetime.now().isoformat()
            if self.auto_save:
                self._save_to_file()
            self._notify_subscribers()

    def fail_pipeline(self, error: str):
        """Mark pipeline as failed."""
        with self._lock:
            self._status.status = PipelineState.FAILED
            self._status.error = error
            self._status.completed_at = datetime.now().isoformat()
            if self.auto_save:
                self._save_to_file()
            self._notify_subscribers()

    def start_agent(self, agent_id: str):
        """Mark agent as started."""
        with self._lock:
            agent = self._status.agents.get(agent_id)
            if agent:
                agent.status = AgentState.RUNNING
                agent.started_at = datetime.now().isoformat()
                agent.completed_at = None
                agent.error = None
                agent.progress = 0
                if self.auto_save:
                    self._save_to_file()
                self._notify_subscribers()

    def complete_agent(self, agent_id: str, metrics: Dict[str, Any] = None):
        """Mark agent as completed."""
        with self._lock:
            agent = self._status.agents.get(agent_id)
            if agent:
                agent.status = AgentState.COMPLETED
                agent.completed_at = datetime.now().isoformat()
                agent.progress = 100
                if metrics:
                    agent.metrics = metrics
                if self.auto_save:
                    self._save_to_file()
                self._notify_subscribers()

    def fail_agent(self, agent_id: str, error: str):
        """Mark agent as failed."""
        with self._lock:
            agent = self._status.agents.get(agent_id)
            if agent:
                agent.status = AgentState.FAILED
                agent.error = error
                agent.completed_at = datetime.now().isoformat()
                if self.auto_save:
                    self._save_to_file()
                self._notify_subscribers()

    def update_agent_progress(
        self,
        agent_id: str,
        progress: int,
        current_step: str = None,
        message: str = None,
    ):
        """Update agent progress."""
        with self._lock:
            agent = self._status.agents.get(agent_id)
            if agent:
                agent.progress = min(100, max(0, progress))
                if current_step:
                    agent.current_step = current_step
                if message:
                    agent.logs.append(f"[{datetime.now().strftime('%H:%M:%S')}] {message}")
                if self.auto_save:
                    self._save_to_file()
                self._notify_subscribers()

    def update_step(
        self,
        agent_id: str,
        step_id: str,
        status: AgentState = None,
        progress: int = None,
        message: str = None,
        details: Dict[str, Any] = None,
    ):
        """Update a specific step within an agent."""
        with self._lock:
            agent = self._status.agents.get(agent_id)
            if agent and step_id in agent.steps:
                step = agent.steps[step_id]
                if status:
                    step.status = status
                    if status == AgentState.RUNNING:
                        step.started_at = datetime.now().isoformat()
                    elif status in (AgentState.COMPLETED, AgentState.FAILED):
                        step.completed_at = datetime.now().isoformat()
                if progress is not None:
                    step.progress = min(100, max(0, progress))
                if message:
                    step.message = message
                if details:
                    step.details.update(details)

                # Recalculate agent progress from steps
                steps = list(agent.steps.values())
                if steps:
                    agent.progress = sum(s.progress for s in steps) // len(steps)

                if self.auto_save:
                    self._save_to_file()
                self._notify_subscribers()

    def add_agent_log(self, agent_id: str, message: str):
        """Add a log message to an agent."""
        with self._lock:
            agent = self._status.agents.get(agent_id)
            if agent:
                timestamp = datetime.now().strftime('%H:%M:%S')
                agent.logs.append(f"[{timestamp}] {message}")
                # Keep only last 100 logs
                if len(agent.logs) > 100:
                    agent.logs = agent.logs[-100:]
                if self.auto_save:
                    self._save_to_file()
                self._notify_subscribers()

    def reset(self):
        """Reset pipeline to initial state."""
        with self._lock:
            self._status = self._create_initial_status()
            if self.auto_save:
                self._save_to_file()
            self._notify_subscribers()
