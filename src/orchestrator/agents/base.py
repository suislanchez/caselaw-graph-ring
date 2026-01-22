"""
Base agent class for pipeline agents.
"""

import asyncio
from abc import ABC, abstractmethod
from typing import Dict, Any, List
from ..status import StatusManager, AgentState


class BaseAgent(ABC):
    """Abstract base class for pipeline agents."""

    def __init__(
        self,
        agent_id: str,
        status_manager: StatusManager,
        dependencies: List[str] = None,
    ):
        self.agent_id = agent_id
        self.status = status_manager
        self.dependencies = dependencies or []
        self._cancelled = False

    @abstractmethod
    async def run(self) -> Dict[str, Any]:
        """Execute the agent's main task."""
        pass

    def cancel(self):
        self._cancelled = True

    async def wait_for_dependencies(self, poll_interval: float = 2.0):
        """Wait for all dependencies to complete."""
        if not self.dependencies:
            return

        self.log(f"Waiting for: {', '.join(self.dependencies)}")

        while True:
            if self._cancelled:
                raise asyncio.CancelledError("Agent cancelled")

            all_complete = True
            for dep_id in self.dependencies:
                dep_status = self.status.get_agent_status(dep_id)
                if not dep_status:
                    continue
                status = dep_status.get("status", "pending")
                if status == "failed":
                    raise RuntimeError(f"Dependency '{dep_id}' failed")
                if status != "completed":
                    all_complete = False
                    break

            if all_complete:
                return
            await asyncio.sleep(poll_interval)

    def start(self):
        self.status.start_agent(self.agent_id)

    def complete(self, metrics: Dict[str, Any] = None):
        self.status.complete_agent(self.agent_id, metrics)

    def fail(self, error: str):
        self.status.fail_agent(self.agent_id, error)

    def update_progress(self, progress: int, message: str = None):
        self.status.update_agent_progress(self.agent_id, progress, message=message)

    def start_step(self, step_id: str, message: str = None):
        self.status.update_step(self.agent_id, step_id, status=AgentState.RUNNING, message=message)
        self.status.update_agent_progress(self.agent_id, progress=self.status.get_agent_status(self.agent_id).get("progress", 0), current_step=step_id)

    def complete_step(self, step_id: str, details: Dict[str, Any] = None):
        self.status.update_step(self.agent_id, step_id, status=AgentState.COMPLETED, progress=100, details=details)

    def update_step_progress(self, step_id: str, progress: int, message: str = None, details: Dict[str, Any] = None):
        self.status.update_step(self.agent_id, step_id, progress=progress, message=message, details=details)

    def log(self, message: str):
        self.status.add_agent_log(self.agent_id, message)

    async def run_with_status(self) -> Dict[str, Any]:
        """Run agent with automatic status updates."""
        try:
            await self.wait_for_dependencies()
            self.start()
            result = await self.run()
            self.complete(result)
            return result
        except asyncio.CancelledError:
            self.fail("Agent cancelled")
            raise
        except Exception as e:
            self.fail(str(e))
            raise
