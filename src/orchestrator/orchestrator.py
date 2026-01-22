"""
Main pipeline orchestrator for LegalGPT.

Coordinates 4 agents running in dependency order:
Citations -> Graph -> Model -> Evaluation
"""

import asyncio
from typing import Dict, Any, List, Optional
from pathlib import Path
from datetime import datetime

from .status import StatusManager, PipelineState
from .agents import CitationsAgent, GraphAgent, ModelAgent, EvaluationAgent


class PipelineOrchestrator:
    """
    Orchestrates the 4-agent LegalGPT pipeline.

    Usage:
        orchestrator = PipelineOrchestrator()
        results = await orchestrator.run()
    """

    def __init__(self, status_path: Optional[Path] = None):
        """
        Initialize orchestrator.

        Args:
            status_path: Path to status JSON file (default: results/pipeline_status.json)
        """
        if status_path is None:
            status_path = Path("results/pipeline_status.json")

        self.status_path = Path(status_path)
        self.status = StatusManager(self.status_path)

        # Initialize agents
        self.agents = {
            "citations": CitationsAgent(self.status),
            "graph": GraphAgent(self.status),
            "model": ModelAgent(self.status),
            "evaluation": EvaluationAgent(self.status),
        }

    async def run(
        self,
        skip_agents: List[str] = None,
        only_agents: List[str] = None,
    ) -> Dict[str, Any]:
        """
        Run the full pipeline.

        Args:
            skip_agents: List of agent IDs to skip
            only_agents: List of agent IDs to run (exclusive with skip_agents)

        Returns:
            Dictionary with results from all agents
        """
        skip_agents = skip_agents or []
        results = {}

        # Determine which agents to run
        agents_to_run = list(self.agents.keys())
        if only_agents:
            agents_to_run = [a for a in agents_to_run if a in only_agents]
        else:
            agents_to_run = [a for a in agents_to_run if a not in skip_agents]

        # Start pipeline
        self.status.start_pipeline()

        try:
            # Run agents in dependency order
            for agent_id in ["citations", "graph", "model", "evaluation"]:
                if agent_id not in agents_to_run:
                    self.status.add_agent_log(agent_id, "Skipped")
                    continue

                agent = self.agents[agent_id]
                try:
                    result = await agent.run_with_status()
                    results[agent_id] = result
                except Exception as e:
                    results[agent_id] = {"error": str(e)}
                    # Don't fail entire pipeline on agent failure
                    self.status.add_agent_log(agent_id, f"Failed: {e}")

            # Complete pipeline
            self.status.complete_pipeline()

        except Exception as e:
            self.status.fail_pipeline(str(e))
            raise

        return results

    async def run_agent(self, agent_id: str) -> Dict[str, Any]:
        """Run a single agent."""
        if agent_id not in self.agents:
            raise ValueError(f"Unknown agent: {agent_id}")

        agent = self.agents[agent_id]
        return await agent.run_with_status()

    def get_status(self) -> Dict[str, Any]:
        """Get current pipeline status."""
        return self.status.get_status()

    def reset(self):
        """Reset pipeline to initial state."""
        self.status.reset()


async def run_pipeline(
    skip: List[str] = None,
    only: List[str] = None,
    status_path: Path = None,
) -> Dict[str, Any]:
    """
    Convenience function to run the pipeline.

    Args:
        skip: Agents to skip
        only: Only run these agents
        status_path: Custom status file path

    Returns:
        Results dictionary
    """
    orchestrator = PipelineOrchestrator(status_path)
    return await orchestrator.run(skip_agents=skip, only_agents=only)


def get_status(status_path: Path = None) -> Dict[str, Any]:
    """Get current pipeline status."""
    if status_path is None:
        status_path = Path("results/pipeline_status.json")
    status = StatusManager(status_path)
    return status.get_status()
