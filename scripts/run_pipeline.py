#!/usr/bin/env python3
"""
LegalGPT Pipeline Orchestrator CLI.

Run the 4-agent pipeline with status tracking.

Usage:
    python scripts/run_pipeline.py              # Run full pipeline
    python scripts/run_pipeline.py --status     # Show current status
    python scripts/run_pipeline.py --reset      # Reset pipeline
    python scripts/run_pipeline.py --skip model # Skip specific agents
    python scripts/run_pipeline.py --only citations graph  # Run only specific agents
"""

import asyncio
import argparse
import json
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.orchestrator import PipelineOrchestrator


def print_status(status: dict):
    """Pretty print pipeline status."""
    pipeline = status.get("pipeline", {})
    agents = status.get("agents", {})

    print("\n" + "=" * 60)
    print(f"Pipeline: {pipeline.get('id', 'unknown')}")
    print(f"Status: {pipeline.get('status', 'unknown').upper()}")
    if pipeline.get("started_at"):
        print(f"Started: {pipeline.get('started_at')}")
    if pipeline.get("completed_at"):
        print(f"Completed: {pipeline.get('completed_at')}")
    if pipeline.get("error"):
        print(f"Error: {pipeline.get('error')}")
    print("=" * 60)

    for agent_id, agent in agents.items():
        status_symbol = {
            "pending": "⏳",
            "running": "🔄",
            "completed": "✅",
            "failed": "❌",
            "skipped": "⏭️",
        }.get(agent.get("status", "pending"), "❓")

        progress = agent.get("progress", 0)
        progress_bar = "█" * (progress // 10) + "░" * (10 - progress // 10)

        print(f"\n{status_symbol} {agent.get('name', agent_id)}")
        print(f"   Status: {agent.get('status', 'pending')} [{progress_bar}] {progress}%")

        if agent.get("current_step"):
            print(f"   Current: {agent.get('current_step')}")

        if agent.get("error"):
            print(f"   Error: {agent.get('error')}")

        # Show step details
        steps = agent.get("steps", {})
        for step_id, step in steps.items():
            step_status = step.get("status", "pending")
            step_symbol = {"pending": "○", "running": "◐", "completed": "●", "failed": "✗"}.get(step_status, "?")
            print(f"      {step_symbol} {step.get('name', step_id)}: {step.get('progress', 0)}%")

        # Show metrics
        metrics = agent.get("metrics", {})
        if metrics:
            print(f"   Metrics: {json.dumps(metrics, indent=6)[:200]}...")

    print("\n" + "=" * 60)
    print(f"Last updated: {status.get('last_updated', 'unknown')}")


async def main():
    parser = argparse.ArgumentParser(
        description="LegalGPT Pipeline Orchestrator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python scripts/run_pipeline.py              # Run full pipeline
    python scripts/run_pipeline.py --status     # Show status
    python scripts/run_pipeline.py --reset      # Reset and start fresh
    python scripts/run_pipeline.py --skip model evaluation  # Skip training
    python scripts/run_pipeline.py --only citations  # Run only citations
        """
    )

    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Show current pipeline status"
    )

    parser.add_argument(
        "--reset", "-r",
        action="store_true",
        help="Reset pipeline to initial state"
    )

    parser.add_argument(
        "--skip",
        nargs="*",
        choices=["citations", "graph", "model", "evaluation"],
        default=[],
        help="Agents to skip"
    )

    parser.add_argument(
        "--only",
        nargs="*",
        choices=["citations", "graph", "model", "evaluation"],
        help="Only run these agents (overrides --skip)"
    )

    parser.add_argument(
        "--json",
        action="store_true",
        help="Output status as JSON"
    )

    args = parser.parse_args()

    # Initialize orchestrator
    orchestrator = PipelineOrchestrator()

    if args.status:
        status = orchestrator.get_status()
        if args.json:
            print(json.dumps(status, indent=2))
        else:
            print_status(status)
        return

    if args.reset:
        orchestrator.reset()
        print("Pipeline reset to initial state.")
        return

    # Run pipeline
    print("Starting LegalGPT Pipeline...")
    print(f"Skip agents: {args.skip or 'none'}")
    print(f"Only agents: {args.only or 'all'}")
    print()

    try:
        results = await orchestrator.run(
            skip_agents=args.skip,
            only_agents=args.only,
        )

        print("\n" + "=" * 60)
        print("Pipeline completed!")
        print("=" * 60)

        if args.json:
            print(json.dumps(results, indent=2, default=str))
        else:
            for agent_id, result in results.items():
                print(f"\n{agent_id}:")
                if isinstance(result, dict):
                    for k, v in result.items():
                        print(f"  {k}: {v}")
                else:
                    print(f"  {result}")

    except KeyboardInterrupt:
        print("\nPipeline interrupted.")
        status = orchestrator.get_status()
        print_status(status)

    except Exception as e:
        print(f"\nPipeline failed: {e}")
        status = orchestrator.get_status()
        print_status(status)
        raise


if __name__ == "__main__":
    asyncio.run(main())
