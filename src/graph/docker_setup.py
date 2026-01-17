"""Neo4j Docker container management for LegalGPT."""

import subprocess
import time
from typing import Optional

from neo4j import GraphDatabase
from neo4j.exceptions import ServiceUnavailable

from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


CONTAINER_NAME = "legalgpt-neo4j"
NEO4J_IMAGE = "neo4j:5.15.0"
NEO4J_HTTP_PORT = 7474
NEO4J_BOLT_PORT = 7687
NEO4J_MEMORY = "2G"


def _run_docker_cmd(args: list[str], check: bool = True) -> subprocess.CompletedProcess:
    """Run a docker command."""
    return subprocess.run(
        ["docker"] + args,
        capture_output=True,
        text=True,
        check=check,
    )


def is_container_running() -> bool:
    """Check if the Neo4j container is running."""
    result = _run_docker_cmd(
        ["ps", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}"],
        check=False,
    )
    return CONTAINER_NAME in result.stdout


# Alias for convenience
is_running = is_container_running


def is_container_exists() -> bool:
    """Check if the Neo4j container exists (running or stopped)."""
    result = _run_docker_cmd(
        ["ps", "-a", "--filter", f"name={CONTAINER_NAME}", "--format", "{{.Names}}"],
        check=False,
    )
    return CONTAINER_NAME in result.stdout


def start_neo4j(
    password: Optional[str] = None,
    memory: str = NEO4J_MEMORY,
    wait_for_ready: bool = True,
    timeout: int = 60,
) -> bool:
    """
    Start the Neo4j Docker container.

    Args:
        password: Neo4j password (defaults to config value)
        memory: Memory allocation for Neo4j heap
        wait_for_ready: Wait for Neo4j to be ready to accept connections
        timeout: Maximum seconds to wait for Neo4j to be ready

    Returns:
        True if Neo4j started successfully
    """
    password = password or NEO4J_PASSWORD

    if is_container_running():
        print(f"Container '{CONTAINER_NAME}' is already running.")
        return True

    if is_container_exists():
        print(f"Starting existing container '{CONTAINER_NAME}'...")
        _run_docker_cmd(["start", CONTAINER_NAME])
    else:
        print(f"Creating and starting new container '{CONTAINER_NAME}'...")
        _run_docker_cmd([
            "run", "-d",
            "--name", CONTAINER_NAME,
            "-p", f"{NEO4J_HTTP_PORT}:7474",
            "-p", f"{NEO4J_BOLT_PORT}:7687",
            "-e", f"NEO4J_AUTH={NEO4J_USER}/{password}",
            "-e", f"NEO4J_dbms_memory_heap_initial__size={memory}",
            "-e", f"NEO4J_dbms_memory_heap_max__size={memory}",
            "-e", "NEO4J_PLUGINS=[\"apoc\"]",
            "-v", "legalgpt-neo4j-data:/data",
            NEO4J_IMAGE,
        ])

    if wait_for_ready:
        return wait_for_neo4j(timeout=timeout)

    return True


def stop_neo4j() -> bool:
    """Stop the Neo4j Docker container."""
    if not is_container_running():
        print(f"Container '{CONTAINER_NAME}' is not running.")
        return True

    print(f"Stopping container '{CONTAINER_NAME}'...")
    result = _run_docker_cmd(["stop", CONTAINER_NAME], check=False)
    return result.returncode == 0


def remove_neo4j(remove_volume: bool = False) -> bool:
    """
    Remove the Neo4j Docker container.

    Args:
        remove_volume: Also remove the data volume (destructive!)
    """
    stop_neo4j()

    if is_container_exists():
        print(f"Removing container '{CONTAINER_NAME}'...")
        _run_docker_cmd(["rm", CONTAINER_NAME], check=False)

    if remove_volume:
        print("Removing data volume 'legalgpt-neo4j-data'...")
        _run_docker_cmd(["volume", "rm", "legalgpt-neo4j-data"], check=False)

    return True


def wait_for_neo4j(timeout: int = 60) -> bool:
    """
    Wait for Neo4j to be ready to accept connections.

    Args:
        timeout: Maximum seconds to wait

    Returns:
        True if Neo4j is ready, False if timeout exceeded
    """
    print("Waiting for Neo4j to be ready...")
    start_time = time.time()

    while time.time() - start_time < timeout:
        if check_health():
            print("Neo4j is ready!")
            return True
        time.sleep(2)

    print(f"Timeout waiting for Neo4j after {timeout} seconds.")
    return False


# Alias for convenience
wait_for_ready = wait_for_neo4j


def get_container_logs(tail: int = 100) -> str:
    """
    Get logs from the Neo4j container.

    Args:
        tail: Number of lines to return from the end of logs

    Returns:
        Container log output as string
    """
    if not is_container_exists():
        return f"Container '{CONTAINER_NAME}' does not exist."

    result = _run_docker_cmd(
        ["logs", "--tail", str(tail), CONTAINER_NAME],
        check=False,
    )
    return result.stdout + result.stderr


def check_health() -> bool:
    """Check if Neo4j is healthy and accepting connections."""
    try:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("RETURN 1 AS health")
            result.single()
        driver.close()
        return True
    except (ServiceUnavailable, Exception):
        return False


def get_connection():
    """Get a Neo4j driver connection."""
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


def get_stats() -> dict:
    """Get Neo4j database statistics."""
    driver = get_connection()
    try:
        with driver.session() as session:
            # Count nodes
            node_result = session.run("MATCH (n) RETURN count(n) AS count")
            node_count = node_result.single()["count"]

            # Count relationships
            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) AS count")
            rel_count = rel_result.single()["count"]

            # Count by label
            label_result = session.run("""
                CALL db.labels() YIELD label
                CALL apoc.cypher.run('MATCH (n:`' + label + '`) RETURN count(n) AS count', {})
                YIELD value
                RETURN label, value.count AS count
            """)
            labels = {record["label"]: record["count"] for record in label_result}

            return {
                "total_nodes": node_count,
                "total_relationships": rel_count,
                "nodes_by_label": labels,
            }
    except Exception as e:
        # Fallback without APOC
        with driver.session() as session:
            node_result = session.run("MATCH (n) RETURN count(n) AS count")
            node_count = node_result.single()["count"]

            rel_result = session.run("MATCH ()-[r]->() RETURN count(r) AS count")
            rel_count = rel_result.single()["count"]

            return {
                "total_nodes": node_count,
                "total_relationships": rel_count,
                "nodes_by_label": {},
            }
    finally:
        driver.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python docker_setup.py [start|stop|restart|status|remove]")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "start":
        start_neo4j()
    elif command == "stop":
        stop_neo4j()
    elif command == "restart":
        stop_neo4j()
        start_neo4j()
    elif command == "status":
        if is_container_running():
            print(f"Container '{CONTAINER_NAME}' is running.")
            if check_health():
                stats = get_stats()
                print(f"  Nodes: {stats['total_nodes']}")
                print(f"  Relationships: {stats['total_relationships']}")
            else:
                print("  (Not healthy yet)")
        elif is_container_exists():
            print(f"Container '{CONTAINER_NAME}' exists but is stopped.")
        else:
            print(f"Container '{CONTAINER_NAME}' does not exist.")
    elif command == "remove":
        remove_volume = "--volume" in sys.argv or "-v" in sys.argv
        remove_neo4j(remove_volume=remove_volume)
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
