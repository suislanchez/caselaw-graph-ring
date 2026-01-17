"""Neo4j schema definitions for LegalGPT citation graph."""

from neo4j import GraphDatabase

from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


# Human-readable schema documentation
SCHEMA_CYPHER = """
// Case node constraint
CREATE CONSTRAINT case_id IF NOT EXISTS FOR (c:Case) REQUIRE c.id IS UNIQUE;

// Properties: id, name, date, court, text_embedding (list), outcome
// Edge type: CITES with properties: citation_text, weight
"""


# Cypher statements for schema setup
SCHEMA_STATEMENTS = [
    # Constraints
    """
    CREATE CONSTRAINT case_id IF NOT EXISTS
    FOR (c:Case) REQUIRE c.id IS UNIQUE
    """,
    """
    CREATE CONSTRAINT case_cap_id IF NOT EXISTS
    FOR (c:Case) REQUIRE c.cap_id IS UNIQUE
    """,
    # Indexes for common query patterns
    """
    CREATE INDEX case_name IF NOT EXISTS
    FOR (c:Case) ON (c.name)
    """,
    """
    CREATE INDEX case_date IF NOT EXISTS
    FOR (c:Case) ON (c.date)
    """,
    """
    CREATE INDEX case_court IF NOT EXISTS
    FOR (c:Case) ON (c.court)
    """,
    """
    CREATE INDEX case_outcome IF NOT EXISTS
    FOR (c:Case) ON (c.outcome)
    """,
    # Vector index for embeddings (Neo4j 5.x)
    """
    CREATE VECTOR INDEX case_embedding IF NOT EXISTS
    FOR (c:Case) ON (c.embedding)
    OPTIONS {indexConfig: {
        `vector.dimensions`: 768,
        `vector.similarity_function`: 'cosine'
    }}
    """,
]


# Node property descriptions
CASE_NODE_PROPERTIES = {
    "id": "Unique internal identifier (string)",
    "cap_id": "Caselaw Access Project case ID (string)",
    "scdb_id": "Supreme Court Database case ID (string, optional)",
    "name": "Case name, e.g., 'Brown v. Board of Education' (string)",
    "date": "Decision date (date)",
    "year": "Decision year (integer)",
    "court": "Court name (string)",
    "citation": "Primary citation, e.g., '347 U.S. 483' (string)",
    "outcome": "Case outcome: 'petitioner' or 'respondent' (string, optional)",
    "text": "Full case text (string)",
    "text_length": "Character count of case text (integer)",
    "embedding": "768-dimensional text embedding (list of floats)",
    "graphsage_embedding": "GraphSAGE node embedding (list of floats)",
}


# Relationship descriptions
CITES_RELATIONSHIP_PROPERTIES = {
    "citation_text": "Raw citation string as it appears in source case (string)",
    "weight": "Citation weight/importance score (float, default 1.0)",
    "context": "Surrounding text context of citation (string, optional)",
}


def create_schema(driver=None) -> list[str]:
    """
    Create the Neo4j schema (constraints and indexes).

    Args:
        driver: Neo4j driver instance (creates new one if not provided)

    Returns:
        List of executed statements
    """
    close_driver = False
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        close_driver = True

    executed = []
    try:
        with driver.session() as session:
            for statement in SCHEMA_STATEMENTS:
                try:
                    session.run(statement.strip())
                    executed.append(statement.strip().split("\n")[1].strip())
                except Exception as e:
                    # Some statements may fail if already exists or not supported
                    if "already exists" not in str(e).lower():
                        print(f"Warning: {e}")
    finally:
        if close_driver:
            driver.close()

    return executed


def drop_schema(driver=None) -> list[str]:
    """
    Drop all schema constraints and indexes.

    Args:
        driver: Neo4j driver instance

    Returns:
        List of dropped items
    """
    close_driver = False
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        close_driver = True

    dropped = []
    try:
        with driver.session() as session:
            # Get all constraints
            constraints = session.run("SHOW CONSTRAINTS")
            for record in constraints:
                name = record["name"]
                session.run(f"DROP CONSTRAINT {name} IF EXISTS")
                dropped.append(f"constraint:{name}")

            # Get all indexes
            indexes = session.run("SHOW INDEXES")
            for record in indexes:
                name = record["name"]
                if not name.startswith("constraint"):  # Skip constraint-backed indexes
                    session.run(f"DROP INDEX {name} IF EXISTS")
                    dropped.append(f"index:{name}")
    finally:
        if close_driver:
            driver.close()

    return dropped


def clear_all_data(driver=None) -> int:
    """
    Delete all nodes and relationships from the database.

    Args:
        driver: Neo4j driver instance

    Returns:
        Number of deleted nodes
    """
    close_driver = False
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        close_driver = True

    try:
        with driver.session() as session:
            # Delete in batches to avoid memory issues
            total_deleted = 0
            while True:
                result = session.run("""
                    MATCH (n)
                    WITH n LIMIT 10000
                    DETACH DELETE n
                    RETURN count(*) AS deleted
                """)
                deleted = result.single()["deleted"]
                total_deleted += deleted
                if deleted == 0:
                    break
            return total_deleted
    finally:
        if close_driver:
            driver.close()


def get_schema_info(driver=None) -> dict:
    """
    Get current schema information.

    Args:
        driver: Neo4j driver instance

    Returns:
        Dict with constraints and indexes
    """
    close_driver = False
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        close_driver = True

    try:
        with driver.session() as session:
            # Get constraints
            constraints_result = session.run("SHOW CONSTRAINTS")
            constraints = [
                {
                    "name": r["name"],
                    "type": r["type"],
                    "entity_type": r["entityType"],
                    "labelsOrTypes": r["labelsOrTypes"],
                    "properties": r["properties"],
                }
                for r in constraints_result
            ]

            # Get indexes
            indexes_result = session.run("SHOW INDEXES")
            indexes = [
                {
                    "name": r["name"],
                    "type": r["type"],
                    "entity_type": r["entityType"],
                    "labelsOrTypes": r["labelsOrTypes"],
                    "properties": r["properties"],
                    "state": r["state"],
                }
                for r in indexes_result
            ]

            return {
                "constraints": constraints,
                "indexes": indexes,
            }
    finally:
        if close_driver:
            driver.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python schema.py [create|drop|clear|info]")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "create":
        print("Creating schema...")
        executed = create_schema()
        print(f"Executed {len(executed)} schema statements:")
        for stmt in executed:
            print(f"  - {stmt}")
    elif command == "drop":
        print("Dropping schema...")
        dropped = drop_schema()
        print(f"Dropped {len(dropped)} items:")
        for item in dropped:
            print(f"  - {item}")
    elif command == "clear":
        confirm = input("This will DELETE ALL DATA. Type 'yes' to confirm: ")
        if confirm.lower() == "yes":
            print("Clearing all data...")
            deleted = clear_all_data()
            print(f"Deleted {deleted} nodes.")
        else:
            print("Cancelled.")
    elif command == "info":
        info = get_schema_info()
        print("Constraints:")
        for c in info["constraints"]:
            print(f"  - {c['name']}: {c['type']} on {c['labelsOrTypes']}.{c['properties']}")
        print("\nIndexes:")
        for i in info["indexes"]:
            print(f"  - {i['name']}: {i['type']} on {i['labelsOrTypes']}.{i['properties']} [{i['state']}]")
    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
