"""Graph visualization utilities for LegalGPT citation network."""

from pathlib import Path
from typing import Optional

import networkx as nx
import numpy as np
from neo4j import GraphDatabase

from src.config import (
    NEO4J_URI,
    NEO4J_USER,
    NEO4J_PASSWORD,
    RESULTS_DIR,
)


def load_subgraph_from_neo4j(
    center_case_id: str,
    max_hops: int = 2,
    max_nodes: int = 100,
    driver=None,
) -> nx.DiGraph:
    """
    Load a subgraph centered on a case from Neo4j.

    Args:
        center_case_id: ID of the center case
        max_hops: Maximum hops from center
        max_nodes: Maximum nodes to include
        driver: Neo4j driver instance

    Returns:
        NetworkX DiGraph
    """
    close_driver = False
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        close_driver = True

    try:
        with driver.session() as session:
            # Get subgraph nodes and edges
            result = session.run(
                """
                MATCH path = (center:Case {id: $center_id})-[:CITES*0..$max_hops]-(c:Case)
                WITH DISTINCT c
                LIMIT $max_nodes
                WITH collect(c) AS nodes
                UNWIND nodes AS n1
                UNWIND nodes AS n2
                OPTIONAL MATCH (n1)-[r:CITES]->(n2)
                WITH n1, n2, r
                WHERE r IS NOT NULL
                RETURN n1.id AS source_id,
                       n1.name AS source_name,
                       n1.outcome AS source_outcome,
                       n1.year AS source_year,
                       n2.id AS target_id,
                       n2.name AS target_name,
                       n2.outcome AS target_outcome,
                       n2.year AS target_year,
                       r.weight AS weight
                """,
                center_id=center_case_id,
                max_hops=max_hops,
                max_nodes=max_nodes,
            )

            G = nx.DiGraph()
            edges = list(result)

            for e in edges:
                # Add source node
                if e["source_id"] not in G:
                    G.add_node(
                        e["source_id"],
                        name=e["source_name"],
                        outcome=e["source_outcome"],
                        year=e["source_year"],
                        is_center=(e["source_id"] == center_case_id),
                    )

                # Add target node
                if e["target_id"] not in G:
                    G.add_node(
                        e["target_id"],
                        name=e["target_name"],
                        outcome=e["target_outcome"],
                        year=e["target_year"],
                        is_center=(e["target_id"] == center_case_id),
                    )

                # Add edge
                G.add_edge(
                    e["source_id"],
                    e["target_id"],
                    weight=e["weight"] or 1.0,
                )

            # If center node wasn't in any edge, add it
            if center_case_id not in G:
                center_result = session.run(
                    """
                    MATCH (c:Case {id: $center_id})
                    RETURN c.name AS name, c.outcome AS outcome, c.year AS year
                    """,
                    center_id=center_case_id,
                )
                center_data = center_result.single()
                if center_data:
                    G.add_node(
                        center_case_id,
                        name=center_data["name"],
                        outcome=center_data["outcome"],
                        year=center_data["year"],
                        is_center=True,
                    )

            return G

    finally:
        if close_driver:
            driver.close()


def visualize_citation_graph_matplotlib(
    G: nx.DiGraph,
    output_path: Optional[Path] = None,
    figsize: tuple[int, int] = (14, 10),
    title: str = "Citation Network",
    show_labels: bool = True,
) -> None:
    """
    Visualize citation graph using matplotlib.

    Args:
        G: NetworkX DiGraph
        output_path: Path to save figure (displays if None)
        figsize: Figure size
        title: Plot title
        show_labels: Whether to show node labels
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Node colors by outcome
    node_colors = []
    for node in G.nodes():
        outcome = G.nodes[node].get("outcome")
        is_center = G.nodes[node].get("is_center", False)
        if is_center:
            node_colors.append("#FFD700")  # Gold for center
        elif outcome == "petitioner":
            node_colors.append("#4CAF50")  # Green
        elif outcome == "respondent":
            node_colors.append("#F44336")  # Red
        else:
            node_colors.append("#9E9E9E")  # Gray for unknown

    # Node sizes
    node_sizes = [
        800 if G.nodes[node].get("is_center", False) else 300
        for node in G.nodes()
    ]

    # Draw
    nx.draw_networkx_nodes(
        G, pos, node_color=node_colors, node_size=node_sizes, alpha=0.8, ax=ax
    )
    nx.draw_networkx_edges(
        G, pos, edge_color="#CCCCCC", arrows=True, arrowsize=10, alpha=0.5, ax=ax
    )

    if show_labels:
        # Truncate labels
        labels = {
            node: G.nodes[node].get("name", node)[:30] + "..."
            if len(G.nodes[node].get("name", node)) > 30
            else G.nodes[node].get("name", node)
            for node in G.nodes()
        }
        nx.draw_networkx_labels(G, pos, labels, font_size=8, ax=ax)

    ax.set_title(title)
    ax.axis("off")

    # Legend
    legend_elements = [
        plt.scatter([], [], c="#FFD700", s=100, label="Query Case"),
        plt.scatter([], [], c="#4CAF50", s=100, label="Petitioner Win"),
        plt.scatter([], [], c="#F44336", s=100, label="Respondent Win"),
        plt.scatter([], [], c="#9E9E9E", s=100, label="Unknown"),
    ]
    ax.legend(handles=legend_elements, loc="upper left")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_citation_graph_plotly(
    G: nx.DiGraph,
    output_path: Optional[Path] = None,
    title: str = "Citation Network",
) -> None:
    """
    Create interactive citation graph visualization using Plotly.

    Args:
        G: NetworkX DiGraph
        output_path: Path to save HTML file (displays if None)
        title: Plot title
    """
    import plotly.graph_objects as go

    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

    # Edge traces
    edge_x = []
    edge_y = []
    for edge in G.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=0.5, color="#888"),
        hoverinfo="none",
        mode="lines",
    )

    # Node traces
    node_x = []
    node_y = []
    node_colors = []
    node_sizes = []
    node_text = []

    for node in G.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)

        outcome = G.nodes[node].get("outcome")
        is_center = G.nodes[node].get("is_center", False)

        if is_center:
            node_colors.append("#FFD700")
            node_sizes.append(25)
        elif outcome == "petitioner":
            node_colors.append("#4CAF50")
            node_sizes.append(15)
        elif outcome == "respondent":
            node_colors.append("#F44336")
            node_sizes.append(15)
        else:
            node_colors.append("#9E9E9E")
            node_sizes.append(12)

        name = G.nodes[node].get("name", node)
        year = G.nodes[node].get("year", "")
        node_text.append(f"{name}<br>Year: {year}<br>Outcome: {outcome}")

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=node_text,
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=1, color="#fff"),
        ),
    )

    # Create figure
    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            title=title,
            showlegend=False,
            hovermode="closest",
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            margin=dict(b=0, l=0, r=0, t=40),
        ),
    )

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(str(output_path))
        print(f"Saved interactive visualization to {output_path}")
    else:
        fig.show()


def visualize_embedding_space(
    node_ids: list[str],
    embeddings: np.ndarray,
    outcomes: Optional[list[str]] = None,
    output_path: Optional[Path] = None,
    method: str = "tsne",
    title: str = "Case Embedding Space",
) -> None:
    """
    Visualize case embeddings in 2D using dimensionality reduction.

    Args:
        node_ids: List of case IDs
        embeddings: Embedding matrix [num_nodes, embedding_dim]
        outcomes: Optional list of outcomes for coloring
        output_path: Path to save figure
        method: 'tsne' or 'umap'
        title: Plot title
    """
    import matplotlib.pyplot as plt

    # Reduce dimensionality
    if method == "tsne":
        from sklearn.manifold import TSNE

        reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    elif method == "umap":
        try:
            from umap import UMAP

            reducer = UMAP(n_components=2, random_state=42)
        except ImportError:
            print("UMAP not installed, falling back to t-SNE")
            from sklearn.manifold import TSNE

            reducer = TSNE(n_components=2, random_state=42, perplexity=min(30, len(embeddings) - 1))
    else:
        raise ValueError(f"Unknown method: {method}")

    coords = reducer.fit_transform(embeddings)

    # Plot
    fig, ax = plt.subplots(figsize=(12, 10))

    if outcomes:
        colors = {
            "petitioner": "#4CAF50",
            "respondent": "#F44336",
            None: "#9E9E9E",
        }
        for outcome in ["petitioner", "respondent", None]:
            mask = [o == outcome for o in outcomes]
            if any(mask):
                coords_subset = coords[mask]
                label = outcome if outcome else "Unknown"
                ax.scatter(
                    coords_subset[:, 0],
                    coords_subset[:, 1],
                    c=colors[outcome],
                    label=label,
                    alpha=0.6,
                    s=20,
                )
        ax.legend()
    else:
        ax.scatter(coords[:, 0], coords[:, 1], alpha=0.6, s=20)

    ax.set_title(title)
    ax.set_xlabel("Dimension 1")
    ax.set_ylabel("Dimension 2")

    plt.tight_layout()

    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    else:
        plt.show()

    plt.close()


def visualize_subgraph(
    center_case_id: str,
    depth: int = 2,
    driver=None,
    output_path: Optional[Path] = None,
) -> "plt.Figure":
    """
    Visualize a subgraph centered on a specific case.

    This is the main visualization function matching the specification.

    Args:
        center_case_id: ID of the center case
        depth: Number of hops to include
        driver: Neo4j driver instance
        output_path: Optional path to save the figure

    Returns:
        Matplotlib Figure object
    """
    import matplotlib.pyplot as plt

    G = load_subgraph_from_neo4j(center_case_id, max_hops=depth, driver=driver)
    return visualize_citation_graph_matplotlib(
        G,
        output_path=output_path,
        title=f"Citation Network: {center_case_id} (depth={depth})",
    )


def export_to_gephi(driver=None, output_path: Optional[Path] = None) -> Path:
    """
    Export the citation graph to GEXF format for Gephi visualization.

    Args:
        driver: Neo4j driver instance
        output_path: Path to save the GEXF file

    Returns:
        Path to the exported file
    """
    close_driver = False
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        close_driver = True

    output_path = output_path or (RESULTS_DIR / "citation_graph.gexf")
    output_path = Path(output_path)

    try:
        with driver.session() as session:
            # Get all nodes
            nodes_result = session.run("""
                MATCH (c:Case)
                RETURN c.id AS id, c.name AS name, c.court AS court,
                       c.year AS year, c.outcome AS outcome
            """)
            nodes = list(nodes_result)

            # Get all edges
            edges_result = session.run("""
                MATCH (s:Case)-[r:CITES]->(t:Case)
                RETURN s.id AS source, t.id AS target,
                       coalesce(r.weight, 1.0) AS weight
            """)
            edges = list(edges_result)

        # Build NetworkX graph
        G = nx.DiGraph()

        for node in nodes:
            G.add_node(
                node["id"],
                name=node["name"] or "",
                court=node["court"] or "",
                year=node["year"] or 0,
                outcome=node["outcome"] or "",
            )

        for edge in edges:
            G.add_edge(
                edge["source"],
                edge["target"],
                weight=edge["weight"],
            )

        # Export to GEXF
        output_path.parent.mkdir(parents=True, exist_ok=True)
        nx.write_gexf(G, str(output_path))

        print(f"Exported {len(nodes)} nodes and {len(edges)} edges to {output_path}")
        return output_path

    finally:
        if close_driver:
            driver.close()


def get_graph_statistics(driver=None) -> dict:
    """
    Compute graph statistics from Neo4j.

    Returns:
        Dict with graph statistics
    """
    close_driver = False
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        close_driver = True

    try:
        with driver.session() as session:
            # Basic counts
            counts = session.run("""
                MATCH (c:Case)
                OPTIONAL MATCH (c)-[r:CITES]->()
                RETURN count(DISTINCT c) AS num_nodes,
                       count(r) AS num_edges
            """).single()

            # Degree statistics
            degree_stats = session.run("""
                MATCH (c:Case)
                OPTIONAL MATCH (c)-[out:CITES]->()
                OPTIONAL MATCH ()-[in:CITES]->(c)
                WITH c, count(DISTINCT out) AS out_degree, count(DISTINCT in) AS in_degree
                RETURN avg(out_degree) AS avg_out_degree,
                       max(out_degree) AS max_out_degree,
                       avg(in_degree) AS avg_in_degree,
                       max(in_degree) AS max_in_degree
            """).single()

            # Outcome distribution
            outcome_dist = session.run("""
                MATCH (c:Case)
                RETURN c.outcome AS outcome, count(*) AS count
            """)
            outcomes = {r["outcome"]: r["count"] for r in outcome_dist}

            # Year range
            year_range = session.run("""
                MATCH (c:Case)
                WHERE c.year IS NOT NULL
                RETURN min(c.year) AS min_year, max(c.year) AS max_year
            """).single()

            return {
                "num_nodes": counts["num_nodes"],
                "num_edges": counts["num_edges"],
                "avg_out_degree": round(degree_stats["avg_out_degree"] or 0, 2),
                "max_out_degree": degree_stats["max_out_degree"] or 0,
                "avg_in_degree": round(degree_stats["avg_in_degree"] or 0, 2),
                "max_in_degree": degree_stats["max_in_degree"] or 0,
                "outcome_distribution": outcomes,
                "year_range": (year_range["min_year"], year_range["max_year"]),
            }

    finally:
        if close_driver:
            driver.close()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python visualize.py [stats|graph <case_id>|embeddings]")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "stats":
        print("Computing graph statistics...")
        stats = get_graph_statistics()
        print(f"\nGraph Statistics:")
        print(f"  Nodes: {stats['num_nodes']}")
        print(f"  Edges: {stats['num_edges']}")
        print(f"  Avg out-degree: {stats['avg_out_degree']}")
        print(f"  Max out-degree: {stats['max_out_degree']}")
        print(f"  Avg in-degree: {stats['avg_in_degree']}")
        print(f"  Max in-degree: {stats['max_in_degree']}")
        print(f"  Year range: {stats['year_range'][0]} - {stats['year_range'][1]}")
        print(f"  Outcome distribution: {stats['outcome_distribution']}")

    elif command == "graph":
        if len(sys.argv) < 3:
            print("Usage: python visualize.py graph <case_id>")
            sys.exit(1)

        case_id = sys.argv[2]
        print(f"Loading subgraph for case {case_id}...")
        G = load_subgraph_from_neo4j(case_id)
        print(f"Loaded {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")

        output_path = RESULTS_DIR / f"citation_graph_{case_id}.html"
        visualize_citation_graph_plotly(G, output_path=output_path)

    elif command == "embeddings":
        from src.graph.graphsage import load_embeddings

        print("Loading embeddings...")
        node_ids, embeddings = load_embeddings()

        # Get outcomes from Neo4j
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        with driver.session() as session:
            result = session.run("""
                MATCH (c:Case)
                WHERE c.id IN $node_ids
                RETURN c.id AS id, c.outcome AS outcome
            """, node_ids=node_ids)
            outcome_map = {r["id"]: r["outcome"] for r in result}
        driver.close()

        outcomes = [outcome_map.get(nid) for nid in node_ids]

        output_path = RESULTS_DIR / "embedding_space.png"
        visualize_embedding_space(
            node_ids,
            embeddings.numpy(),
            outcomes=outcomes,
            output_path=output_path,
        )

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
