"""Citation graph visualization utilities."""

import json
from pathlib import Path
from typing import List, Dict, Optional, Any, Tuple
import sys

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.config import DATA_DIR, RESULTS_DIR


def load_citation_edges(edges_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load citation edges from CSV."""
    import csv
    
    if edges_path is None:
        edges_path = DATA_DIR / "citations" / "edges.csv"
    
    edges = []
    if edges_path.exists():
        with open(edges_path) as f:
            reader = csv.DictReader(f)
            for row in reader:
                edges.append({
                    "source": row.get("source_case_id", row.get("source", "")),
                    "target": row.get("target_case_id", row.get("target", "")),
                    "weight": float(row.get("weight", 1.0)),
                })
    
    return edges


def load_cases(cases_path: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Load case metadata."""
    if cases_path is None:
        cases_path = DATA_DIR / "processed" / "scdb_matched.parquet"
    
    cases = []
    
    if cases_path.exists() and cases_path.suffix == ".parquet":
        try:
            import pandas as pd
            df = pd.read_parquet(cases_path)
            cases = df.to_dict("records")
        except ImportError:
            pass
    elif cases_path.with_suffix(".json").exists():
        with open(cases_path.with_suffix(".json")) as f:
            cases = json.load(f)
    
    return cases


def create_networkx_graph(
    edges: List[Dict[str, Any]],
    cases: Optional[List[Dict[str, Any]]] = None,
):
    """Create NetworkX graph from edges."""
    import networkx as nx
    
    G = nx.DiGraph()
    
    # Add edges
    for edge in edges:
        G.add_edge(
            edge["source"],
            edge["target"],
            weight=edge.get("weight", 1.0),
        )
    
    # Add node attributes if cases provided
    if cases:
        case_dict = {c.get("id", c.get("case_id", "")): c for c in cases}
        for node in G.nodes():
            if node in case_dict:
                case = case_dict[node]
                G.nodes[node]["name"] = case.get("name", "")[:50]
                G.nodes[node]["outcome"] = case.get("outcome", "unknown")
                G.nodes[node]["year"] = case.get("year", case.get("date", "")[:4])
    
    return G


def compute_graph_stats(G) -> Dict[str, Any]:
    """Compute graph statistics."""
    import networkx as nx
    
    stats = {
        "num_nodes": G.number_of_nodes(),
        "num_edges": G.number_of_edges(),
        "density": nx.density(G),
        "avg_in_degree": sum(d for n, d in G.in_degree()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
        "avg_out_degree": sum(d for n, d in G.out_degree()) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0,
    }
    
    # Connected components (for undirected version)
    G_undirected = G.to_undirected()
    stats["num_components"] = nx.number_connected_components(G_undirected)
    
    # Largest component
    largest_cc = max(nx.connected_components(G_undirected), key=len)
    stats["largest_component_size"] = len(largest_cc)
    stats["largest_component_ratio"] = len(largest_cc) / G.number_of_nodes() if G.number_of_nodes() > 0 else 0
    
    # Top cited cases (highest in-degree)
    in_degrees = sorted(G.in_degree(), key=lambda x: x[1], reverse=True)[:10]
    stats["top_cited"] = [{"case_id": n, "citations": d} for n, d in in_degrees]
    
    return stats


def plot_citation_network(
    G,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 12),
    node_size_factor: float = 100,
    show_labels: bool = False,
    color_by_outcome: bool = True,
    title: str = "Supreme Court Citation Network",
):
    """
    Plot citation network using matplotlib.
    
    Args:
        G: NetworkX graph
        output_path: Path to save figure
        figsize: Figure size
        node_size_factor: Multiplier for node sizes (based on degree)
        show_labels: Whether to show node labels
        color_by_outcome: Color nodes by case outcome
        title: Plot title
    """
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np
    
    fig, ax = plt.subplots(figsize=figsize)
    
    # Layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Node sizes based on in-degree
    in_degrees = dict(G.in_degree())
    node_sizes = [node_size_factor * (1 + in_degrees.get(n, 0)) for n in G.nodes()]
    
    # Node colors
    if color_by_outcome:
        color_map = {"petitioner": "#28a745", "respondent": "#dc3545", "unknown": "#6c757d"}
        node_colors = [
            color_map.get(G.nodes[n].get("outcome", "unknown"), "#6c757d")
            for n in G.nodes()
        ]
    else:
        node_colors = "#1f77b4"
    
    # Draw
    nx.draw_networkx_nodes(
        G, pos,
        node_size=node_sizes,
        node_color=node_colors,
        alpha=0.7,
        ax=ax,
    )
    
    nx.draw_networkx_edges(
        G, pos,
        edge_color="#cccccc",
        alpha=0.3,
        arrows=True,
        arrowsize=10,
        ax=ax,
    )
    
    if show_labels:
        labels = {n: G.nodes[n].get("name", n)[:15] for n in G.nodes()}
        nx.draw_networkx_labels(
            G, pos,
            labels=labels,
            font_size=6,
            ax=ax,
        )
    
    ax.set_title(title, fontsize=16, fontweight="bold")
    ax.axis("off")
    
    # Legend
    if color_by_outcome:
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#28a745', markersize=10, label='Petitioner wins'),
            Line2D([0], [0], marker='o', color='w', markerfacecolor='#dc3545', markersize=10, label='Respondent wins'),
        ]
        ax.legend(handles=legend_elements, loc='upper left')
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    
    return fig, ax


def plot_degree_distribution(
    G,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 5),
):
    """Plot in-degree and out-degree distributions."""
    import matplotlib.pyplot as plt
    import numpy as np
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # In-degree
    in_degrees = [d for n, d in G.in_degree()]
    axes[0].hist(in_degrees, bins=30, color="#1f77b4", alpha=0.7, edgecolor="black")
    axes[0].set_xlabel("In-degree (Citations Received)")
    axes[0].set_ylabel("Frequency")
    axes[0].set_title("In-degree Distribution")
    axes[0].axvline(np.mean(in_degrees), color="red", linestyle="--", label=f"Mean: {np.mean(in_degrees):.1f}")
    axes[0].legend()
    
    # Out-degree
    out_degrees = [d for n, d in G.out_degree()]
    axes[1].hist(out_degrees, bins=30, color="#ff7f0e", alpha=0.7, edgecolor="black")
    axes[1].set_xlabel("Out-degree (Citations Made)")
    axes[1].set_ylabel("Frequency")
    axes[1].set_title("Out-degree Distribution")
    axes[1].axvline(np.mean(out_degrees), color="red", linestyle="--", label=f"Mean: {np.mean(out_degrees):.1f}")
    axes[1].legend()
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    
    return fig, axes


def create_interactive_graph(
    G,
    output_path: Optional[Path] = None,
) -> str:
    """Create interactive HTML visualization using pyvis."""
    try:
        from pyvis.network import Network
    except ImportError:
        return "<p>pyvis not installed. Run: pip install pyvis</p>"
    
    # Create pyvis network
    net = Network(
        height="800px",
        width="100%",
        bgcolor="#ffffff",
        font_color="black",
        directed=True,
    )
    
    # Physics settings
    net.barnes_hut(gravity=-5000, central_gravity=0.3, spring_length=200)
    
    # Add nodes
    color_map = {"petitioner": "#28a745", "respondent": "#dc3545", "unknown": "#6c757d"}
    
    for node in G.nodes():
        outcome = G.nodes[node].get("outcome", "unknown")
        name = G.nodes[node].get("name", node)[:30]
        in_deg = G.in_degree(node)
        
        net.add_node(
            node,
            label=name,
            title=f"{name}\nOutcome: {outcome}\nCitations: {in_deg}",
            color=color_map.get(outcome, "#6c757d"),
            size=10 + in_deg * 2,
        )
    
    # Add edges
    for source, target in G.edges():
        net.add_edge(source, target, color="#cccccc")
    
    # Generate HTML
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        net.save_graph(str(output_path))
        print(f"Saved interactive graph to {output_path}")
        return str(output_path)
    
    return net.generate_html()


def visualize_case_neighborhood(
    G,
    case_id: str,
    depth: int = 2,
    output_path: Optional[Path] = None,
):
    """Visualize the citation neighborhood of a specific case."""
    import networkx as nx
    import matplotlib.pyplot as plt
    
    # Get ego graph (neighborhood)
    ego = nx.ego_graph(G, case_id, radius=depth, undirected=True)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    pos = nx.spring_layout(ego, k=1.5, seed=42)
    
    # Node colors - highlight the focal case
    node_colors = []
    for n in ego.nodes():
        if n == case_id:
            node_colors.append("#ffc107")  # Yellow for focal
        elif ego.nodes[n].get("outcome") == "petitioner":
            node_colors.append("#28a745")
        elif ego.nodes[n].get("outcome") == "respondent":
            node_colors.append("#dc3545")
        else:
            node_colors.append("#6c757d")
    
    nx.draw(
        ego, pos,
        node_color=node_colors,
        node_size=500,
        with_labels=True,
        font_size=8,
        edge_color="#cccccc",
        alpha=0.8,
        ax=ax,
    )
    
    case_name = G.nodes[case_id].get("name", case_id)[:40]
    ax.set_title(f"Citation Neighborhood: {case_name}", fontsize=14, fontweight="bold")
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        print(f"Saved figure to {output_path}")
    
    return fig, ax


def generate_all_visualizations(
    output_dir: Optional[Path] = None,
    edges_path: Optional[Path] = None,
    cases_path: Optional[Path] = None,
) -> Dict[str, Path]:
    """Generate all visualization files."""
    output_dir = Path(output_dir or RESULTS_DIR / "figures")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading data...")
    edges = load_citation_edges(edges_path)
    cases = load_cases(cases_path)
    
    if not edges:
        print("No edges found. Run citation extraction first.")
        return {}
    
    print(f"Loaded {len(edges)} edges and {len(cases)} cases")
    
    print("Creating NetworkX graph...")
    G = create_networkx_graph(edges, cases)
    
    outputs = {}
    
    # Graph stats
    print("Computing graph statistics...")
    stats = compute_graph_stats(G)
    stats_path = output_dir / "graph_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)
    outputs["stats"] = stats_path
    print(f"Stats: {stats['num_nodes']} nodes, {stats['num_edges']} edges")
    
    # Network plot
    print("Plotting citation network...")
    plot_path = output_dir / "citation_network.png"
    plot_citation_network(G, output_path=plot_path)
    outputs["network"] = plot_path
    
    # Degree distribution
    print("Plotting degree distribution...")
    dist_path = output_dir / "degree_distribution.png"
    plot_degree_distribution(G, output_path=dist_path)
    outputs["degree_dist"] = dist_path
    
    # Interactive graph
    print("Creating interactive graph...")
    try:
        html_path = output_dir / "interactive_graph.html"
        create_interactive_graph(G, output_path=html_path)
        outputs["interactive"] = html_path
    except Exception as e:
        print(f"Interactive graph failed: {e}")
    
    print(f"\nAll visualizations saved to {output_dir}")
    return outputs


# CLI entrypoint
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate citation graph visualizations")
    parser.add_argument("--output-dir", type=str, help="Output directory")
    parser.add_argument("--edges", type=str, help="Path to edges.csv")
    parser.add_argument("--cases", type=str, help="Path to cases file")
    
    args = parser.parse_args()
    
    generate_all_visualizations(
        output_dir=Path(args.output_dir) if args.output_dir else None,
        edges_path=Path(args.edges) if args.edges else None,
        cases_path=Path(args.cases) if args.cases else None,
    )
