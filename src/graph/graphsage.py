"""GraphSAGE model for learning case embeddings from citation graph."""

from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import NeighborLoader
from torch_geometric.nn import SAGEConv
from tqdm import tqdm

from src.config import (
    GRAPHSAGE_HIDDEN_DIM,
    GRAPHSAGE_NUM_LAYERS,
    EMBEDDINGS_DIR,
)


class GraphSAGE(nn.Module):
    """
    GraphSAGE model for node embedding learning.

    Uses mean aggregation over neighbors with multi-layer message passing.
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = GRAPHSAGE_HIDDEN_DIM,
        out_channels: int = GRAPHSAGE_HIDDEN_DIM,
        num_layers: int = GRAPHSAGE_NUM_LAYERS,
        dropout: float = 0.3,
    ):
        """
        Initialize GraphSAGE model.

        Args:
            in_channels: Input feature dimension
            hidden_channels: Hidden layer dimension
            out_channels: Output embedding dimension
            num_layers: Number of GraphSAGE layers
            dropout: Dropout probability
        """
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        self.convs = nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through GraphSAGE layers.

        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]

        Returns:
            Node embeddings [num_nodes, out_channels]
        """
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.convs[-1](x, edge_index)
        return x

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Get normalized embeddings for retrieval."""
        self.eval()
        with torch.no_grad():
            embeddings = self.forward(x, edge_index)
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


class LinkPredictor(nn.Module):
    """Link prediction head for training GraphSAGE on citation prediction."""

    def __init__(self, hidden_channels: int):
        super().__init__()
        self.lin1 = nn.Linear(hidden_channels * 2, hidden_channels)
        self.lin2 = nn.Linear(hidden_channels, 1)

    def forward(self, z_src: torch.Tensor, z_dst: torch.Tensor) -> torch.Tensor:
        """
        Predict link probability between source and destination nodes.

        Args:
            z_src: Source node embeddings
            z_dst: Destination node embeddings

        Returns:
            Link probabilities
        """
        z = torch.cat([z_src, z_dst], dim=-1)
        z = F.relu(self.lin1(z))
        z = self.lin2(z)
        return torch.sigmoid(z).squeeze(-1)


def build_graph_data(
    node_features: np.ndarray,
    edge_index: np.ndarray,
    node_ids: list[str],
) -> Data:
    """
    Build PyTorch Geometric Data object from arrays.

    Args:
        node_features: Node feature matrix [num_nodes, feature_dim]
        edge_index: Edge indices [2, num_edges]
        node_ids: List of node IDs

    Returns:
        PyTorch Geometric Data object
    """
    data = Data(
        x=torch.tensor(node_features, dtype=torch.float),
        edge_index=torch.tensor(edge_index, dtype=torch.long),
    )
    data.node_ids = node_ids
    data.num_nodes = len(node_ids)
    return data


def load_graph_from_neo4j(driver=None) -> Data:
    """
    Load graph data from Neo4j.

    Args:
        driver: Neo4j driver instance

    Returns:
        PyTorch Geometric Data object
    """
    from neo4j import GraphDatabase
    from src.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

    close_driver = False
    if driver is None:
        driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
        close_driver = True

    try:
        with driver.session() as session:
            # Get all nodes with embeddings
            node_result = session.run("""
                MATCH (c:Case)
                WHERE c.embedding IS NOT NULL
                RETURN c.id AS id, c.embedding AS embedding
                ORDER BY c.id
            """)
            nodes = list(node_result)

            if not nodes:
                raise ValueError("No nodes with embeddings found in Neo4j")

            node_ids = [n["id"] for n in nodes]
            node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}
            node_features = np.array([n["embedding"] for n in nodes])

            # Get all edges
            edge_result = session.run("""
                MATCH (s:Case)-[r:CITES]->(t:Case)
                WHERE s.embedding IS NOT NULL AND t.embedding IS NOT NULL
                RETURN s.id AS source, t.id AS target
            """)
            edges = list(edge_result)

            # Build edge index
            src_indices = []
            dst_indices = []
            for e in edges:
                src_id = e["source"]
                dst_id = e["target"]
                if src_id in node_id_to_idx and dst_id in node_id_to_idx:
                    src_indices.append(node_id_to_idx[src_id])
                    dst_indices.append(node_id_to_idx[dst_id])

            edge_index = np.array([src_indices, dst_indices])

            return build_graph_data(node_features, edge_index, node_ids)

    finally:
        if close_driver:
            driver.close()


class GraphSAGETrainer:
    """
    Trainer class for GraphSAGE link prediction.

    Provides a clean interface for training and embedding extraction.
    """

    def __init__(self, model: GraphSAGE, device: str = "cpu"):
        """
        Initialize the trainer.

        Args:
            model: GraphSAGE model instance
            device: Device to use for training
        """
        self.model = model.to(device)
        self.device = device
        self.link_predictor = LinkPredictor(model.convs[-1].out_channels).to(device)

    def train_link_prediction(self, data: Data, epochs: int = 100, lr: float = 0.01) -> list:
        """
        Train the model on link prediction task.

        Args:
            data: PyTorch Geometric Data object
            epochs: Number of training epochs
            lr: Learning rate

        Returns:
            List of training losses
        """
        optimizer = torch.optim.Adam(
            list(self.model.parameters()) + list(self.link_predictor.parameters()),
            lr=lr,
        )

        data = data.to(self.device)
        num_edges = data.edge_index.shape[1]
        perm = torch.randperm(num_edges)
        train_edges = data.edge_index[:, perm[: int(0.8 * num_edges)]].to(self.device)

        loss_history = []
        for epoch in range(epochs):
            self.model.train()
            self.link_predictor.train()

            # Sample negative edges
            neg_src = torch.randint(0, data.num_nodes, (train_edges.shape[1],), device=self.device)
            neg_dst = torch.randint(0, data.num_nodes, (train_edges.shape[1],), device=self.device)

            z = self.model(data.x, data.edge_index)

            pos_score = self.link_predictor(z[train_edges[0]], z[train_edges[1]])
            neg_score = self.link_predictor(z[neg_src], z[neg_dst])

            loss = (
                F.binary_cross_entropy(pos_score, torch.ones_like(pos_score)) +
                F.binary_cross_entropy(neg_score, torch.zeros_like(neg_score))
            )

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            loss_history.append(loss.item())

            if epoch % 10 == 0:
                print(f"Epoch {epoch}: loss={loss.item():.4f}")

        return loss_history

    def get_embeddings(self, data: Data) -> torch.Tensor:
        """
        Get node embeddings from the trained model.

        Args:
            data: PyTorch Geometric Data object

        Returns:
            Node embeddings tensor
        """
        return self.model.get_embeddings(data.x.to(self.device), data.edge_index.to(self.device))

    def save_embeddings(self, embeddings: torch.Tensor, path: Path):
        """
        Save embeddings to disk.

        Args:
            embeddings: Embeddings tensor
            path: Path to save to
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(embeddings.cpu(), path)
        print(f"Saved embeddings to {path}")


def train_graphsage(
    data: Data,
    epochs: int = 100,
    lr: float = 0.01,
    hidden_dim: int = GRAPHSAGE_HIDDEN_DIM,
    num_layers: int = GRAPHSAGE_NUM_LAYERS,
    batch_size: int = 512,
    num_neighbors: list[int] = None,
    device: str = "auto",
) -> tuple[GraphSAGE, list[float]]:
    """
    Train GraphSAGE model on link prediction task.

    Args:
        data: PyTorch Geometric Data object
        epochs: Number of training epochs
        lr: Learning rate
        hidden_dim: Hidden dimension
        num_layers: Number of GraphSAGE layers
        batch_size: Training batch size
        num_neighbors: Number of neighbors to sample per layer
        device: Device to use ('auto', 'cuda', 'cpu')

    Returns:
        Trained model and loss history
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if num_neighbors is None:
        num_neighbors = [15, 10]  # Sample 15 neighbors in first layer, 10 in second

    print(f"Training on {device}")
    print(f"Nodes: {data.num_nodes}, Edges: {data.edge_index.shape[1]}")

    # Initialize model
    model = GraphSAGE(
        in_channels=data.x.shape[1],
        hidden_channels=hidden_dim,
        out_channels=hidden_dim,
        num_layers=num_layers,
    ).to(device)

    link_predictor = LinkPredictor(hidden_dim).to(device)

    optimizer = torch.optim.Adam(
        list(model.parameters()) + list(link_predictor.parameters()),
        lr=lr,
    )

    # Create train/val edge split
    num_edges = data.edge_index.shape[1]
    perm = torch.randperm(num_edges)
    train_edges = data.edge_index[:, perm[: int(0.8 * num_edges)]]
    val_edges = data.edge_index[:, perm[int(0.8 * num_edges) :]]

    # Create negative edges for training
    def sample_negative_edges(edge_index: torch.Tensor, num_nodes: int, num_neg: int) -> torch.Tensor:
        neg_src = torch.randint(0, num_nodes, (num_neg,))
        neg_dst = torch.randint(0, num_nodes, (num_neg,))
        return torch.stack([neg_src, neg_dst])

    data = data.to(device)
    train_edges = train_edges.to(device)
    val_edges = val_edges.to(device)

    loss_history = []
    best_val_loss = float("inf")
    best_model_state = None

    for epoch in range(epochs):
        model.train()
        link_predictor.train()

        # Sample negative edges
        neg_edges = sample_negative_edges(
            train_edges, data.num_nodes, train_edges.shape[1]
        ).to(device)

        # Forward pass
        z = model(data.x, data.edge_index)

        # Positive edges
        pos_score = link_predictor(z[train_edges[0]], z[train_edges[1]])
        pos_loss = F.binary_cross_entropy(pos_score, torch.ones_like(pos_score))

        # Negative edges
        neg_score = link_predictor(z[neg_edges[0]], z[neg_edges[1]])
        neg_loss = F.binary_cross_entropy(neg_score, torch.zeros_like(neg_score))

        loss = pos_loss + neg_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Validation
        if epoch % 10 == 0:
            model.eval()
            link_predictor.eval()
            with torch.no_grad():
                z = model(data.x, data.edge_index)

                val_neg_edges = sample_negative_edges(
                    val_edges, data.num_nodes, val_edges.shape[1]
                ).to(device)

                val_pos_score = link_predictor(z[val_edges[0]], z[val_edges[1]])
                val_neg_score = link_predictor(z[val_neg_edges[0]], z[val_neg_edges[1]])

                val_loss = (
                    F.binary_cross_entropy(val_pos_score, torch.ones_like(val_pos_score))
                    + F.binary_cross_entropy(val_neg_score, torch.zeros_like(val_neg_score))
                ).item()

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_model_state = model.state_dict().copy()

            print(f"Epoch {epoch:3d}: train_loss={loss.item():.4f}, val_loss={val_loss:.4f}")

        loss_history.append(loss.item())

    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    return model, loss_history


def save_embeddings(
    model: GraphSAGE,
    data: Data,
    output_path: Path = EMBEDDINGS_DIR / "graphsage_embeddings.pt",
    device: str = "auto",
):
    """
    Generate and save GraphSAGE embeddings.

    Args:
        model: Trained GraphSAGE model
        data: Graph data
        output_path: Path to save embeddings
        device: Device to use
    """
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    model = model.to(device)
    data = data.to(device)

    embeddings = model.get_embeddings(data.x, data.edge_index)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "node_ids": data.node_ids,
            "embeddings": embeddings.cpu(),
        },
        output_path,
    )
    print(f"Saved embeddings to {output_path}")


def load_embeddings(
    embedding_path: Path = EMBEDDINGS_DIR / "graphsage_embeddings.pt",
) -> tuple[list[str], torch.Tensor]:
    """
    Load saved GraphSAGE embeddings.

    Args:
        embedding_path: Path to saved embeddings

    Returns:
        Tuple of (node_ids, embeddings tensor)
    """
    data = torch.load(embedding_path)
    return data["node_ids"], data["embeddings"]


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python graphsage.py [train|export]")
        print("  train: Train GraphSAGE on the citation graph")
        print("  export: Export trained embeddings to Neo4j")
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "train":
        print("Loading graph from Neo4j...")
        data = load_graph_from_neo4j()

        print(f"Training GraphSAGE...")
        model, loss_history = train_graphsage(data, epochs=100)

        print("Saving embeddings...")
        save_embeddings(model, data)

        print("Done!")

    elif command == "export":
        from src.graph.loader import update_embeddings

        print("Updating Neo4j with GraphSAGE embeddings...")
        updated = update_embeddings()
        print(f"Updated {updated} nodes.")

    else:
        print(f"Unknown command: {command}")
        sys.exit(1)
