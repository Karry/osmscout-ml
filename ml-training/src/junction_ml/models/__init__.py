"""
PyTorch Geometric models for junction lane suggestion prediction.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv, GraphConv, global_mean_pool, global_max_pool # type: ignore[import-untyped]
from torch_geometric.data import Data, Batch # type: ignore[import-untyped]
from typing import Dict, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


class JunctionGNN(nn.Module):
    """
    Graph Neural Network for predicting junction lane suggestions.
    
    This model uses graph convolutions to process junction topology and
    predicts suggestedFrom, suggestedTo, and suggestedTurn for each edge.
    """
    
    def __init__(self,
                 node_features: int = 2,  # lat, lon
                 edge_features: int = 16,  # length, laneCount, angle, oneway, route, type, + 10 lane turns
                 hidden_dim: int = 64,
                 num_layers: int = 3,
                 conv_type: str = 'gcn',  # 'gcn', 'gat', or 'graph'
                 dropout: float = 0.1,
                 use_edge_attr: bool = True):
        """
        Initialize the Junction GNN model.
        
        Args:
            node_features: Number of node features
            edge_features: Number of edge features
            hidden_dim: Hidden dimension size
            num_layers: Number of GNN layers
            conv_type: Type of graph convolution ('gcn', 'gat', 'graph')
            dropout: Dropout probability
            use_edge_attr: Whether to use edge attributes
        """
        super().__init__()
        
        self.node_features = node_features
        self.edge_features = edge_features
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.use_edge_attr = use_edge_attr
        self.dropout = dropout
        
        # Node embedding layer
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        
        # Edge embedding layer
        if use_edge_attr:
            self.edge_embedding = nn.Linear(edge_features, hidden_dim)
        
        # Graph convolution layers
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        
        for i in range(num_layers):
            if conv_type == 'gcn':
                conv = GCNConv(hidden_dim, hidden_dim)
            elif conv_type == 'gat':
                conv = GATConv(hidden_dim, hidden_dim // 8, heads=8, dropout=dropout)
            elif conv_type == 'graph':
                conv = GraphConv(hidden_dim, hidden_dim)
            else:
                raise ValueError(f"Unknown conv_type: {conv_type}")
            
            self.convs.append(conv)
            self.norms.append(nn.LayerNorm(hidden_dim))
        
        # Edge prediction heads
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2 + (hidden_dim if use_edge_attr else 0), hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Multi-task prediction heads
        self.suggested_from_head = nn.Linear(hidden_dim // 2, 1)
        self.suggested_to_head = nn.Linear(hidden_dim // 2, 1)
        self.suggested_turn_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """
        Forward pass of the model.
        
        Args:
            data: PyTorch Geometric Data object
            
        Returns:
            Dictionary with predictions for each task
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        batch = getattr(data, 'batch', None)
        
        # Node embeddings
        x = self.node_embedding(x)
        x = F.relu(x)
        
        # Edge embeddings
        if self.use_edge_attr and edge_attr is not None:
            edge_embed = self.edge_embedding(edge_attr)
            edge_embed = F.relu(edge_embed)
        
        # Graph convolutions
        for conv, norm in zip(self.convs, self.norms):
            x_new = conv(x, edge_index)
            x_new = norm(x_new)
            x_new = F.relu(x_new)
            x_new = F.dropout(x_new, p=self.dropout, training=self.training)
            x = x + x_new  # Residual connection
        
        # Edge-level predictions
        edge_predictions = self._predict_edges(x, edge_index, edge_embed if self.use_edge_attr and edge_attr is not None else None)
        
        return edge_predictions
    
    def _predict_edges(self, 
                      node_embeddings: torch.Tensor, 
                      edge_index: torch.Tensor,
                      edge_embeddings: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Predict edge-level outputs.
        
        Args:
            node_embeddings: Node embeddings [num_nodes, hidden_dim]
            edge_index: Edge indices [2, num_edges]
            edge_embeddings: Optional edge embeddings [num_edges, hidden_dim]
            
        Returns:
            Dictionary with edge predictions
        """
        row, col = edge_index
        
        # Concatenate source and target node embeddings
        edge_features = torch.cat([node_embeddings[row], node_embeddings[col]], dim=1)
        
        # Add edge embeddings if available
        if edge_embeddings is not None:
            edge_features = torch.cat([edge_features, edge_embeddings], dim=1)
        
        # Edge prediction
        edge_repr = self.edge_predictor(edge_features)
        
        # Multi-task predictions
        suggested_from = self.suggested_from_head(edge_repr).squeeze(-1)
        suggested_to = self.suggested_to_head(edge_repr).squeeze(-1)
        suggested_turn = self.suggested_turn_head(edge_repr).squeeze(-1)
        
        return {
            'suggested_from': suggested_from,
            'suggested_to': suggested_to,
            'suggested_turn': suggested_turn
        }


class JunctionTransformer(nn.Module):
    """
    Transformer-based model for junction analysis.
    
    Uses self-attention mechanisms to model relationships between
    nodes and edges in junction graphs.
    """
    
    def __init__(self,
                 node_features: int = 2,
                 edge_features: int = 16,
                 hidden_dim: int = 128,
                 num_heads: int = 8,
                 num_layers: int = 4,
                 dropout: float = 0.1):
        """
        Initialize the Junction Transformer model.
        
        Args:
            node_features: Number of node features
            edge_features: Number of edge features
            hidden_dim: Hidden dimension size
            num_heads: Number of attention heads
            num_layers: Number of transformer layers
            dropout: Dropout probability
        """
        super().__init__()
        
        self.hidden_dim = hidden_dim
        
        # Embeddings
        self.node_embedding = nn.Linear(node_features, hidden_dim)
        self.edge_embedding = nn.Linear(edge_features, hidden_dim)
        
        # Positional encoding for edges
        self.edge_pos_encoding = nn.Parameter(torch.randn(1000, hidden_dim))
        
        # Transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output heads
        self.output_projection = nn.Linear(hidden_dim, hidden_dim // 2)
        self.suggested_from_head = nn.Linear(hidden_dim // 2, 1)
        self.suggested_to_head = nn.Linear(hidden_dim // 2, 1)
        self.suggested_turn_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, data: Data) -> Dict[str, torch.Tensor]:
        """Forward pass of the transformer model."""
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr
        num_edges = edge_index.size(1)
        
        # Create edge representations by combining node pairs
        row, col = edge_index
        edge_node_features = torch.cat([x[row], x[col]], dim=1)
        
        # Combine with edge attributes
        if edge_attr is not None:
            edge_input = torch.cat([edge_node_features, edge_attr], dim=1)
        else:
            edge_input = edge_node_features
        
        # Project to hidden dimension
        edge_embeddings = self.edge_embedding(edge_input)
        
        # Add positional encoding
        pos_ids = torch.arange(num_edges, device=edge_embeddings.device)
        pos_embeddings = self.edge_pos_encoding[pos_ids % 1000]
        edge_embeddings = edge_embeddings + pos_embeddings
        
        # Add batch dimension for transformer
        edge_embeddings = edge_embeddings.unsqueeze(0)
        
        # Transformer processing
        transformed = self.transformer(edge_embeddings)
        transformed = transformed.squeeze(0)
        
        # Output projections
        output_repr = self.output_projection(transformed)
        output_repr = F.relu(output_repr)
        
        # Multi-task predictions
        suggested_from = self.suggested_from_head(output_repr).squeeze(-1)
        suggested_to = self.suggested_to_head(output_repr).squeeze(-1)
        suggested_turn = self.suggested_turn_head(output_repr).squeeze(-1)
        
        return {
            'suggested_from': suggested_from,
            'suggested_to': suggested_to,
            'suggested_turn': suggested_turn
        }


def create_model(model_type: str = 'gnn', **kwargs: Any) -> nn.Module:
    """
    Create a model instance.
    
    Args:
        model_type: Type of model ('gnn' or 'transformer')
        **kwargs: Additional arguments for model initialization
        
    Returns:
        Model instance
    """
    if model_type == 'gnn':
        return JunctionGNN(**kwargs)
    elif model_type == 'transformer':
        return JunctionTransformer(**kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
