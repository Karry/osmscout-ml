"""
Data loading and preprocessing utilities for junction graphs.
"""
import json
import os
from pathlib import Path
from typing import List, Dict, Any, Tuple, Optional, Callable
import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data, Dataset # type: ignore[import-untyped]
from sklearn.preprocessing import StandardScaler, LabelEncoder  # type: ignore[import-untyped]
import logging

logger = logging.getLogger(__name__)

EdgeFeatureCount = 9
"""
Number of edge features for lane-level representation:
1. length
2. laneCount (total lanes on highway)
3. angle (turn angle at junction)
4. route (highway is part of route)
5. type (highway type)
6. usable (lane/highway is usable)
7. virtual (virtual reverse edge)
8. relativeLanePosition (0.0=left, 1.0=right)
9. laneTurn (turn direction for this lane)
"""

NodeFeatureCount = 4
"""
Number of node features:
lat, lon (normalized), incoming edge count, outgoing edge count
"""


class JunctionGraphDataset(Dataset):
    """
    PyTorch Geometric dataset for junction graphs.

    Loads JSON files containing junction graph data and converts them to
    PyTorch Geometric Data objects for training.
    """

    def __init__(self,
                 data_dir: Optional[str] = None,
                 transform: Optional[Callable] = None,
                 pre_transform: Optional[Callable] = None,
                 pre_filter: Optional[Callable] = None) -> None:
        """
        Initialize the dataset.

        Args:
            data_dir: Directory containing JSON files with junction graphs
            transform: Optional transform to apply to each data object
            pre_transform: Optional transform to apply before caching
            pre_filter: Optional filter to apply before caching
        """
        self.data_dir = Path(data_dir) if data_dir else None
        self.feature_scaler = StandardScaler()
        self.label_encoders: dict[str, Any] = {}
        self.feature_names: list[str] = []

        super().__init__(data_dir, transform, pre_transform, pre_filter)

    @property
    def raw_file_names(self) -> List[str]:
        """Get list of raw JSON files."""
        if not self.data_dir or not self.data_dir.exists():
            return []
        return [f.name for f in self.data_dir.glob("*.json")]

    @property
    def processed_file_names(self) -> List[str]:
        """Get list of processed files."""
        return ['data.pt', 'feature_info.pt']

    def download(self) -> None:
        """Download raw data (not needed as files already exist)."""
        pass

    def process(self) -> None:
        """Process raw JSON files into PyTorch Geometric format."""
        if not self.data_dir:
            logger.info(f"Not loading data as no data directory is specified.")
            return
        logger.info(f"Processing {len(self.raw_file_names)} junction files...")

        data_list = []
        all_features = []

        for json_file in self.raw_file_names:
            file_path = self.data_dir / json_file
            try:
                with open(file_path, 'r') as f:
                    graph_data = json.load(f)

                data = self._convert_to_pyg_data(graph_data)
                if data is not None:
                    data_list.append(data)
                    # Collect features for normalization
                    if data.x is not None:
                        all_features.append(data.x.numpy())

            except Exception as e:
                logger.warning(f"Failed to process {json_file}: {e}")
                continue

        if not data_list:
            raise ValueError("No valid junction graphs found!")

        # Calculate class weights for handling imbalance
        total_positive = 0
        total_negative = 0
        for data in data_list:
            if hasattr(data, 'y_suggested'):
                positive = (data.y_suggested > 0.5).sum().item()
                total_positive += positive
                total_negative += len(data.y_suggested) - positive

        pos_weight = total_negative / max(1, total_positive) if total_positive > 0 else 1.0
        logger.info(f"Dataset statistics: {total_positive} suggested lanes, {total_negative} non-suggested lanes")
        logger.info(f"Calculated pos_weight for BCEWithLogitsLoss: {pos_weight:.2f}")

        # Fit feature scaler on all features
        if all_features:
            all_features_array = np.vstack(all_features)
            self.feature_scaler.fit(all_features_array)

            # Apply scaling to all data
            for data in data_list:
                if data.x is not None:
                    data.x = torch.tensor(
                        self.feature_scaler.transform(data.x.numpy()),
                        dtype=torch.float32
                    )

        # Save processed data
        torch.save(data_list, self.processed_paths[0])
        torch.save({
            'feature_scaler': self.feature_scaler,
            'label_encoders': self.label_encoders,
            'feature_names': self.feature_names,
            'pos_weight': pos_weight
        }, self.processed_paths[1])

        logger.info(f"Processed {len(data_list)} junction graphs successfully")

    def _convert_to_pyg_data(self, graph_data: Dict[str, Any]) -> Optional[Data]:
        """
        Convert JSON graph data to PyTorch Geometric Data object.

        Args:
            graph_data: Dictionary containing nodes and edges

        Returns:
            PyTorch Geometric Data object or None if invalid
        """
        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])

        if not nodes or not edges:
            return None

        # Create node mapping
        node_id_to_idx = {node['id']: idx for idx, node in enumerate(nodes)}

        # Node features (coordinates)
        node_features = []
        for node in nodes:
            features = [node['normLat'], node['normLon'], node['incoming'], node['outgoing']]
            assert(len(features) == NodeFeatureCount), f"Expected {NodeFeatureCount} node features, got {len(features)}"
            node_features.append(features)

        x = torch.tensor(node_features, dtype=torch.float32)

        # Edge indices and features
        edge_indices = []
        edge_features = []
        edge_labels: list[float] = []  # Single binary label: suggested or not

        for edge in edges:
            from_idx = node_id_to_idx.get(edge['from'])
            to_idx = node_id_to_idx.get(edge['to'])

            if from_idx is None or to_idx is None:
                continue

            edge_indices.append([from_idx, to_idx])

            # Extract new lane-level edge features (9 features)
            features = [
                edge.get('length', 0.0),           # 1. length
                edge.get('laneCount', 0.0),        # 2. laneCount (total lanes on highway)
                edge.get('angle', 0.0),            # 3. angle
                edge.get('route', 0.0),            # 4. route (part of route)
                edge.get('type', -1.0),            # 5. type (highway type)
                edge.get('usable', 0.0),           # 6. usable
                edge.get('virtual', 0.0),          # 7. virtual
                edge.get('relativeLanePosition', 0.0),  # 8. relativeLanePosition (0.0=left, 1.0=right)
                edge.get('laneTurn', -1.0)         # 9. laneTurn (single value for this lane)
            ]

            assert(len(features) == EdgeFeatureCount), f"Expected {EdgeFeatureCount} edge features, got {len(features)}"

            edge_features.append(features)

            # Extract single binary target label: is this lane suggested?
            edge_labels.append(edge.get('suggested', 0.0))

        if not edge_indices:
            return None

        edge_index = torch.tensor(edge_indices, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(edge_features, dtype=torch.float32)

        # Convert single binary label to tensor
        y_suggested = torch.tensor(edge_labels, dtype=torch.float32)

        # Create mask for edges that are part of the route (have valid labels)
        # Virtual edges and non-route edges will have suggested=0.0 but are still valid
        valid_suggested = torch.ones_like(y_suggested, dtype=torch.bool)

        return Data(
            x=x,
            edge_index=edge_index,
            edge_attr=edge_attr,
            y_suggested=y_suggested,
            valid_suggested=valid_suggested,
            num_nodes=len(nodes)
        )

    def len(self) -> int:
        """Get dataset length."""
        if not hasattr(self, '_data_list'):
            self._data_list = torch.load(self.processed_paths[0], weights_only=False)
        return len(self._data_list)

    def get(self, idx: int) -> Data:
        """Get data object at index."""
        if not hasattr(self, '_data_list'):
            self._data_list = torch.load(self.processed_paths[0], weights_only=False)
        return self._data_list[idx]


def load_junction_data(data_dir: str = "../tmp-junctions") -> JunctionGraphDataset:
    """
    Load and preprocess junction graph data.

    Args:
        data_dir: Directory containing JSON files

    Returns:
        JunctionGraphDataset instance
    """
    return JunctionGraphDataset(data_dir)


def create_train_val_test_split(dataset: JunctionGraphDataset,
                               train_ratio: float = 0.7,
                               val_ratio: float = 0.15,
                               test_ratio: float = 0.15,
                               random_seed: int = 42) -> Tuple[Dataset, Dataset, Dataset]:
    """
    Split dataset into train, validation, and test sets.

    Args:
        dataset: JunctionGraphDataset instance
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        random_seed: Random seed for reproducibility

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
        "Ratios must sum to 1.0"

    torch.manual_seed(random_seed)
    dataset_size = len(dataset)

    train_size = int(train_ratio * dataset_size)
    val_size = int(val_ratio * dataset_size)
    test_size = dataset_size - train_size - val_size

    split: list[Dataset] = torch.utils.data.random_split(
        dataset, [train_size, val_size, test_size]
    )
    assert len(split) == 3
    return (split[0], split[1], split[2])


def analyze_dataset_statistics(dataset: JunctionGraphDataset) -> Dict[str, Any]:
    """
    Analyze and return statistics about the dataset.

    Args:
        dataset: JunctionGraphDataset instance

    Returns:
        Dictionary containing dataset statistics
    """
    stats = {
        'total_graphs': len(dataset),
        'node_counts': [],
        'edge_counts': [],
        'feature_stats': {},
        'label_distributions': {}
    }

    all_suggested_from = []
    all_suggested_to = []
    all_suggested_turn = []

    for i in range(len(dataset)):
        data = dataset[i]

        assert type(stats['node_counts']) == list
        assert type(stats['edge_counts']) == list
        stats['node_counts'].append(data.num_nodes)
        stats['edge_counts'].append(data.edge_index.size(1))

        # Collect valid labels
        valid_from = data.y_suggested_from[data.valid_from]
        valid_to = data.y_suggested_to[data.valid_to]
        valid_turn = data.y_suggested_turn[data.valid_turn]

        all_suggested_from.extend(valid_from.tolist())
        all_suggested_to.extend(valid_to.tolist())
        all_suggested_turn.extend(valid_turn.tolist())

    assert type(stats['node_counts']) == list
    assert type(stats['edge_counts']) == list
    stats['avg_nodes'] = np.mean(stats['node_counts'])
    stats['avg_edges'] = np.mean(stats['edge_counts'])
    stats['total_valid_labels'] = {
        'suggestedFrom': len(all_suggested_from),
        'suggestedTo': len(all_suggested_to),
        'suggestedTurn': len(all_suggested_turn)
    }

    return stats
