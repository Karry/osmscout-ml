#!/bin/env python3
import argparse
import random
import logging

import torch
from torch_geometric.data import DataLoader # type: ignore[import-untyped]

from junction_ml.data import JunctionGraphDataset, EdgeFeatureCount, NodeFeatureCount
from junction_ml.models import JunctionGNN
from junction_ml.training import create_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train junction lane suggestion model.")
    parser.add_argument('--data-dir', type=str, default="../tmp-junctions", help="Directory with junction graph JSON files")
    parser.add_argument('--log-dir', type=str, default="runs", help="Tensorboard log directory")
    parser.add_argument('--save-dir', type=str, default="checkpoints", help="Model checkpoint directory")
    parser.add_argument('--batch-size', type=int, default=32, help="Batch size")
    parser.add_argument('--epochs', type=int, default=50, help="Number of training epochs")
    parser.add_argument('--learning-rate', type=float, default=1e-3, help="Learning rate")
    parser.add_argument('--weight-decay', type=float, default=1e-5, help="Weight decay")
    parser.add_argument('--hidden-dim', type=int, default=64, help="Model hidden dimension")
    parser.add_argument('--num-layers', type=int, default=3, help="Number of GNN layers")
    parser.add_argument('--dropout', type=float, default=0.1, help="Dropout rate")
    parser.add_argument('--val-ratio', type=float, default=0.1, help="Validation split ratio")
    parser.add_argument('--seed', type=int, default=42, help="Random seed")
    parser.add_argument('--loss', type=str, default='bce', choices=['bce', 'focal', 'dice'],
                        help="Loss function: 'bce' (default), 'focal', or 'dice'")
    parser.add_argument('--focal-alpha', type=float, default=0.25,
                        help="Focal loss alpha (positive class balance, 0-1). Default 0.25")
    parser.add_argument('--focal-gamma', type=float, default=2.0,
                        help="Focal loss gamma (focusing parameter, >=0). Default 2.0")
    parser.add_argument('--dice-smooth', type=float, default=1.0,
                        help="Dice loss smoothing constant. Default 1.0")
    parser.add_argument('--dice-bce-weight', type=float, default=0.5,
                        help="Weight of auxiliary BCE term in Dice loss (0=pure Dice). Default 0.5")
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose output')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Load dataset
    dataset = JunctionGraphDataset(data_dir=args.data_dir)

    # Load pos_weight from dataset metadata
    feature_info = torch.load(dataset.processed_paths[1], weights_only=False)
    pos_weight = feature_info.get('pos_weight', 1.0)
    print(f"Using pos_weight={pos_weight:.2f} for handling class imbalance")

    # Validate that dataset is in new format
    if len(dataset) > 0:
        sample = dataset[0]
        if hasattr(sample, 'edge_attr') and sample.edge_attr is not None:
            num_features = sample.edge_attr.shape[1] if len(sample.edge_attr.shape) > 1 else sample.edge_attr.shape[0]
            if num_features != EdgeFeatureCount:
                print(f"\n⚠️  WARNING: Dataset has {num_features} edge features, expected {EdgeFeatureCount}")
                print("⚠️  Your dataset appears to be in the OLD format!")
                print("⚠️  Please regenerate the dataset using the updated JunctionGraphExport tool:")
                print("    ./cmake-build-debug/JunctionGraphExport --osm-data /path/to/map.osm --output tmp-junctions")
                print("⚠️  Training will continue but may not work correctly.\n")

    indices = list(range(len(dataset)))
    random.shuffle(indices)
    val_size = int(len(indices) * args.val_ratio)
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    train_dataset = dataset[train_indices]
    val_dataset = dataset[val_indices]

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

    # Build model
    model = JunctionGNN(
        node_features=NodeFeatureCount,
        edge_features=EdgeFeatureCount,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=args.dropout
    )

    # Trainer
    trainer = create_trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        pos_weight=pos_weight,
        loss_type=args.loss,
        focal_alpha=args.focal_alpha,
        focal_gamma=args.focal_gamma,
        dice_smooth=args.dice_smooth,
        dice_bce_weight=args.dice_bce_weight,
        log_dir=args.log_dir,
        save_dir=args.save_dir
    )

    # Train
    trainer.train(num_epochs=args.epochs)

    # Save final TorchScript model
    trainer.save_final_torchscript()


if __name__ == "__main__":
    main()
