#!/bin/env python3
import argparse
import random

import torch
from torch_geometric.data import DataLoader # type: ignore[import-untyped]

from junction_ml.data import JunctionGraphDataset
from junction_ml.models import JunctionGNN
from junction_ml.training import create_trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train junction lane suggestion model.")
    parser.add_argument('--data-dir', type=str, default="../../tmp-junctions", help="Directory with junction graph JSON files")
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    torch.manual_seed(args.seed)
    random.seed(args.seed)

    # Load dataset
    dataset = JunctionGraphDataset(data_dir=args.data_dir)
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
        node_features=2,
        edge_features=16,
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
        log_dir=args.log_dir,
        save_dir=args.save_dir
    )

    # Train
    trainer.train(num_epochs=args.epochs)


if __name__ == "__main__":
    main()
