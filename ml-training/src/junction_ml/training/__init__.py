"""
Training utilities and loops for junction lane prediction models.
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader # type: ignore[import-untyped]
from torch.utils.tensorboard import SummaryWriter
from typing import Dict, List, Optional, Tuple, Any
import numpy as np
from tqdm import tqdm
import logging
import os
from pathlib import Path
import json
from junction_ml.models import JunctionGNN

logger = logging.getLogger(__name__)


class BinarySuggestedLoss(nn.Module):
    """
    Binary classification loss for lane suggestion prediction.

    Uses weighted BCEWithLogitsLoss to handle class imbalance between
    suggested and non-suggested lanes.
    """
    
    def __init__(self, pos_weight: float = 1.0):
        """
        Initialize binary loss.

        Args:
            pos_weight: Weight for positive class (suggested lanes) to handle imbalance
        """
        super().__init__()
        
        self.pos_weight = torch.tensor([pos_weight])
        self.bce_loss = nn.BCEWithLogitsLoss(pos_weight=self.pos_weight, reduction='none')

        logger.info(f"Initialized BinarySuggestedLoss with pos_weight={pos_weight:.2f}")

    def forward(self, 
                predictions: torch.Tensor,
                targets: torch.Tensor,
                valid_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Compute binary classification loss.

        Args:
            predictions: Model predictions (logits) [num_edges]
            targets: Ground truth binary labels [num_edges]
            valid_mask: Optional mask indicating valid labels [num_edges]

        Returns:
            Scalar loss value
        """
        if valid_mask is None:
            valid_mask = torch.ones_like(targets, dtype=torch.bool)

        if valid_mask.sum() == 0:
            return torch.tensor(0.0, device=predictions.device)

        # Move pos_weight to same device as predictions
        if self.pos_weight.device != predictions.device:
            self.pos_weight = self.pos_weight.to(predictions.device)
            self.bce_loss.pos_weight = self.pos_weight

        # Compute BCE loss only for valid edges
        loss = self.bce_loss(predictions[valid_mask], targets[valid_mask])

        return loss.mean()


class JunctionTrainer:
    """
    Trainer class for junction lane prediction models.
    """
    
    def __init__(self,
                 model: JunctionGNN,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: optim.Optimizer,
                 criterion: BinarySuggestedLoss,
                 device: torch.device,
                 log_dir: str = 'runs',
                 save_dir: str = 'checkpoints',
                 patience: int = 10):
        """
        Initialize trainer.
        
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader
            optimizer: Optimizer for training
            criterion: Loss function
            device: Device to train on
            log_dir: Directory for tensorboard logs
            save_dir: Directory for model checkpoints
            patience: Early stopping patience
        """
        self.model: JunctionGNN = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.patience = patience
        
        # Create directories
        os.makedirs(log_dir, exist_ok=True)
        os.makedirs(save_dir, exist_ok=True)
        
        self.log_dir = Path(log_dir)
        self.save_dir = Path(save_dir)
        
        # Tensorboard writer
        self.writer = SummaryWriter(log_dir)
        
        # Training state
        self.epoch = 0
        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.train_losses: list[Any] = []
        self.val_losses: list[Any] = []
        
    def train_epoch(self) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        epoch_losses: list[float] = []
        all_preds: list[float] = []
        all_targets: list[float] = []

        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch} [Train]')
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch)  # [num_edges] logits

            # Get targets and valid mask
            targets = batch.y_suggested
            valid_mask = batch.valid_suggested

            # Compute loss
            loss = self.criterion(predictions, targets, valid_mask)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Log losses and predictions
            epoch_losses.append(loss.item())

            # Collect predictions for metrics (apply sigmoid)
            with torch.no_grad():
                pred_probs = torch.sigmoid(predictions[valid_mask])
                all_preds.extend(pred_probs.cpu().numpy().tolist())
                all_targets.extend(targets[valid_mask].cpu().numpy().tolist())

            # Update progress bar
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Calculate metrics
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        all_preds_np = np.array(all_preds)
        all_targets_np = np.array(all_targets)

        # Binary accuracy
        pred_binary = (all_preds_np > 0.5).astype(int)
        accuracy = (pred_binary == all_targets_np).mean() if len(all_targets_np) > 0 else 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy
        }

    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses: list[float] = []
        all_preds: list[float] = []
        all_targets: list[float] = []

        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.epoch} [Val]')
            
            for batch in pbar:
                batch = batch.to(self.device)
                
                # Forward pass
                predictions = self.model(batch)  # [num_edges] logits

                # Get targets and valid mask
                targets = batch.y_suggested
                valid_mask = batch.valid_suggested

                # Compute loss
                loss = self.criterion(predictions, targets, valid_mask)

                # Log losses and predictions
                epoch_losses.append(loss.item())

                # Collect predictions for metrics
                pred_probs = torch.sigmoid(predictions[valid_mask])
                all_preds.extend(pred_probs.cpu().numpy().tolist())
                all_targets.extend(targets[valid_mask].cpu().numpy().tolist())

                # Update progress bar
                pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        # Calculate metrics
        avg_loss = np.mean(epoch_losses) if epoch_losses else 0.0
        all_preds_np = np.array(all_preds)
        all_targets_np = np.array(all_targets)

        # Binary accuracy
        pred_binary = (all_preds_np > 0.5).astype(int)
        accuracy = (pred_binary == all_targets_np).mean() if len(all_targets_np) > 0 else 0.0

        # Precision, recall, F1 for positive class (suggested lanes)
        from sklearn.metrics import precision_recall_fscore_support
        if len(all_targets_np) > 0:
            precision, recall, f1, _ = precision_recall_fscore_support(
                all_targets_np, pred_binary, average='binary', zero_division=0
            )
        else:
            precision, recall, f1 = 0.0, 0.0, 0.0

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    def save_checkpoint(self, is_best: bool = False) -> None:
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_val_loss': self.best_val_loss,
            'train_losses': self.train_losses,
            'val_losses': self.val_losses
        }
        
        # Save latest checkpoint
        torch.save(checkpoint, self.save_dir / 'latest.pt')
        
        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.save_dir / 'best.pt')
            # Also save as TorchScript
            self.save_torchscript_model()
            logger.info(f"Saved best model with validation loss: {self.best_val_loss:.4f}")
    
    def save_torchscript_model(self, filename: str = 'best_torchscript.pt') -> None:
        """Save model in TorchScript format for inference."""
        try:
            # Import the wrapper class
            from junction_ml.models import JunctionGNNTorchScript

            # Set model to evaluation mode
            self.model.eval()

            # Create TorchScript-compatible wrapper
            torchscript_wrapper = JunctionGNNTorchScript(self.model)
            torchscript_wrapper.eval()

            # Get a sample batch to extract tensor inputs
            with torch.no_grad():
                sample_batch = next(iter(self.val_loader))
                sample_batch = sample_batch.to(self.device)

                # Extract individual tensors
                node_features = sample_batch.x
                edge_index = sample_batch.edge_index
                edge_features = sample_batch.edge_attr

                # Trace the wrapper model with tensor inputs
                traced_model = torch.jit.trace(
                    torchscript_wrapper,
                    (node_features, edge_index, edge_features)
                )

                # Save the traced model
                torchscript_path = self.save_dir / filename
                traced_model.save(str(torchscript_path))

                logger.info(f"Saved TorchScript model to: {torchscript_path}")

        except Exception as e:
            logger.warning(f"Failed to save TorchScript model with tracing: {e}")
            logger.info("Falling back to scripting method...")

            try:
                # Try scripting the wrapper instead
                from junction_ml.models import JunctionGNNTorchScript
                torchscript_wrapper = JunctionGNNTorchScript(self.model)
                torchscript_wrapper.eval()

                scripted_model = torch.jit.script(torchscript_wrapper)
                torchscript_path = self.save_dir / filename
                scripted_model.save(str(torchscript_path))
                logger.info(f"Saved TorchScript model (scripted) to: {torchscript_path}")

            except Exception as script_e:
                logger.error(f"Failed to save TorchScript model with both tracing and scripting: {script_e}")
                # As a fallback, save just the state dict for manual loading
                try:
                    state_dict_path = self.save_dir / f"model_state_dict_{filename}"
                    torch.save(self.model.state_dict(), state_dict_path)
                    logger.info(f"Saved model state dict to: {state_dict_path}")
                except Exception as fallback_e:
                    logger.error(f"Failed to save even the state dict: {fallback_e}")

    def save_final_torchscript(self) -> None:
        """Save the final trained model in TorchScript format."""
        # Load the best model first
        best_checkpoint_path = self.save_dir / 'best.pt'
        if best_checkpoint_path.exists():
            checkpoint = torch.load(best_checkpoint_path, map_location=self.device, weights_only=False)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            logger.info("Loaded best model for TorchScript conversion")

        # Save as TorchScript
        self.save_torchscript_model('final_model.pt')

    def load_checkpoint(self, checkpoint_path: str) -> None:
        """Load model checkpoint."""
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.epoch = checkpoint['epoch']
        self.best_val_loss = checkpoint['best_val_loss']
        self.train_losses = checkpoint.get('train_losses', [])
        self.val_losses = checkpoint.get('val_losses', [])
        
        logger.info(f"Loaded checkpoint from epoch {self.epoch}")
    
    def train(self, num_epochs: int) -> None:
        """
        Train the model for specified number of epochs.
        
        Args:
            num_epochs: Number of epochs to train
        """
        logger.info(f"Starting training for {num_epochs} epochs...")
        
        for epoch in range(num_epochs):
            self.epoch = epoch
            
            # Train epoch
            train_losses = self.train_epoch()
            self.train_losses.append(train_losses)
            
            # Validate epoch
            val_losses = self.validate_epoch()
            self.val_losses.append(val_losses)
            
            # Log to tensorboard
            for key, value in train_losses.items():
                self.writer.add_scalar(f'Train/{key}', value, epoch)
            
            for key, value in val_losses.items():
                self.writer.add_scalar(f'Validation/{key}', value, epoch)
            
            # Check for improvement
            current_val_loss = val_losses['loss']
            is_best = current_val_loss < self.best_val_loss
            
            if is_best:
                self.best_val_loss = current_val_loss
                self.patience_counter = 0
            else:
                self.patience_counter += 1
            
            # Save checkpoint
            self.save_checkpoint(is_best)
            
            # Early stopping
            if self.patience_counter >= self.patience:
                logger.info(f"Early stopping triggered after {self.patience} epochs without improvement")
                break
            
            # Log epoch summary
            logger.info(
                f"Epoch {epoch}: "
                f"Train Loss: {train_losses['total_loss']:.4f}, "
                f"Val Loss: {current_val_loss:.4f}, "
                f"Best Val Loss: {self.best_val_loss:.4f}"
            )
        
        logger.info("Training completed!")
        self.writer.close()


def create_trainer(model: JunctionGNN,
                   train_loader: DataLoader,
                   val_loader: DataLoader,
                   learning_rate: float = 1e-3,
                   weight_decay: float = 1e-5,
                   pos_weight: Optional[float] = None,
                   device: Optional[torch.device] = None,
                   **trainer_kwargs: Any) -> JunctionTrainer:
    """
    Create a trainer instance with default configurations.
    
    Args:
        model: Model to train
        train_loader: Training data loader
        val_loader: Validation data loader
        learning_rate: Learning rate for optimizer
        weight_decay: Weight decay for optimizer
        pos_weight: Weight for positive class in BCE loss (for class imbalance)
        device: Device to train on
        **trainer_kwargs: Additional arguments for trainer
        
    Returns:
        JunctionTrainer instance
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    model = model.to(device)
    
    # Create optimizer
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    
    # Create loss function with pos_weight for handling class imbalance
    if pos_weight is None:
        pos_weight = 1.0
        logger.warning("pos_weight not provided, using default value of 1.0")

    criterion = BinarySuggestedLoss(pos_weight=pos_weight)

    # Create trainer
    trainer = JunctionTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        **trainer_kwargs
    )
    
    return trainer
