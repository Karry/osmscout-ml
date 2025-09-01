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

logger = logging.getLogger(__name__)


class MultiTaskLoss(nn.Module):
    """
    Multi-task loss function for junction lane prediction.
    
    Combines losses for suggestedFrom, suggestedTo, and suggestedTurn
    with optional task weighting and uncertainty weighting.
    """
    
    def __init__(self, 
                 task_weights: Optional[Dict[str, float]] = None,
                 use_uncertainty_weighting: bool = False):
        """
        Initialize multi-task loss.
        
        Args:
            task_weights: Manual weights for each task
            use_uncertainty_weighting: Whether to use learnable uncertainty weighting
        """
        super().__init__()
        
        self.task_weights = task_weights or {
            'suggested_from': 1.0,
            'suggested_to': 1.0,
            'suggested_turn': 1.0
        }
        
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        if use_uncertainty_weighting:
            # Learnable uncertainty parameters (log variance)
            self.log_vars = nn.Parameter(torch.zeros(3))
        
        self.mse_loss = nn.MSELoss(reduction='none')
        
    def forward(self, 
                predictions: Dict[str, torch.Tensor], 
                targets: Dict[str, torch.Tensor],
                valid_masks: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss.
        
        Args:
            predictions: Model predictions for each task
            targets: Ground truth targets for each task
            valid_masks: Masks indicating valid labels for each task
            
        Returns:
            Dictionary containing individual and total losses
        """
        losses: dict[str, Any] = {}
        total_loss = 0.0
        
        tasks = ['suggested_from', 'suggested_to', 'suggested_turn']
        
        for i, task in enumerate(tasks):
            if task not in predictions or task not in targets:
                continue
                
            pred = predictions[task]
            target = targets[task]
            valid = valid_masks.get(f'valid_{task.split("_")[1]}', torch.ones_like(target, dtype=torch.bool))
            
            if valid.sum() == 0:
                losses[f'{task}_loss'] = torch.tensor(0.0, device=pred.device)
                continue
            
            # Compute MSE loss only for valid labels
            task_loss = self.mse_loss(pred[valid], target[valid]).mean()
            
            if self.use_uncertainty_weighting:
                # Uncertainty weighting: loss = exp(-log_var) * loss + log_var
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = precision * task_loss + self.log_vars[i]
            else:
                weighted_loss = self.task_weights[task] * task_loss
            
            losses[f'{task}_loss'] = task_loss
            total_loss += weighted_loss
        
        losses['total_loss'] = total_loss
        return losses


class JunctionTrainer:
    """
    Trainer class for junction lane prediction models.
    """
    
    def __init__(self,
                 model: nn.Module,
                 train_loader: DataLoader,
                 val_loader: DataLoader,
                 optimizer: optim.Optimizer,
                 criterion: MultiTaskLoss,
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
        self.model = model
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
        epoch_losses: dict[str, list[Any]] = {'total_loss': [], 'suggested_from_loss': [],
                                              'suggested_to_loss': [], 'suggested_turn_loss': []}
        
        pbar = tqdm(self.train_loader, desc=f'Epoch {self.epoch} [Train]')
        
        for batch in pbar:
            batch = batch.to(self.device)
            
            self.optimizer.zero_grad()
            
            # Forward pass
            predictions = self.model(batch)
            
            # Prepare targets and masks
            targets = {
                'suggested_from': batch.y_suggested_from,
                'suggested_to': batch.y_suggested_to,
                'suggested_turn': batch.y_suggested_turn
            }
            
            valid_masks = {
                'valid_from': batch.valid_from,
                'valid_to': batch.valid_to,
                'valid_turn': batch.valid_turn
            }
            
            # Compute loss
            losses = self.criterion(predictions, targets, valid_masks)
            
            # Backward pass
            losses['total_loss'].backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            # Log losses
            for key, value in losses.items():
                if key in epoch_losses:
                    epoch_losses[key].append(value.item())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f"{losses['total_loss'].item():.4f}",
                'from': f"{losses.get('suggested_from_loss', torch.tensor(0)).item():.4f}",
                'to': f"{losses.get('suggested_to_loss', torch.tensor(0)).item():.4f}",
                'turn': f"{losses.get('suggested_turn_loss', torch.tensor(0)).item():.4f}"
            })
        
        # Average losses
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items() if values}
        return avg_losses
    
    def validate_epoch(self) -> Dict[str, float]:
        """Validate for one epoch."""
        self.model.eval()
        epoch_losses: dict[str, list[Any]] = {'total_loss': [], 'suggested_from_loss': [],
                                              'suggested_to_loss': [], 'suggested_turn_loss': []}
        
        with torch.no_grad():
            pbar = tqdm(self.val_loader, desc=f'Epoch {self.epoch} [Val]')
            
            for batch in pbar:
                batch = batch.to(self.device)
                
                # Forward pass
                predictions = self.model(batch)
                
                # Prepare targets and masks
                targets = {
                    'suggested_from': batch.y_suggested_from,
                    'suggested_to': batch.y_suggested_to,
                    'suggested_turn': batch.y_suggested_turn
                }
                
                valid_masks = {
                    'valid_from': batch.valid_from,
                    'valid_to': batch.valid_to,
                    'valid_turn': batch.valid_turn
                }
                
                # Compute loss
                losses = self.criterion(predictions, targets, valid_masks)
                
                # Log losses
                for key, value in losses.items():
                    if key in epoch_losses:
                        epoch_losses[key].append(value.item())
                
                # Update progress bar
                pbar.set_postfix({
                    'loss': f"{losses['total_loss'].item():.4f}",
                    'from': f"{losses.get('suggested_from_loss', torch.tensor(0)).item():.4f}",
                    'to': f"{losses.get('suggested_to_loss', torch.tensor(0)).item():.4f}",
                    'turn': f"{losses.get('suggested_turn_loss', torch.tensor(0)).item():.4f}"
                })
        
        # Average losses
        avg_losses = {key: np.mean(values) for key, values in epoch_losses.items() if values}
        return avg_losses
    
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
            current_val_loss = val_losses['total_loss']
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


def create_trainer(model: nn.Module,
                  train_loader: DataLoader,
                  val_loader: DataLoader,
                  learning_rate: float = 1e-3,
                  weight_decay: float = 1e-5,
                  task_weights: Optional[Dict[str, float]] = None,
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
        task_weights: Weights for multi-task loss
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
    
    # Create loss function
    criterion = MultiTaskLoss(task_weights=task_weights)
    
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
