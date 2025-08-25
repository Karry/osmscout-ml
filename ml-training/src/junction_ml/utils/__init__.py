"""
Utility functions for junction ML project.
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Any, Optional
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


def setup_logging(log_level: str = 'INFO', log_file: Optional[str] = None):
    """
    Set up logging configuration.
    
    Args:
        log_level: Logging level ('DEBUG', 'INFO', 'WARNING', 'ERROR')
        log_file: Optional log file path
    """
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            *([logging.FileHandler(log_file)] if log_file else [])
        ]
    )


def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility.
    
    Args:
        seed: Random seed value
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def count_parameters(model: torch.nn.Module) -> int:
    """
    Count the number of trainable parameters in a model.
    
    Args:
        model: PyTorch model
        
    Returns:
        Number of trainable parameters
    """
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_config(config: Dict[str, Any], save_path: str):
    """
    Save configuration to JSON file.
    
    Args:
        config: Configuration dictionary
        save_path: Path to save the config
    """
    with open(save_path, 'w') as f:
        json.dump(config, f, indent=2)


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from JSON file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path, 'r') as f:
        return json.load(f)


def plot_training_curves(train_losses: List[Dict[str, float]], 
                        val_losses: List[Dict[str, float]],
                        save_path: Optional[str] = None):
    """
    Plot training and validation loss curves.
    
    Args:
        train_losses: List of training losses per epoch
        val_losses: List of validation losses per epoch
        save_path: Optional path to save the plot
    """
    epochs = range(1, len(train_losses) + 1)
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle('Training and Validation Curves')
    
    # Total loss
    axes[0, 0].plot(epochs, [l['total_loss'] for l in train_losses], 'b-', label='Train')
    axes[0, 0].plot(epochs, [l['total_loss'] for l in val_losses], 'r-', label='Validation')
    axes[0, 0].set_title('Total Loss')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].legend()
    axes[0, 0].grid(True)
    
    # Suggested From loss
    if 'suggested_from_loss' in train_losses[0]:
        axes[0, 1].plot(epochs, [l['suggested_from_loss'] for l in train_losses], 'b-', label='Train')
        axes[0, 1].plot(epochs, [l['suggested_from_loss'] for l in val_losses], 'r-', label='Validation')
        axes[0, 1].set_title('Suggested From Loss')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Loss')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
    
    # Suggested To loss
    if 'suggested_to_loss' in train_losses[0]:
        axes[1, 0].plot(epochs, [l['suggested_to_loss'] for l in train_losses], 'b-', label='Train')
        axes[1, 0].plot(epochs, [l['suggested_to_loss'] for l in val_losses], 'r-', label='Validation')
        axes[1, 0].set_title('Suggested To Loss')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Loss')
        axes[1, 0].legend()
        axes[1, 0].grid(True)
    
    # Suggested Turn loss
    if 'suggested_turn_loss' in train_losses[0]:
        axes[1, 1].plot(epochs, [l['suggested_turn_loss'] for l in train_losses], 'b-', label='Train')
        axes[1, 1].plot(epochs, [l['suggested_turn_loss'] for l in val_losses], 'r-', label='Validation')
        axes[1, 1].set_title('Suggested Turn Loss')
        axes[1, 1].set_xlabel('Epoch')
        axes[1, 1].set_ylabel('Loss')
        axes[1, 1].legend()
        axes[1, 1].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved training curves to {save_path}")
    
    plt.show()


def evaluate_model(model: torch.nn.Module, 
                  data_loader: torch.utils.data.DataLoader,
                  device: torch.device) -> Dict[str, float]:
    """
    Evaluate model on a dataset.
    
    Args:
        model: PyTorch model
        data_loader: Data loader for evaluation
        device: Device to run evaluation on
        
    Returns:
        Dictionary with evaluation metrics
    """
    model.eval()
    
    all_predictions = {'suggested_from': [], 'suggested_to': [], 'suggested_turn': []}
    all_targets = {'suggested_from': [], 'suggested_to': [], 'suggested_turn': []}
    all_valid = {'suggested_from': [], 'suggested_to': [], 'suggested_turn': []}
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            
            # Forward pass
            predictions = model(batch)
            
            # Collect predictions and targets
            for task in ['suggested_from', 'suggested_to', 'suggested_turn']:
                if task in predictions:
                    all_predictions[task].append(predictions[task].cpu())
                    
                target_key = f'y_{task}'
                valid_key = f'valid_{task.split("_")[1]}'
                
                if hasattr(batch, target_key):
                    all_targets[task].append(getattr(batch, target_key).cpu())
                if hasattr(batch, valid_key):
                    all_valid[task].append(getattr(batch, valid_key).cpu())
    
    # Concatenate all predictions and targets
    metrics = {}
    
    for task in ['suggested_from', 'suggested_to', 'suggested_turn']:
        if all_predictions[task] and all_targets[task]:
            preds = torch.cat(all_predictions[task])
            targets = torch.cat(all_targets[task])
            valid = torch.cat(all_valid[task]) if all_valid[task] else torch.ones_like(targets, dtype=torch.bool)
            
            if valid.sum() > 0:
                valid_preds = preds[valid]
                valid_targets = targets[valid]
                
                # Calculate metrics
                mse = torch.mean((valid_preds - valid_targets) ** 2).item()
                mae = torch.mean(torch.abs(valid_preds - valid_targets)).item()
                
                metrics[f'{task}_mse'] = mse
                metrics[f'{task}_mae'] = mae
                metrics[f'{task}_rmse'] = np.sqrt(mse)
    
    return metrics


def visualize_predictions(model: torch.nn.Module,
                         data_loader: torch.utils.data.DataLoader,
                         device: torch.device,
                         num_samples: int = 100,
                         save_path: Optional[str] = None):
    """
    Visualize model predictions vs ground truth.
    
    Args:
        model: PyTorch model
        data_loader: Data loader
        device: Device to run on
        num_samples: Number of samples to visualize
        save_path: Optional path to save the plot
    """
    model.eval()
    
    predictions = {'suggested_from': [], 'suggested_to': [], 'suggested_turn': []}
    targets = {'suggested_from': [], 'suggested_to': [], 'suggested_turn': []}
    
    with torch.no_grad():
        for batch in data_loader:
            batch = batch.to(device)
            preds = model(batch)
            
            for task in ['suggested_from', 'suggested_to', 'suggested_turn']:
                if task in preds:
                    target_key = f'y_{task}'
                    valid_key = f'valid_{task.split("_")[1]}'
                    
                    if hasattr(batch, target_key) and hasattr(batch, valid_key):
                        valid = getattr(batch, valid_key)
                        if valid.sum() > 0:
                            predictions[task].extend(preds[task][valid].cpu().numpy())
                            targets[task].extend(getattr(batch, target_key)[valid].cpu().numpy())
            
            # Stop when we have enough samples
            if len(predictions['suggested_from']) >= num_samples:
                break
    
    # Create scatter plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for i, task in enumerate(['suggested_from', 'suggested_to', 'suggested_turn']):
        if predictions[task] and targets[task]:
            preds = np.array(predictions[task][:num_samples])
            targs = np.array(targets[task][:num_samples])
            
            axes[i].scatter(targs, preds, alpha=0.6)
            
            # Perfect prediction line
            min_val = min(targs.min(), preds.min())
            max_val = max(targs.max(), preds.max())
            axes[i].plot([min_val, max_val], [min_val, max_val], 'r--', label='Perfect Prediction')
            
            axes[i].set_xlabel(f'True {task.replace("_", " ").title()}')
            axes[i].set_ylabel(f'Predicted {task.replace("_", " ").title()}')
            axes[i].set_title(f'{task.replace("_", " ").title()} Predictions')
            axes[i].legend()
            axes[i].grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved prediction visualization to {save_path}")
    
    plt.show()


class EarlyStopping:
    """Early stopping utility class."""
    
    def __init__(self, patience: int = 10, min_delta: float = 0.0):
        """
        Initialize early stopping.
        
        Args:
            patience: Number of epochs to wait for improvement
            min_delta: Minimum change to qualify as an improvement
        """
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float('inf')
        
    def __call__(self, val_loss: float) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            
        return self.counter >= self.patience
