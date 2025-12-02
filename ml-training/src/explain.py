#!/usr/bin/env python3
"""
Explainability script for junction lane suggestions.

This script loads a trained model and explains the predictions by showing which 
features and edges are most important for the model's decisions.
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union, Tuple
import logging
from torch import Tensor

# Add safe globals for numpy objects in PyTorch checkpoints
torch.serialization.add_safe_globals([np.core.multiarray.scalar])  # type: ignore

from junction_ml.data import JunctionGraphDataset, EdgeFeatureCount, NodeFeatureCount
from junction_ml.models import JunctionGNN

# Try to import explainability libraries
try:
    from torch_geometric.explain import Explainer, GNNExplainer # type: ignore[import-untyped]
    HAS_PYG_EXPLAINER = True
except ImportError:
    HAS_PYG_EXPLAINER = False
    print("Warning: torch_geometric.explain not available. Feature importance will be limited.")

try:
    from captum.attr import IntegratedGradients, Saliency # type: ignore[import-untyped]
    HAS_CAPTUM = True
except ImportError:
    HAS_CAPTUM = False
    print("Warning: captum not available. Using gradient-based attribution.")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Feature names for better interpretability (9 features for lane-level architecture)
EDGE_FEATURE_NAMES = [
    'length', 'laneCount', 'angle', 'route', 'type', 'usable', 'virtual',
    'relativeLanePosition', 'laneTurn'
]

NODE_FEATURE_NAMES = [
    'lat', 'lon', 'incoming edge count', 'outgoing edge count'
    # Add more if you have additional node features  
]


def load_model(model_path: str, device: torch.device) -> Union[torch.nn.Module, torch.jit.ScriptModule]:
    """Load a trained model from checkpoint."""
    if model_path.endswith('.pt') and 'torchscript' in model_path:
        # Load TorchScript model
        torchscript_model: torch.jit.ScriptModule = torch.jit.load(model_path, map_location=device)
        torchscript_model.eval()
        return torchscript_model
    else:
        # Load regular PyTorch model
        checkpoint = torch.load(model_path, map_location=device, weights_only=False)

        # Extract model configuration if available
        if 'model_config' in checkpoint:
            model_config = checkpoint['model_config']
        else:
            # Try to infer configuration from state_dict
            state_dict = checkpoint['model_state_dict']

            # Detect number of layers by finding the highest conv layer index
            conv_layer_indices = [int(k.split('.')[1]) for k in state_dict.keys() if k.startswith('convs.')]
            num_layers = max(conv_layer_indices) + 1 if conv_layer_indices else 3

            # Detect hidden_dim from the first conv layer weight shape
            hidden_dim = 64  # default
            if 'convs.0.lin.weight' in state_dict:
                hidden_dim = state_dict['convs.0.lin.weight'].shape[0]

            logger.info(f"Inferred model config: num_layers={num_layers}, hidden_dim={hidden_dim}")

            model_config = {
                'node_features': NodeFeatureCount,
                'edge_features': EdgeFeatureCount,
                'hidden_dim': hidden_dim,
                'num_layers': num_layers,
                'dropout': 0.1
            }

        pytorch_model: torch.nn.Module = JunctionGNN(**model_config)
        pytorch_model.load_state_dict(checkpoint['model_state_dict'])
        pytorch_model.to(device)
        pytorch_model.eval()
        return pytorch_model


def json_to_data(json_data: Dict[str, Any]) -> Optional[Any]:
    """Convert JSON data to PyTorch Geometric Data object."""
    # Create a temporary dataset instance to use its conversion method
    dataset = JunctionGraphDataset()
    return dataset._convert_to_pyg_data(json_data)


def gradient_based_attribution(model: torch.nn.Module, 
                               data: Any, 
                               target_edge: int,
                               target_output: int = 0) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute feature attribution using gradients.
    
    Args:
        model: The trained model
        data: PyTorch Geometric data object
        target_edge: Edge index to explain
        target_output: Which output to explain (0=suggestedFrom, 1=suggestedTo, 2=suggestedTurn)
        
    Returns:
        Tuple of (node_attribution, edge_attribution)
    """
    # Enable gradients for input features
    data.x.requires_grad_(True)
    data.edge_attr.requires_grad_(True)
    
    # Forward pass
    predictions = model(data)
    
    # Select the target prediction (single binary output per edge)
    target_logit = predictions[target_edge]

    # Backward pass
    target_logit.backward()
    
    # Get gradients
    node_grads = data.x.grad.abs() if data.x.grad is not None else torch.zeros_like(data.x)
    edge_grads = data.edge_attr.grad.abs() if data.edge_attr.grad is not None else torch.zeros_like(data.edge_attr)
    
    return node_grads, edge_grads


def captum_attribution(model: torch.nn.Module,
                      data: Any,
                      target_edge: int,
                      target_output: int = 0,
                      method: str = 'integrated_gradients') -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute feature attribution using Captum library.
    
    Args:
        model: The trained model
        data: PyTorch Geometric data object
        target_edge: Edge index to explain
        target_output: Which output to explain
        method: Attribution method ('integrated_gradients' or 'saliency')
        
    Returns:
        Tuple of (node_attribution, edge_attribution)
    """
    if not HAS_CAPTUM:
        return gradient_based_attribution(model, data, target_edge, target_output)
    
    def forward_func(node_feat: Tensor, edge_feat: Tensor) -> Tensor:
        # Create a new data object with modified features
        new_data = data.clone()
        new_data.x = node_feat
        new_data.edge_attr = edge_feat
        
        predictions = model(new_data)
        # Single binary prediction per edge
        return predictions[target_edge].unsqueeze(0)

    # Initialize attribution method
    if method == 'integrated_gradients':
        attr_method = IntegratedGradients(forward_func)
    else:
        attr_method = Saliency(forward_func)
    
    # Compute attributions
    node_attr, edge_attr = attr_method.attribute(
        (data.x, data.edge_attr),
        target=0
    )
    
    return node_attr.abs(), edge_attr.abs()


def pyg_explainer_attribution(model: torch.nn.Module,
                             data: Any,
                             target_edge: int) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
    """
    Use PyTorch Geometric's explainer for attribution.
    
    Args:
        model: The trained model
        data: PyTorch Geometric data object
        target_edge: Edge index to explain
        
    Returns:
        Tuple of (node_mask, edge_mask) or (None, None) if not available
    """
    if not HAS_PYG_EXPLAINER:
        return None, None
    
    try:
        explainer = Explainer(
            model=model,
            algorithm=GNNExplainer(epochs=200),
            explanation_type='model',
            node_mask_type='attributes',
            edge_mask_type='object'
        )
        
        # Note: GNNExplainer typically explains node predictions
        # For edge predictions, we might need a different approach
        explanation = explainer(
            x=data.x,
            edge_index=data.edge_index,
            edge_attr=data.edge_attr
        )
        
        return explanation.node_mask, explanation.edge_mask
    except Exception as e:
        logger.warning(f"PyG Explainer failed: {e}")
        return None, None


def explain_junction(model: Union[torch.nn.Module, torch.jit.ScriptModule],
                    json_data: Dict[str, Any],
                    device: torch.device,
                    method: str = 'gradients',
                    top_k: int = 5) -> Dict[str, Any]:
    """
    Explain predictions for a junction graph.
    
    Args:
        model: Trained model
        json_data: Junction graph data in JSON format
        device: Device to run inference on
        method: Attribution method ('gradients', 'integrated_gradients', 'saliency', 'gnn_explainer')
        top_k: Number of top features to report
        
    Returns:
        Dictionary with explanations and predictions
    """
    # Convert JSON to PyTorch Geometric data
    data = json_to_data(json_data)
    if data is None:
        raise ValueError("Could not convert JSON data to PyTorch Geometric format")
    
    # Move data to device
    data = data.to(device)
    
    # First get predictions
    with torch.no_grad():
        if hasattr(model, 'forward'):
            predictions = model(data)
        else:
            predictions = model(data.x, data.edge_index, data.edge_attr)

    # Apply sigmoid to get probabilities for binary classification
    suggested_probs = torch.sigmoid(predictions).cpu().numpy()

    # Prepare results structure
    results: Dict[str, Any] = {
        'predictions': {
            'suggested': suggested_probs.tolist()
        },
        'metadata': {
            'num_nodes': len(json_data['nodes']),
            'num_edges': len(json_data['edges']),
            'num_lane_edges': sum(1 for e in json_data['edges'] if e.get('virtual', 0.0) < 0.5),
            'explanation_method': method,
            'top_k_features': top_k
        },
        'explanations': []
    }
    
    # For TorchScript models, convert back to regular model for explanations
    if hasattr(model, '_c'):
        logger.warning("Attribution may not work properly with TorchScript models")
    
    # Explain each lane edge prediction (skip virtual edges)
    for edge_idx in range(len(json_data['edges'])):
        edge = json_data['edges'][edge_idx]
        
        # Skip virtual edges (convert to float to avoid numpy array issues)
        if float(edge.get('virtual', 0.0)) > 0.5:
            continue

        edge_explanation = {
            'edge_index': edge_idx,
            'from': edge['from'],
            'to': edge['to'],
            'lane_position': edge.get('relativeLanePosition', -1.0),
            'lane_turn': edge.get('laneTurn', -1.0),
            'predicted_suggested': float(suggested_probs[edge_idx]),
            'predicted_binary': bool(suggested_probs[edge_idx] > 0.5),
            'ground_truth': edge.get('suggested', None),
            'feature_importance': {}
        }
        
        # Compute attributions for the binary prediction
        try:
            if method == 'gnn_explainer':
                node_attr, edge_attr = pyg_explainer_attribution(model, data, edge_idx)
            elif method in ['integrated_gradients', 'saliency'] and HAS_CAPTUM:
                node_attr, edge_attr = captum_attribution(model, data, edge_idx, 0, method)
            else:
                node_attr, edge_attr = gradient_based_attribution(model, data, edge_idx, 0)

            if node_attr is not None and edge_attr is not None:
                # Node feature importance (aggregate across all nodes)
                node_importance = node_attr.mean(dim=0)  # Average across nodes
                top_node_features = torch.topk(node_importance, min(top_k, len(node_importance)))

                # Edge feature importance (9 features in new architecture)
                edge_importance = edge_attr[edge_idx] if edge_idx < edge_attr.size(0) else edge_attr.mean(dim=0)
                top_edge_features = torch.topk(edge_importance, min(top_k, len(edge_importance)))

                # Store feature importance
                edge_explanation['feature_importance'] = {
                    'node_features': [
                        {
                            'feature_name': NODE_FEATURE_NAMES[idx] if idx < len(NODE_FEATURE_NAMES) else f'node_feat_{idx}',
                            'importance': float(importance),
                            'feature_index': int(idx)
                        }
                        for idx, importance in zip(top_node_features.indices, top_node_features.values)
                    ],
                    'edge_features': [
                        {
                            'feature_name': EDGE_FEATURE_NAMES[idx] if idx < len(EDGE_FEATURE_NAMES) else f'edge_feat_{idx}',
                            'importance': float(importance),
                            'feature_index': int(idx)
                        }
                        for idx, importance in zip(top_edge_features.indices, top_edge_features.values)
                    ]
                }

                # Store edge importance (which edges in the graph contribute to this prediction)
                if edge_attr.dim() > 1:
                    graph_edge_importance = edge_attr.mean(dim=1)  # Average across feature dimensions
                    top_edges = torch.topk(graph_edge_importance, min(top_k, len(graph_edge_importance)))

                    edge_explanation['important_graph_edges'] = [
                        {
                            'edge_index': int(idx),
                            'importance': float(importance),
                            'from_node': int(data.edge_index[0, idx]) if idx < data.edge_index.size(1) else -1,
                            'to_node': int(data.edge_index[1, idx]) if idx < data.edge_index.size(1) else -1
                        }
                        for idx, importance in zip(top_edges.indices, top_edges.values)
                    ]

        except Exception as e:
            logger.warning(f"Attribution failed for edge {edge_idx}: {e}")
            edge_explanation['feature_importance'] = {'error': str(e)}

        results['explanations'].append(edge_explanation)
    
    return results


def explain_multiple_files(model: Union[torch.nn.Module, torch.jit.ScriptModule],
                          input_files: List[str],
                          device: torch.device,
                          method: str = 'gradients',
                          top_k: int = 5,
                          output_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Explain predictions on multiple JSON files.
    
    Args:
        model: Trained model
        input_files: List of JSON file paths
        device: Device to run inference on
        method: Attribution method
        top_k: Number of top features to report
        output_file: Optional output file to save all results
        
    Returns:
        List of explanation results
    """
    all_results = []
    
    for file_path in input_files:
        logger.info(f"Explaining {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            
            result = explain_junction(model, json_data, device, method, top_k)
            result['source_file'] = file_path
            all_results.append(result)
            
            logger.info(f"Successfully explained {file_path}: "
                       f"{result['metadata']['num_edges']} edges analyzed")
                       
        except Exception as e:
            logger.error(f"Error explaining {file_path}: {e}")
            all_results.append({
                'source_file': file_path,
                'error': str(e)
            })
    
    if output_file:
        logger.info(f"Saving explanations to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    return all_results


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Explain junction lane suggestion predictions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Explain predictions on a single file
  python explain.py --model checkpoints/best.pt --input junction.json
  
  # Use integrated gradients method
  python explain.py --model checkpoints/best.pt --input junction.json --method integrated_gradients
  
  # Show top 10 features and save to file
  python explain.py --model checkpoints/best.pt --input junction.json --top-k 10 --output explanations.json
  
  # Explain all files in a directory
  python explain.py --model checkpoints/best.pt --input-dir ../tmp-junctions/ --method saliency
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pt file)')
    
    parser.add_argument('--input', type=str, nargs='*',
                       help='Input JSON file(s) to explain')
    
    parser.add_argument('--input-dir', type=str,
                       help='Directory containing JSON files to explain')
    
    parser.add_argument('--output', type=str,
                       help='Output file to save explanations (JSON format)')
    
    parser.add_argument('--method', type=str, default='gradients',
                       choices=['gradients', 'integrated_gradients', 'saliency', 'gnn_explainer'],
                       help='Attribution method to use')
    
    parser.add_argument('--top-k', type=int, default=5,
                       help='Number of top features to show for each prediction')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to run inference on')
    
    parser.add_argument('--verbose', '-v', action='store_true',
                       help='Enable verbose output')
    
    return parser.parse_args()


def main() -> None:
    """Main function."""
    args = parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Determine device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Attribution method: {args.method}")
    
    # Load model
    logger.info(f"Loading model from {args.model}")
    model = load_model(args.model, device)
    logger.info("Model loaded successfully")
    
    # Collect input files
    input_files = []
    
    if args.input:
        input_files.extend(args.input)
    
    if args.input_dir:
        input_dir = Path(args.input_dir)
        if not input_dir.exists():
            raise ValueError(f"Input directory {input_dir} does not exist")
        
        json_files = list(input_dir.glob("*.json"))
        input_files.extend([str(f) for f in json_files])
    
    if not input_files:
        raise ValueError("No input files specified. Use --input or --input-dir")
    
    logger.info(f"Found {len(input_files)} input files")
    
    # Process files
    if len(input_files) == 1:
        # Single file explanation with detailed output
        file_path = input_files[0]
        logger.info(f"Explaining single file: {file_path}")
        
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        
        result = explain_junction(model, json_data, device, args.method, args.top_k)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Explanations saved to {args.output}")
        else:
            # Print results to stdout
            print(json.dumps(result, indent=2))
    
    else:
        # Multiple files explanation
        logger.info(f"Explaining {len(input_files)} files")
        results = explain_multiple_files(model, input_files, device, args.method, args.top_k, args.output)
        
        # Print summary
        successful = sum(1 for r in results if 'error' not in r)
        failed = len(results) - successful
        
        logger.info(f"Explanation complete: {successful} successful, {failed} failed")
        
        if not args.output:
            # Print summary results
            for result in results:
                if 'error' in result:
                    print(f"ERROR {result['source_file']}: {result['error']}")
                else:
                    print(f"SUCCESS {result['source_file']}: "
                          f"{result['metadata']['num_edges']} edges explained")


if __name__ == "__main__":
    main()
