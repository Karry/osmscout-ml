#!/usr/bin/env python3
"""
Prediction script for junction lane suggestions.

This script loads a trained model and predicts suggestedFrom, suggestedTo, and suggestedTurn
features for junction graphs provided in JSON format.
"""
import argparse
import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Optional, Union
import logging

# Add safe globals for numpy objects in PyTorch checkpoints
torch.serialization.add_safe_globals([np.core.multiarray.scalar]) # type: ignore

from junction_ml.data import EdgeFeatureCount, NodeFeatureCount, JunctionGraphDataset
from junction_ml.models import JunctionGNN

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


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


def predict_junction(model: Union[torch.nn.Module, torch.jit.ScriptModule],
                    json_data: Dict[str, Any],
                    device: torch.device) -> Dict[str, Any]:
    """
    Predict lane suggestions for a single junction graph.
    
    Args:
        model: Trained model
        json_data: Junction graph data in JSON format
        device: Device to run inference on
        
    Returns:
        Dictionary with predictions and metadata
    """
    # Convert JSON to PyTorch Geometric data
    data = json_to_data(json_data)
    if data is None:
        raise ValueError("Could not convert JSON data to PyTorch Geometric format")
    
    # Move data to device
    data = data.to(device)
    
    # Make prediction
    with torch.no_grad():
        if hasattr(model, 'forward'):
            # Regular PyTorch model
            predictions = model(data)
        else:
            # TorchScript model - pass individual tensors
            predictions = model(data.x, data.edge_index, data.edge_attr)

    # Apply sigmoid to get probabilities for binary classification
    suggested_probs = torch.sigmoid(predictions).cpu().numpy()

    # Group edges by highway (from-to node pair) for better visualization
    edges_by_highway: Dict[str, List[tuple]] = {}
    for i, edge in enumerate(json_data['edges']):
        if edge.get('virtual', 0.0) > 0.5:
            continue  # Skip virtual edges
        highway_key = f"{edge['from']}->{edge['to']}"
        if highway_key not in edges_by_highway:
            edges_by_highway[highway_key] = []
        edges_by_highway[highway_key].append((i, edge))

    # Prepare results
    results: Dict[str, Any] = {
        'predictions': {
            'suggested': suggested_probs.tolist()
        },
        'metadata': {
            'num_nodes': len(json_data['nodes']),
            'num_edges': len(json_data['edges']),
            'num_lane_edges': sum(1 for e in json_data['edges'] if e.get('virtual', 0.0) < 0.5),
            'model_type': 'torchscript' if hasattr(model, '_c') else 'pytorch'
        },
        'highways': []
    }
    
    # Group predictions by highway for easier interpretation
    for highway_key, lane_edges in edges_by_highway.items():
        # Extract from and to nodes from the first edge in this highway
        first_edge = lane_edges[0][1]
        highway_result = {
            'from_node': first_edge['from'],
            'to_node': first_edge['to'],
            'lanes': []
        }
        
        for i, edge in lane_edges:
            lane_result = {
                'lane_position': edge.get('relativeLanePosition', -1.0),
                'lane_turn': edge.get('laneTurn', -1.0),
                'predicted_suggested': float(suggested_probs[i]),
                'predicted_binary': bool(suggested_probs[i] > 0.5),
            }

            # Add ground truth if available
            if 'suggested' in edge:
                lane_result['ground_truth_suggested'] = float(edge['suggested'])

            # Add highway-level features (same for all lanes)
            if not highway_result.get('features'):
                highway_result['features'] = {
                    'length': edge.get('length', 0),
                    'total_lanes': edge.get('laneCount', 0),
                    'angle': edge.get('angle', 0),
                    'route': edge.get('route', 0),
                    'type': edge.get('type', -1),
                    'usable': edge.get('usable', 0)
                }

            highway_result['lanes'].append(lane_result)

        # Sort lanes by position
        highway_result['lanes'].sort(key=lambda x: x['lane_position'])
        results['highways'].append(highway_result)

    return results


def predict_multiple_files(model: Union[torch.nn.Module, torch.jit.ScriptModule],
                          input_files: List[str],
                          device: torch.device,
                          output_file: Optional[str] = None) -> List[Dict[str, Any]]:
    """
    Predict on multiple JSON files.
    
    Args:
        model: Trained model
        input_files: List of JSON file paths
        device: Device to run inference on
        output_file: Optional output file to save all results
        
    Returns:
        List of prediction results
    """
    all_results = []
    
    for file_path in input_files:
        logger.info(f"Processing {file_path}")
        
        try:
            with open(file_path, 'r') as f:
                json_data = json.load(f)
            
            result = predict_junction(model, json_data, device)
            result['source_file'] = file_path
            all_results.append(result)
            
            logger.info(f"Successfully processed {file_path}: "
                       f"{result['metadata']['num_edges']} edges predicted")
                       
        except Exception as e:
            logger.error(f"Error processing {file_path}: {e}")
            all_results.append({
                'source_file': file_path,
                'error': str(e)
            })
    
    if output_file:
        logger.info(f"Saving results to {output_file}")
        with open(output_file, 'w') as f:
            json.dump(all_results, f, indent=2)
    
    return all_results


def print_predictions(result: Dict[str, Any]) -> None:
    """Print predictions in a human-readable format."""
    print("\n" + "="*80)
    print("JUNCTION LANE PREDICTIONS")
    print("="*80)

    metadata = result['metadata']
    print(f"\nMetadata:")
    print(f"  Nodes: {metadata['num_nodes']}")
    print(f"  Total edges: {metadata['num_edges']}")
    print(f"  Lane edges: {metadata['num_lane_edges']}")
    print(f"  Model type: {metadata['model_type']}")

    # Lane turn mapping
    lane_turn_names = {
        -1.0: "Unknown",
        0.0: "None",
        1.0: "Slight-Left",
        2.0: "Left",
        3.0: "Sharp-Left",
        4.0: "U-Turn",
        15.0: "Slight-Right",
        16.0: "Right",
        17.0: "Through",
        18.0: "Sharp-Right"
    }

    print(f"\n{'='*80}")
    print("HIGHWAYS AND LANES")
    print("="*80)

    for highway in result['highways']:
        print(f"\n--- Highway: Node {highway['from_node']} → Node {highway['to_node']} ---")

        features = highway.get('features', {})
        print(f"  Length: {features.get('length', 0):.1f}m")
        print(f"  Total lanes: {int(features.get('total_lanes', 0))}")
        print(f"  Angle: {features.get('angle', 0):.1f}°")
        print(f"  On route: {'Yes' if features.get('route', 0) > 0.5 else 'No'}")

        print(f"\n  Lanes:")
        for lane in highway['lanes']:
            pos = lane['lane_position']
            turn = lane['lane_turn']
            pred = lane['predicted_suggested']
            is_suggested = lane['predicted_binary']

            # Position label
            if pos < 0.33:
                pos_label = "LEFT "
            elif pos > 0.67:
                pos_label = "RIGHT"
            else:
                pos_label = "MID  "

            # Turn name
            turn_name = lane_turn_names.get(turn, f"Unknown({turn})")

            # Suggestion indicator
            indicator = "✓ SUGGESTED" if is_suggested else "  not suggested"

            print(f"    Lane {pos:.2f} ({pos_label}): {turn_name:15s} → {pred:.3f} {indicator}")

            # Show ground truth if available
            if 'ground_truth_suggested' in lane:
                gt = lane['ground_truth_suggested']
                gt_label = "✓ CORRECT" if (gt > 0.5) == is_suggested else "✗ WRONG"
                print(f"      Ground truth: {gt:.1f} ({gt_label})")

    print("\n" + "="*80)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Predict junction lane suggestions from JSON files",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Predict on a single file
  python predict.py --model checkpoints/best.pt --input junction.json
  
  # Predict on multiple files
  python predict.py --model checkpoints/best_torchscript.pt --input file1.json file2.json
  
  # Predict on all JSON files in a directory
  python predict.py --model checkpoints/best.pt --input-dir ../tmp-junctions/
  
  # Save results to file
  python predict.py --model checkpoints/best.pt --input junction.json --output results.json
  
  # Use GPU if available
  python predict.py --model checkpoints/best.pt --input junction.json --device cuda
        """
    )
    
    parser.add_argument('--model', type=str, required=True,
                       help='Path to trained model (.pt file)')
    
    parser.add_argument('--input', type=str, nargs='*',
                       help='Input JSON file(s) to predict on')
    
    parser.add_argument('--input-dir', type=str,
                       help='Directory containing JSON files to predict on')
    
    parser.add_argument('--output', type=str,
                       help='Output file to save predictions (JSON format)')
    
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to run inference on')
    
    parser.add_argument('--batch-process', action='store_true',
                       help='Process all files and save summary results')
    
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
    if len(input_files) == 1 and not args.batch_process:
        # Single file prediction with detailed output
        file_path = input_files[0]
        logger.info(f"Processing single file: {file_path}")
        
        with open(file_path, 'r') as f:
            json_data = json.load(f)
        
        result = predict_junction(model, json_data, device)
        
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(result, f, indent=2)
            logger.info(f"Results saved to {args.output}")
        else:
            # Print human-readable results
            print_predictions(result)

    else:
        # Multiple files or batch processing
        logger.info(f"Processing {len(input_files)} files")
        results = predict_multiple_files(model, input_files, device, args.output)
        
        # Print summary
        successful = sum(1 for r in results if 'error' not in r)
        failed = len(results) - successful
        
        logger.info(f"Processing complete: {successful} successful, {failed} failed")
        
        if not args.output:
            # Print summary results
            for result in results:
                if 'error' in result:
                    print(f"ERROR {result['source_file']}: {result['error']}")
                else:
                    print(f"SUCCESS {result['source_file']}: "
                          f"{result['metadata']['num_edges']} edges predicted")


if __name__ == "__main__":
    main()
