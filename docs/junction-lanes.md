# Implementation Step 01: Junction Lane Prediction Model

This document summarizes the current implementation of the OSMScout ML project for predicting junction lane suggestions using Graph Neural Networks (GNNs). This serves as context for future development sessions and onboarding new developers.

## Project Overview

The OSMScout ML project is a playground for machine learning experiments with OpenStreetMap data, specifically focused on improving navigation in the harbour-osmscout application through junction lane prediction using GNNs.

### Core Problem
Predict lane suggestions (`suggestedFrom`, `suggestedTo`, `suggestedTurn`) at road junctions based on junction topology, road characteristics, and routing context.

## Current Architecture

### C++ Data Export Pipeline

#### 1. Graph Data Structure (`src/JunctionGraphProcessor.h`)

**GraphNode:**
- `Id id`: Unique node identifier
- `GeoCoord location`: Geographic coordinates (lat/lon)

**GraphEdge:**
- `Id fromNode`, `Id toNode`: Edge connectivity
- `Distance length`: Physical edge length
- `std::unordered_map<std::string, double> features`: Feature dictionary

**Graph:**
- `std::vector<GraphNode> nodes`: All junction nodes
- `std::vector<GraphEdge> edges`: All junction edges
- `void Export(const std::filesystem::path &filePath)`: JSON export functionality

#### 2. Edge Features (18 total features)

Current edge features defined in `GraphFeature` namespace:

**Core Road Features:**
- `LANE_COUNT`: Number of lanes on the road
- `ANGLE`: Turn angle at junction
- `ONEWAY`: Boolean indicating one-way road
- `TYPE`: Road type (highway classification)
- `USABLE`: Boolean indicating if edge is usable by current vehicle
- `VIRTUAL`: Boolean indicating virtual edges for GNN information propagation

**Routing Context:**
- `ROUTE`: Boolean indicating if edge is part of current route
- `SUGGESTED_FROM`, `SUGGESTED_TO`, `SUGGESTED_TURN`: Target prediction labels

**Lane Turn Features (10 additional):**
- Detailed lane-level turn information for complex junctions

#### 3. Junction Graph Processors

**JunctionGraphProcessor:**
- Base class implementing `RoutePostprocessor::Postprocessor`
- Processes route descriptions and extracts junction graphs

**JunctionGraphExportProcessor:**
- Exports junction graphs to JSON files for ML training
- Simple junction export functionality

**ComplexJunctionGraphExportProcessor:**
- Extended export for complex junctions with enhanced features

### Python ML Pipeline

#### 1. Dependencies (`ml-training/pyproject.toml`)

**Core ML Libraries:**
- `torch ^2.8.0`: PyTorch for deep learning
- `torch-geometric ^2.6.1`: Graph neural network library
- `scikit-learn ^1.3.0`: Data preprocessing and metrics
- `numpy ^1.24.0`, `pandas ^2.0.0`: Data manipulation
- `captum ^0.8.0`: Model interpretability and feature attribution

**Development Tools:**
- `pytest ^7.4.0`: Testing framework
- `mypy ^1.5.0`: Type checking
- `black ^23.0.0`: Code formatting
- `tensorboard ^2.13.0`: Training visualization

#### 2. Data Loading (`junction_ml/data/__init__.py`)

**Feature Configuration:**
- `EdgeFeatureCount = 18`: Total edge features including lane turns
- `NodeFeatureCount = 2`: Node features (lat, lon coordinates)

**JunctionGraphDataset:**
- Custom PyTorch Geometric dataset
- Loads JSON files exported from C++ pipeline
- Handles data preprocessing and feature normalization
- Converts junction graphs to PyTorch Geometric `Data` objects

#### 3. Graph Neural Network Model (`junction_ml/models/__init__.py`)

**JunctionGNN Architecture:**
- Multi-layer GNN with configurable architecture
- Supports GCN, GAT, and GraphConv layer types
- Node embedding: Linear transformation of geographic coordinates
- Edge embedding: Linear transformation of road features
- Multiple graph convolution layers with layer normalization
- Edge prediction heads for multi-task learning

**Model Configuration:**
- `node_features=2`: Geographic coordinates (lat, lon)
- `edge_features=18`: All road and routing features
- `hidden_dim=64`: Hidden layer dimension (configurable)
- `num_layers=3`: Number of GNN layers (configurable)
- `conv_type='gcn'`: Graph convolution type (gcn/gat/graph)
- `dropout=0.1`: Dropout for regularization
- `use_edge_attr=True`: Whether to use edge attributes

**Prediction Outputs:**
- `suggestedFrom`: Lane suggestions for incoming traffic
- `suggestedTo`: Lane suggestions for outgoing traffic  
- `suggestedTurn`: Turn direction suggestions

#### 4. Training Pipeline (`src/train.py`)

**Training Features:**
- Multi-task learning for three prediction targets
- Support for different loss functions (MSE, CrossEntropy)
- Early stopping based on validation loss
- Model checkpointing (best and latest models)
- TensorBoard logging for training visualization
- GPU acceleration when available

#### 5. Inference Tools

**Predict Tool (`src/predict.py`):**
- Loads trained models for inference
- Processes single junction graphs
- Outputs predictions for lane suggestions

**Explain Tool (`src/explain.py`):**
- Uses Captum library for model interpretability
- Integrated Gradients attribution method
- Shows which features and edges are most important for predictions
- Helps understand model decision-making process

## Libraries and Technologies Used

### C++ Dependencies
- **libosmscout**: Core OSM data processing and routing
- **nlohmann/json**: JSON serialization for data export
- **CMake**: Build system configuration

### Python Dependencies  
- **PyTorch**: Deep learning framework
- **PyTorch Geometric**: Specialized GNN library
- **Captum**: Model interpretability and explainability
- **scikit-learn**: Data preprocessing and evaluation metrics
- **NumPy/Pandas**: Data manipulation and analysis
- **TensorBoard**: Training visualization and monitoring

## Key Implementation Techniques

### 1. Graph Construction
- Junction-centered graph extraction from routing data
- Virtual edges for improved information propagation in GNNs
- Feature engineering from OSM road attributes and routing context

### 2. Coordinate Normalization
Junction graphs may have different orientations (north-facing vs east-facing). Current implementation uses raw lat/lon coordinates, but normalization strategies being considered:
- Center-relative positioning
- Rotation normalization based on main road direction
- Polar coordinate transformation

### 3. Multi-task Learning
The model simultaneously predicts three related targets:
- `suggestedFrom`: Input lane suggestions
- `suggestedTo`: Output lane suggestions  
- `suggestedTurn`: Turn direction recommendations

### 4. Graph Convolution Strategy
GNNs propagate information bidirectionally regardless of edge orientation, making them suitable for junction topology where traffic flow depends on routing context rather than pure graph structure.

## Current Status and Limitations

### Implemented Features
✅ C++ junction graph extraction and JSON export  
✅ Python data loading and preprocessing pipeline  
✅ Configurable GNN architecture with multiple layer types  
✅ Multi-task training pipeline with validation  
✅ Model inference and prediction tools  
✅ Model interpretability with feature attribution  
✅ Comprehensive testing and type checking setup

### Known Limitations
- Raw geographic coordinates without orientation normalization
- Limited to junction-level predictions (no route-level optimization)
- Feature engineering may need refinement based on model performance
- No production deployment pipeline yet

### Next Steps for Development
1. Implement coordinate normalization for orientation-invariant features
2. Expand feature engineering based on model interpretation results
3. Add route-level prediction and optimization
4. Implement production deployment pipeline
5. Add more comprehensive evaluation metrics and benchmarks

## Development Environment Setup

### C++ Build
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=<libosmscout-install-dir> ..
make -j $(nproc)
```

### Python Environment
```bash
cd ml-training
poetry install
poetry shell  # or poetry run <command>
```

### Running Tools
```bash
# Train model
poetry run python src/train.py

# Make predictions  
poetry run python src/predict.py --model output/model/best.pt --input <junction.json>

# Explain predictions
poetry run python src/explain.py --model output/model/best.pt --input <junction.json>

# Run tests
poetry run pytest
```

This implementation provides a solid foundation for junction lane prediction using GNNs and can be extended for more sophisticated navigation improvements.
