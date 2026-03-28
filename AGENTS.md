# AGENTS.md

## Project Overview

OSMScout ML predicts junction lane suggestions for navigation using Graph Neural Networks on OpenStreetMap data. It has two tightly coupled components:

1. **C++ data pipeline** (`src/`) — extracts junction graphs from OSM routing data via [libosmscout](https://github.com/Framstag/libosmscout), exports them as JSON, and optionally runs inference via LibTorch.
2. **Python ML training** (`ml-training/`) — trains a PyTorch Geometric GNN (`JunctionGNN`) on those JSON graphs and exports TorchScript models consumed by C++.

The JSON files in `tmp-junctions/` are the shared contract between the two components. Each file represents one junction graph with nodes and lane-level edges.

## Architecture & Data Flow

```
OSM DB → [C++ JunctionGraphExport/RandomJunctionGraphExport]
       → tmp-junctions/*.json
       → [Python train.py] → TorchScript model (.pt)
       → [C++ PredictJunction / Python predict.py] → lane suggestions
```

- **Graph representation**: Each edge is a *lane edge* (not a highway edge). A 3-lane highway produces 3 edges + 1 virtual reverse edge. See `docs/FEATURE_MAPPING.md`.
- **Feature counts are synced across C++ and Python**: `GraphFeature::EdgeFeatureCount = 9` in `src/JunctionGraphProcessor.h` must match `EdgeFeatureCount = 9` in `ml-training/src/junction_ml/data/__init__.py`. Changing features requires updating both.
- Node features: `[lat, lon, incoming_count, outgoing_count]` (4 features, normalized).
- Graphs are normalized so the first edge points east from origin `(0,0)` — see `Graph::Normalize()` in `JunctionGraphProcessor.cpp`.

## Build & Test Commands

### C++ (requires libosmscout installed)
```bash
mkdir -p build && cd build
cmake -G "Unix Makefiles" -DCMAKE_PREFIX_PATH=/opt/osmscout/ ..
make -j $(nproc)
```
If libtorch is broken locally (e.g. Ubuntu 25.10), use Docker:
```bash
docker build -t osmscout-ml-build:latest .
```
PyTorch/LibTorch is optional — without it, only `JunctionGraphExport` and `RandomJunctionGraphExport` are built. `PredictJunction` and `EvaluateJunctions` require it.

### Python
```bash
cd ml-training
python3 -m venv .venv && source .venv/bin/activate
pip install poetry==2.1.2
poetry install
poetry run pytest   # runs pytest + mypy (strict mode, configured in pyproject.toml)
```

## Key Conventions

- **C++ standard**: C++20. All targets set via `set_property(TARGET ... PROPERTY CXX_STANDARD 20)`.
- **Python typing**: Strict mypy is enforced (`disallow_untyped_defs`, `strict_equality`, etc.). Every function must have type annotations. Tests are exempt from `disallow_untyped_defs`.
- **License**: GPL-2.0. All C++ source files carry the full license header.
- **Processor pattern**: Junction graph extraction uses libosmscout's `RoutePostprocessor::Postprocessor` interface. `JunctionGraphProcessor` is the base; `JunctionGraphExportProcessor` writes JSON; `JunctionGraphPredictProcessor` (in `PredictJunction.cpp`) runs inference. Extend this hierarchy for new processing modes.
- **JSON uses nlohmann/json** via libosmscout's bundled copy (`osmscoutclient/json/json.hpp`).

## Key Files

| File | Role |
|---|---|
| `src/JunctionGraphProcessor.{h,cpp}` | Core graph structures (`Graph`, `GraphNode`, `GraphEdge`), feature constants, JSON export/import, normalization |
| `src/RoutingUtils.h` | `ComputeRoute()` helper shared by all C++ executables |
| `ml-training/src/junction_ml/data/__init__.py` | `JunctionGraphDataset` — JSON→PyG conversion, feature counts |
| `ml-training/src/junction_ml/models/__init__.py` | `JunctionGNN` model definition |
| `ml-training/src/train.py` | Training entry point; saves TorchScript via `trainer.save_final_torchscript()` |
| `regenerate-and-retrain.sh` | End-to-end: regenerate dataset + train multiple model configurations |
| `docs/FEATURE_MAPPING.md` | Documents old (highway-level) vs new (lane-level) edge feature formats |
| `docs/copilot.md` | Build instructions for AI agents |

