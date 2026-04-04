# Lane-Level Graph Architecture - Implementation Summary

## Date: 2025-11-29

## Overview
Successfully transformed the junction graph architecture from **edge-per-highway** to **edge-per-lane**, simplifying the prediction task from multi-task regression to binary classification.

## What Was Changed

### 1. C++ Graph Generation (✅ COMPLETED)

#### `src/JunctionGraphProcessor.h`
- Updated `EdgeFeatureCount` from 18 to 9
- Removed obsolete constants: `ONEWAY`, `SUGGESTED_FROM`, `SUGGESTED_TO`, `SUGGESTED_TURN`
- Added new constants: `RELATIVE_LANE_POSITION`, `LANE_TURN`, `SUGGESTED`
- Changed string constants from `constexpr` to `inline const` for C++17/20 compatibility

#### `src/JunctionGraphProcessor.cpp`
- **Replaced `MakeEdge()` with `MakeLaneEdges()`**:
  - Returns `std::vector<GraphEdge>` (multiple edges) instead of pair
  - Creates one edge per lane with lane-specific features
  - Calculates `relativeLanePosition` = `laneIndex / (laneCount - 1)`
  - Sets `laneTurn` from lane description
  - Sets `suggested` = 1.0 if lane in suggested range, 0.0 otherwise
  - Creates ONE virtual reverse edge per highway (not per lane)
  
- **Updated `TraverseWay()`**:
  - Generates lane-level edges for alternate routes
  - Handles missing lane info with warning + assumes single lane
  
- **Updated `Process()` loop**:
  - Handles multiple edges returned from `MakeLaneEdges()`
  - Maintains graph connectivity

#### `src/PredictJunction.cpp`
- Updated feature extraction to use 9 features (removed oneway, laneTurn0-9 array)
- Changed model output parsing from tuple of 3 tensors to single tensor
- Improved output display:
  - Groups lanes by highway
  - Shows per-lane predictions with position and turn direction
  - Compares model prediction vs heuristic

### 2. Python Data Loading (✅ COMPLETED)

#### `ml-training/src/junction_ml/data/__init__.py`
- Updated `EdgeFeatureCount` from 18 to 9
- Modified `_convert_to_pyg_data()`:
  - Removed `oneway` and `laneTurn0-9` features
  - Added `relativeLanePosition` and single `laneTurn`
  - Changed from 3 labels to 1 label: `y_suggested`
  - Updated Data object structure
  
- **Added class weight calculation**:
  - Calculates `pos_weight = num_negative / num_positive`
  - Stores in dataset metadata
  - Logs statistics for debugging

### 3. Model Architecture (✅ COMPLETED)

#### `ml-training/src/junction_ml/models/__init__.py`
- **Simplified `JunctionGNN`**:
  - Removed 3 prediction heads (`suggested_from_head`, `suggested_to_head`, `suggested_turn_head`)
  - Added single head: `suggested_head = nn.Linear(hidden_dim // 2, 1)`
  - Changed `forward()` return type from `Dict[str, Tensor]` to `Tensor`
  - Updated `_predict_edges()` to return single tensor
  
- **Updated `JunctionGNNTorchScript`**:
  - Changed return type from `Tuple[Tensor, Tensor, Tensor]` to `Tensor`
  - Returns single prediction tensor for C++ compatibility

### 4. Training Code (✅ COMPLETED)

#### `ml-training/src/junction_ml/training/__init__.py`
- **Replaced `MultiTaskLoss` with `BinarySuggestedLoss`**:
  - Uses `BCEWithLogitsLoss` with `pos_weight` parameter
  - Handles class imbalance automatically
  
- **Updated `JunctionTrainer`**:
  - Modified `train_epoch()` to use single label
  - Modified `validate_epoch()` to use single label
  - Added binary classification metrics: accuracy, precision, recall, F1
  - Removed multi-task metric tracking
  
- **Updated `create_trainer()`**:
  - Accepts `pos_weight` parameter
  - Creates `BinarySuggestedLoss` instead of `MultiTaskLoss`

#### `ml-training/src/train.py`
- Loads `pos_weight` from dataset metadata
- Passes it to `create_trainer()`

## New Feature Set (9 Features)

1. **length** - Highway segment length
2. **laneCount** - Total lanes on highway (same for all lane edges)
3. **angle** - Turn angle at junction (same for all lane edges)
4. **route** - Is highway part of route? (same for all lane edges)
5. **type** - Highway type
6. **usable** - Is lane/highway usable?
7. **virtual** - Is this a virtual reverse edge?
8. **relativeLanePosition** ⭐ NEW - Lane position: 0.0 (left) to 1.0 (right)
9. **laneTurn** ⭐ NEW - Turn direction for this specific lane

## Prediction Task

**Before**: Multi-task regression
- Predict `suggestedFrom` (lane index)
- Predict `suggestedTo` (lane index)
- Predict `suggestedTurn` (turn direction)

**After**: Binary classification
- Predict `suggested` (0 or 1) - Is this lane suggested for the route?

## Key Design Decisions

1. **Oneway Feature**: REMOVED - redundant at lane level
2. **Virtual Reverse Edges**: ONE aggregate edge per highway
3. **Missing Lane Info**: Print WARNING, assume single lane at position 0.0
4. **Class Imbalance**: Use weighted BCE with `pos_weight = num_negative / num_positive`
5. **Super-edges**: NO - use only lane-level edges

## Compilation Status

### C++ Code
- ✅ `RoutingUtils` (library) - Compiled successfully
- ✅ `JunctionGraphExport` - Compiled successfully
- ✅ `RandomJunctionGraphExport` - Compiled successfully
- ⚠️ `PredictJunction` - Code compiles, linker error (known gloo library issue, not our code)

### Python Code
- ✅ No syntax errors
- ✅ All core modules updated
- ⚠️ `predict.py` and `explain.py` still need updates (Phase 4.2, 4.3)

## What Still Needs to Be Done

### Phase 6: Testing & Validation (TODO)
- [ ] Clear existing processed data cache
- [ ] Regenerate junction graphs with new architecture
- [ ] Verify lane edges are created correctly
- [ ] Train new model from scratch
- [ ] Compare performance with old approach
- [ ] Test on diverse junction types

### Phase 7: Documentation (TODO)
- [ ] Update `docs/junction-lanes.md` with new architecture
- [ ] Add examples and migration notes

### Remaining Python Scripts (TODO)
- [ ] Update `ml-training/src/predict.py` for binary predictions
- [ ] Update `ml-training/src/explain.py` for new features

## Next Steps

1. **Regenerate dataset**:
   ```bash
   cd /home/karry/data/cecko/OSM/osmscout-ml
   rm -rf ml-training/processed/*
   ./cmake-build-debug/JunctionGraphExport --help
   # Run with your parameters to regenerate junction graphs
   ```

2. **Train new model**:
   ```bash
   cd ml-training
   python src/train.py --data-dir ../tmp-junctions --epochs 50
   ```

3. **Monitor training**:
   - Check that `pos_weight` is calculated and logged
   - Watch accuracy, precision, recall, F1 metrics
   - Expect ~3-5x more edges in dataset (one per lane)

4. **Test predictions** (once model trained):
   ```bash
   ./cmake-build-debug/PredictJunction --model ml-training/checkpoints/final_model.pt
   ```

## Expected Benefits

- **Simpler Task**: Binary classification easier than multi-task regression
- **Better Representation**: Lanes as first-class entities
- **More Training Data**: ~3-5x more edges (one per lane)
- **Better Interpretability**: Clear per-lane suggestions
- **Natural Features**: Lane position and turn direction attached to each lane

## Performance Expectations

- **Edge Count**: ~3-5x increase (good for training)
- **Class Imbalance**: ~1:5 to 1:10 ratio (handled by pos_weight)
- **Accuracy Target**: Should improve over baseline as task is simpler
- **F1 Score**: Key metric for suggested lanes (minority class)

