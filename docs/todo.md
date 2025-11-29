# Lane-Level Graph Architecture - Work Plan

## Overview
Transform the junction graph architecture from **edge-per-highway** to **edge-per-lane**, predicting only a binary `suggested` feature for each lane.

## Current Architecture
- **Graph Structure**: One edge per highway segment (regardless of lane count)
- **Features**: 18 edge features including lane-specific data (laneTurn0-9)
- **Predictions**: 3 outputs per edge:
  - `suggestedFrom` (lane index)
  - `suggestedTo` (lane index) 
  - `suggestedTurn` (turn direction)

## Target Architecture
- **Graph Structure**: One edge per lane (highway with N lanes → N edges)
- **New Feature**: `relativeLanePosition` (normalized 0.0-1.0, left to right)
- **Prediction**: 1 binary output per edge:
  - `suggested` (boolean: is this lane suggested for the route?)

## Benefits
- Simpler prediction task (binary classification vs multi-task regression)
- Natural representation: lanes as first-class graph entities
- Better feature locality: lane properties (turn direction, position) attached to the lane itself
- Easier for GNN to learn: one decision per lane edge

---

## Implementation Plan

### Phase 1: C++ Graph Generation Changes

#### 1.1 Update Graph Data Structures (`JunctionGraphProcessor.h`)
- [ ] Update `GraphFeature::EdgeFeatureCount` from 18 to 10 (remove 10 laneTurn features, add 2 new ones)
- [ ] Remove feature constants:
  - `SUGGESTED_FROM`
  - `SUGGESTED_TO`
  - `SUGGESTED_TURN`
- [ ] Add new feature constants:
  - `RELATIVE_LANE_POSITION` (float 0.0-1.0)
  - `LANE_TURN` (single lane turn direction, not array)
  - `SUGGESTED` (binary: 0.0 or 1.0)

**New edge feature list (10 features):**
1. `length`
2. `laneCount` (total lanes on the highway - same for all lane edges)
3. `angle` (turn angle at junction - same for all lane edges)
4. `oneway`
5. `route` (highway is part of route - same for all lane edges)
6. `type` (highway type)
7. `usable` (lane/highway is usable by vehicle)
8. `virtual` (virtual edge for reverse direction)
9. `relativeLanePosition` ⭐ NEW
10. `laneTurn` ⭐ NEW (single value, not array)

**Question 1:** Should we keep the `oneway` feature at the edge level or is it redundant with lane-level representation?

#### 1.2 Modify Edge Generation (`JunctionGraphProcessor.cpp`)
- [ ] Refactor `MakeEdge()` function to `MakeLaneEdges()`:
  - Returns `std::vector<GraphEdge>` instead of `std::pair<GraphEdge, GraphEdge>`
  - For each lane in `laneDesc`:
    - Create a separate `GraphEdge`
    - Set `relativeLanePosition` = `laneIndex / max(1, laneCount - 1)`
    - Set `laneTurn` from `laneDesc->GetLaneTurns()[laneIndex]`
    - Set `suggested` based on lane index being in range `[suggestedFrom, suggestedTo]`
    - Copy other features (length, angle, type, etc.)
  - Create virtual reverse edges (one per lane or just one?)

**Question 2:** For virtual reverse edges, should we create one per lane or just one aggregate edge? (Suggest: one aggregate, since reverse direction doesn't have route/suggestion info)

- [ ] Update `TraverseWay()` function:
  - Generate lane edges for alternate routes at junctions
  - Handle cases where lane information is missing (fallback to single edge?)

**Question 3:** What should we do when lane information is not available for a way? Options:
  - A) Generate a single edge with `relativeLanePosition = 0.5` (middle)
  - B) Skip edge generation (might lose graph connectivity)
  - C) Assume 1 lane with position 0.0
  
  **Recommendation: Option A** - ensures connectivity while signaling "unknown" with middle position.

- [ ] Update the main `Process()` loop:
  - Handle multiple edges per highway segment
  - Ensure graph connectivity is maintained

#### 1.3 Update JSON Export
- [ ] Modify `Graph::Export()` to use new feature names
- [ ] Ensure lane edges are properly serialized

---

### Phase 2: Python Data Loading Changes

#### 2.1 Update Constants (`ml-training/src/junction_ml/data/__init__.py`)
- [ ] Update `EdgeFeatureCount` from 18 to 9
- [ ] Update feature extraction in `_convert_to_pyg_data()`:
  - Remove laneTurn0-9 loop
  - Remove `oneway` feature
  - Add `relativeLanePosition`
  - Add single `laneTurn`
  - Remove target labels: `suggestedFrom`, `suggestedTo`, `suggestedTurn`
  - Add single target label: `suggested`

#### 2.2 Update Data Object Structure
- [ ] Modify `Data` object creation:
  - Replace `y_suggested_from`, `y_suggested_to`, `y_suggested_turn` with single `y_suggested`
  - Replace `valid_from`, `valid_to`, `valid_turn` with single `valid_suggested`
  - Update validation masks

#### 2.3 Calculate Class Weights
- [ ] Add function to calculate positive class weight:
  - Count total suggested vs non-suggested lanes across dataset
  - Calculate `pos_weight = num_negative / num_positive`
  - Store in dataset metadata
  - Log statistics for debugging

---

### Phase 3: Model Architecture Changes

#### 3.1 Update Model (`ml-training/src/junction_ml/models/__init__.py`)
- [ ] Update `JunctionGNN.__init__()`:
  - Remove multi-task prediction heads
  - Add single binary classification head:
    ```python
    self.suggested_head = nn.Linear(hidden_dim // 2, 1)
    ```
- [ ] Update `forward()` method:
  - Return single prediction tensor instead of dict
  - Apply sigmoid activation for binary classification

#### 3.2 Update Loss Function (`ml-training/src/junction_ml/training/__init__.py`)
- [ ] Replace `MultiTaskLoss` with binary classification loss:
  - Use `BCEWithLogitsLoss` or weighted version
  - Handle class imbalance (likely more non-suggested lanes than suggested)
  
**Question 4:** Should we use weighted loss to handle class imbalance? Typical ratio might be 1 suggested lane per 5-10 total lanes at a junction.

#### 3.3 Update Training Loop
- [ ] Modify loss computation to use single target
- [ ] Update metrics calculation:
  - Binary accuracy
  - Precision, Recall, F1 for suggested lanes
  - Remove lane range prediction metrics

---

### Phase 4: Training Scripts Updates ✅

#### 4.1 Update Training Script (`ml-training/src/train.py`) ✅
- [x] Load pos_weight from dataset metadata
- [x] Pass pos_weight to create_trainer
- [x] Update to work with new single-output model

#### 4.2 Update Prediction Script (`ml-training/src/predict.py`) ⚠️
- [ ] Update output processing for binary predictions
- [ ] Modify visualization to show per-lane suggestions
- [ ] Update evaluation metrics

#### 4.3 Update Explanation Script (`ml-training/src/explain.py`) ⚠️
- [ ] Update SHAP/explanation code for new features
- [ ] Add visualization for lane-level predictions
- [ ] Update feature importance analysis (now includes relativeLanePosition)

---

### Phase 5: C++ Prediction Tool Updates ✅

#### 5.1 Update Prediction Code (`src/PredictJunction.cpp`) ✅
- [x] Update tensor creation to match new feature count (9 features)
- [x] Modify input preparation:
  - Handle multiple edges per highway
  - Process lane-level edges
- [x] Update output parsing:
  - Single prediction tensor (suggested probability per edge)
  - Remove tuple unpacking of multiple outputs
- [x] Update output display:
  - Show lane-by-lane suggestions
  - Group lanes by highway for readability
  - Show lane position and turn direction with suggestion

**Display format example:**
```
Highway from Node A to Node B (3 lanes):
  Lane 0 (left,  turn: slight-left):  suggested=0.92 ✓
  Lane 1 (mid,   turn: through):      suggested=0.89 ✓
  Lane 2 (right, turn: through):      suggested=0.12
  Heuristic: lanes 0-1, turn: through
```

---

### Phase 6: Testing & Validation

#### 6.1 Generate Test Data
- [ ] Clear existing processed data cache
- [ ] Regenerate junction graphs with new architecture
- [ ] Verify lane edges are created correctly
- [ ] Check edge counts (should be ~3-5x higher on average)

#### 6.2 Training & Evaluation
- [ ] Train new model from scratch
- [ ] Compare performance metrics with old approach
- [ ] Test on diverse junction types (T-junction, 4-way, roundabout, etc.)

#### 6.3 Integration Testing
- [ ] Test C++ prediction tool with new model
- [ ] Verify end-to-end pipeline works
- [ ] Check edge cases (missing lane info, single lane, etc.)

---

### Phase 7: Documentation Updates

#### 7.1 Update Documentation
- [ ] Update `docs/junction-lanes.md`:
  - Describe new architecture
  - Explain lane-level representation
  - Document new features
- [ ] Update README files with examples
- [ ] Add migration notes for existing models

#### 7.2 Code Comments
- [ ] Add comments explaining lane edge generation
- [ ] Document relativeLanePosition calculation
- [ ] Explain design decisions

---

## Questions Summary (Please Review)

1. **Oneway Feature**: Keep at edge level or remove as redundant?
   - *Recommendation*: Keep it - provides highway context

2. **Virtual Reverse Edges**: One per lane or one aggregate?
   - *Recommendation*: One aggregate (saves memory, reverse has no lane info)

3. **Missing Lane Info Fallback**: How to handle?
   - *Recommendation*: Single edge with `relativeLanePosition = 0.5`

4. **Class Imbalance**: Use weighted loss?
   - *Recommendation*: Yes, weight positive class higher (suggested lanes are minority)

5. **Additional consideration**: Should we keep original highway edges as "super-edges" in addition to lane edges for GNN message passing?
   - *Recommendation*: No, cleaner to have only lane-level edges

---

## Estimated Effort

- **Phase 1** (C++ Graph Generation): ~4-6 hours
- **Phase 2** (Python Data Loading): ~2-3 hours  
- **Phase 3** (Model Architecture): ~2-3 hours
- **Phase 4** (Training Scripts): ~2-3 hours
- **Phase 5** (C++ Prediction): ~3-4 hours
- **Phase 6** (Testing): ~4-6 hours
- **Phase 7** (Documentation): ~2-3 hours

**Total**: ~19-28 hours

---

## Next Steps

1. **Review this plan** and answer the questions above
2. **Approve approach** or request modifications
3. **Begin implementation** starting with Phase 1

