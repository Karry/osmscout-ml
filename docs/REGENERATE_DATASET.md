# ⚠️ IMPORTANT: Dataset Must Be Regenerated!

## Current Status

✅ **Training Script**: Fixed and working correctly  
❌ **Dataset**: Still in OLD format - needs regeneration

## The Problem

The train script will run with old data but **WILL NOT WORK CORRECTLY** because:

1. Old data has 18 features per edge (including `laneTurn0-9` array)
2. New model expects 9 features per edge (including `relativeLanePosition` and single `laneTurn`)
3. Old data uses one edge per highway
4. New model expects one edge per lane

When you train with old data:
- Features will be silently filled with 0.0 defaults
- Model will appear to train (loss goes down)
- But predictions will be meaningless
- You'll see `pos_weight=1.0` (wrong - should be 5-10)

## How to Fix

### Step 1: Rebuild C++ Tools

The C++ code has been updated but you need to rebuild:

```bash
cd /home/karry/data/cecko/OSM/osmscout-ml/cmake-build-debug
make clean
make
```

You should see:
```
Built target RoutingUtils
Built target JunctionGraphExport
Built target RandomJunctionGraphExport
```

### Step 2: Regenerate Dataset

```bash
cd /home/karry/data/cecko/OSM/osmscout-ml

# Backup old data (optional)
mv tmp-junctions tmp-junctions-old

# Create new directory
mkdir tmp-junctions

# Run the updated export tool with your parameters
# Example:
./cmake-build-debug/JunctionGraphExport \
    --database /path/to/your/osm/database \
    --output tmp-junctions \
    [other parameters]

# Or use RandomJunctionGraphExport:
./cmake-build-debug/RandomJunctionGraphExport \
    --database /path/to/your/osm/database \
    --output tmp-junctions \
    --count 1000
```

### Step 3: Verify New Format

Check a sample file to confirm it has the new format:

```bash
python3 -c "
import json
with open('tmp-junctions/[any-file].json') as f:
    data = json.load(f)
    edge = data['edges'][0]
    print('Has relativeLanePosition:', 'relativeLanePosition' in edge)
    print('Has suggested:', 'suggested' in edge)
    print('Has laneTurn (single):', 'laneTurn' in edge)
    print('Has OLD format (laneTurn0):', 'laneTurn0' in edge)
"
```

Expected output:
```
Has relativeLanePosition: True
Has suggested: True
Has laneTurn (single): True
Has OLD format (laneTurn0): False
```

### Step 4: Clear Processed Data

```bash
rm -rf tmp-junctions/processed
```

### Step 5: Train Model

Now you can train with the correct data:

```bash
cd ml-training

python3 ./src/train.py \
  --data-dir ../tmp-junctions/ \
  --log-dir output/log \
  --save-dir output/model-500-epochs-3-layers \
  --epochs 500 \
  --num-layers 3 \
  --verbose
```

You should see:
```
Using pos_weight=5.47 for handling class imbalance  # <-- Should be > 1.0
```

## What Changed in the Data Format

### Old Format (one edge per highway):
```json
{
  "from": 12345,
  "to": 67890,
  "laneCount": 3,
  "laneTurn0": 1.0,
  "laneTurn1": 17.0,
  "laneTurn2": 15.0,
  "suggestedFrom": 0,
  "suggestedTo": 1,
  "suggestedTurn": 17.0,
  ...
}
```

### New Format (multiple edges, one per lane):
```json
// Lane 0
{
  "from": 12345,
  "to": 67890,
  "laneCount": 3,
  "relativeLanePosition": 0.0,
  "laneTurn": 1.0,
  "suggested": 1.0,
  ...
}
// Lane 1
{
  "from": 12345,
  "to": 67890,
  "laneCount": 3,
  "relativeLanePosition": 0.5,
  "laneTurn": 17.0,
  "suggested": 1.0,
  ...
}
// Lane 2
{
  "from": 12345,
  "to": 67890,
  "laneCount": 3,
  "relativeLanePosition": 1.0,
  "laneTurn": 15.0,
  "suggested": 0.0,
  ...
}
```

## Expected Results After Regeneration

- **More edges**: 3-5x increase (one per lane instead of one per highway)
- **pos_weight**: Should be 5-10 (more non-suggested lanes than suggested)
- **Better training**: Model should learn lane-level predictions

## Training Script Status

✅ **FIXED** - The following issues have been resolved:

1. ✅ Model architecture updated to single binary classification head
2. ✅ Loss function changed to `BinarySuggestedLoss` with pos_weight
3. ✅ Logging updated to use 'loss' instead of 'total_loss'
4. ✅ Metrics updated to show accuracy, precision, recall, F1
5. ✅ TorchScript export updated for single output
6. ✅ Data validation warnings added

You can now run the train script with:
```bash
python3 ./src/train.py \
  --data-dir ../tmp-junctions/ \
  --log-dir output/log \
  --save-dir output/model-500-epochs-3-layers \
  --epochs 500 \
  --num-layers 3 \
  --verbose
```

**BUT** you must regenerate the dataset first for it to work correctly!

## Need Help?

See these docs:
- `docs/QUICKSTART.md` - Full setup guide
- `docs/IMPLEMENTATION_SUMMARY.md` - What changed
- `docs/FEATURE_MAPPING.md` - Old vs new format

