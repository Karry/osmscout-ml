#!/bin/bash

set -e

EPOCHS=100
LAYERS=6
LOSS=dice
MODEL_DIR="model-${LOSS}-${EPOCHS}-epochs-${LAYERS}-layers"

cd "$(dirname "$0")"
cd "./ml-training"
rm -rf output
mkdir output

source .venv/bin/activate

echo "============================================"
echo "Training model: ${MODEL_DIR}"
echo "============================================"

python3 ./src/train.py \
  --data-dir ../tmp-junctions/ \
  --log-dir output/log \
  --save-dir "output/${MODEL_DIR}" \
  --verbose \
  --epochs $EPOCHS \
  --num-layers $LAYERS \
  --loss $LOSS \
  --focal-alpha 0.24 \
  --focal-gamma 10.0 \
  --dice-smooth 1.0 \
  --dice-bce-weight 0.5


cd ".."

echo "============================================"
echo "Rebuilding Docker image"
echo "============================================"

docker build -t osmscout-ml-build:latest .

echo ""
echo "============================================"
echo "✓ Docker image rebuilt successfully!"
echo "============================================"
echo ""
echo "Test the C++ predict tool:"
echo ""

docker run \
  -v $HOME/Maps:/maps \
  -v $(pwd)/ml-training:/workspace/ml-training \
  -v $(pwd)/tmp-junctions:/workspace/tmp-junctions \
  -it osmscout-ml-build:latest \
  /workspace/osmscout-ml/cmake-build-release/PredictJunction \
  --model "/workspace/ml-training/output/${MODEL_DIR}/best_torchscript.pt" \
  --debug --routeDebug \
  /maps/europe-czech-republic-20251215-015203 \
  50.007487 14.55557 49.99849 14.58843


docker run \
  -v $HOME/Maps:/maps \
  -v $(pwd)/ml-training:/workspace/ml-training \
  -v $(pwd)/tmp-junctions:/workspace/tmp-junctions \
  -it osmscout-ml-build:latest \
    /workspace/osmscout-ml/cmake-build-release/EvaluateJunctions \
    /workspace/tmp-junctions \
    "/workspace/ml-training/output/${MODEL_DIR}/best_torchscript.pt"
