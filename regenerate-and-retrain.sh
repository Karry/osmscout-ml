#!/bin/bash -xe
cd $(dirname "$0")

rm -rf ./tmp-junctions
rm -rf "./???"
mkdir ./tmp-junctions

./cmake-build-debug/RandomJunctionGraphExport \
  --junctionExportDir ./tmp-junctions \
  /home/karry/Maps/europe-czech-republic-20251007-015742

./cmake-build-debug/JunctionGraphExport --debug --routeDebug \
  --junctionExportDir ./tmp-junctions \
  /home/karry/Maps/europe-czech-republic-20251007-015742 \
  50.007487 14.55557 49.99849 14.58843

ls -1 ./tmp-junctions | wc -l

cd ./ml-training
rm -rf output
mkdir output

source .venv/bin/activate

python3 ./src/train.py \
  --data-dir ../tmp-junctions/ \
  --log-dir output/log \
  --save-dir output/model-500-epochs-3-layers \
  --epochs 500 \
  --num-layers 3 \
  --verbose

python3 ./src/train.py \
  --data-dir ../tmp-junctions/ \
  --log-dir output/log \
  --save-dir output/model-500-epochs-4-layers \
  --epochs 500 \
  --num-layers 4 \
  --verbose

python3 ./src/train.py \
  --data-dir ../tmp-junctions/ \
  --log-dir output/log \
  --save-dir output/model-500-epochs-5-layers \
  --epochs 500 \
  --num-layers 5 \
  --verbose

python3 ./src/train.py \
  --data-dir ../tmp-junctions/ \
  --log-dir output/log \
  --save-dir output/model-500-epochs-10-layers \
  --epochs 500 \
  --num-layers 10 \
  --verbose
