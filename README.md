# OSMScout machine learning experiments 

 - dependency: [libosmscout](https://github.com/Framstag/libosmscout)

This repository is playground for machine learning experiments with OpenStreetMap data. 
My plan is to explore possibilities of navigation improvements in [harbour-osmscout application](https://github.com/Karry/osmscout-sailfish/) 
with usage of machine learning.

## Junction Lane Prediction

### Quick Start

1. **Build C++ tools**:
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=<libosmscout-install-dir> ..
make -j $(nproc)
```

2. **Generate dataset**:
```bash
# for specific route
./build/JunctionGraphExport --junctionExportDir tmp-junctions /path/to/osm/db start-lat start-lon end-lat end-lon
# or generate batch of random junctions
./build/RandomJunctionGraphExport --junctionExportDir  tmp-junctions /path/to/osm/db
```

3. **Train model**:
```bash
cd ml-training
python3 src/train.py --data-dir ../tmp-junctions/ --epochs 500 --num-layers 3
```

4. **Make predictions**:
```bash
python3 src/predict.py --model output/MODEL/best.pt --input ../tmp-junctions/FILE.json
```
5. **Explain predictions**:
```bash
python3 src/explain.py --model output/MODEL/best.pt --input ../tmp-junctions/FILE.json --method gradients --top-k 5
```
