# OSMScout machine learning experiments 

 - dependency: [libosmscout](https://github.com/Framstag/libosmscout)

This repository is playground for machine learning experiments with OpenStreetMap data. 
My plan is to explore possibilities of navigation improvements in [harbour-osmscout application](https://github.com/Karry/osmscout-sailfish/) 
with usage of machine learning.

## Build C++ tools:
```bash
mkdir build && cd build
cmake -DCMAKE_BUILD_TYPE=Debug -DCMAKE_PREFIX_PATH=<libosmscout-install-dir> ..
make -j $(nproc)
```
