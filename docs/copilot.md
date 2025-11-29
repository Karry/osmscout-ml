# build

for testing c++ part always use the following commands:
```bash
mkdir -p build && cd build
cmake -G "Unix Makefiles" -DCMAKE_PREFIX_PATH=/opt/osmscout/ ..
make -j $(nproc)
```