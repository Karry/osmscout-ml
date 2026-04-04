# build

for testing c++ part always use the following commands:
```bash
mkdir -p build && cd build
cmake -G "Unix Makefiles" -DCMAKE_PREFIX_PATH=/opt/osmscout/ ..
make -j $(nproc)
```

when compilation fails (e.g. due broken libtorch_cpu.so on Ubuntu 25.10), use docker for the testing:
```bash
docker build -t osmscout-ml-build:latest .
```

for testing python part always use the following commands:
```bash
cd ml-training
if [ ! -d ".venv" ]; then python -m venv .venv ; source venv/bin/activate ; pip install poetry==2.1.2 ; fi
source ./.venv/bin/activate
poetry install
poetry run pytest
```
