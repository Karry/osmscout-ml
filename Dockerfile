FROM ubuntu:25.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Install build dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    git \
    wget \
    unzip \
    pkg-config \
    # libosmscout dependencies
    libxml2-dev \
    libprotobuf-dev \
    protobuf-compiler \
    # Optional libosmscout dependencies
    libcairo2-dev \
    libpango1.0-dev \
    libmarisa-dev \
    # PyTorch C++ API dependencies
    libtorch-dev \
    # Cleanup
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Clone and build libosmscout
RUN git clone https://github.com/Framstag/libosmscout.git && \
    cd libosmscout && \
    mkdir build && cd build && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_INSTALL_PREFIX=/usr/local \
          -DOSMSCOUT_BUILD_MAP=ON \
          -DOSMSCOUT_BUILD_CLIENT_QT=OFF \
          -DOSMSCOUT_BUILD_TESTS=OFF \
          -DOSMSCOUT_BUILD_DEMOS=OFF \
          -DOSMSCOUT_BUILD_BINDING_JAVA=OFF \
          .. && \
    make -j$(nproc) && \
    make install && \
    cd ../.. && rm -rf libosmscout

# Copy the project files
COPY src /workspace/osmscout-ml/src
COPY CMakeLists.txt /workspace/osmscout-ml/

# Build the C++ project
WORKDIR /workspace/osmscout-ml
RUN mkdir -p cmake-build-release && cd cmake-build-release && \
    cmake -DCMAKE_BUILD_TYPE=Release \
          -DCMAKE_PREFIX_PATH=/usr/local \
          .. && \
    make -j$(nproc)

# Set the default command to bash for interactive use
CMD ["/bin/bash"]

