# Use Ubuntu 24.04 as the base image
FROM --platform=linux/amd64 ubuntu:24.04

# Set environment variables for non-interactive apt installs and vcpkg
ENV DEBIAN_FRONTEND=noninteractive
ENV VCPKG_FORCE_SYSTEM_BINARIES=true

# Install basic dependencies
RUN apt update && apt upgrade -y && apt install -y \
    bash \
    curl \
    git \
    wget \
    zip \
    unzip \
    tar \
    build-essential \
    clang \
    python3 \
    pkg-config \
    clang-tools \
    cmake \
    ninja-build \
    linux-headers-generic \
    && rm -rf /var/lib/apt/lists/*

# Set environment variables for vcpkg and Clang
ENV CXX=/usr/bin/clang++
ENV CC=/usr/bin/clang

# Download and install libtorch
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcpu.zip -O libtorch.zip \
    && unzip libtorch.zip -d /opt/ \
    && rm libtorch.zip

# Copy the libtorch files to the appropriate system directories
RUN cp -R /opt/libtorch/include/* /usr/include/ \
    && cp -R /opt/libtorch/share/* /usr/share/ \
    && cp -R /opt/libtorch/lib/* /usr/lib/

RUN rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["bash"]