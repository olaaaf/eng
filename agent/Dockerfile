# Use a base image that has the necessary build tools and C++ environment
FROM ubuntu:latest as builder

# Install build dependencies
RUN apt-get update && apt-get install -y \
    g++ \
    cmake \
    unzip \
    curl \
    python3-pip && \
    pip3 install meson \
    pip3 install wandb

# Download libtorch
RUN curl -L -o libtorch.zip "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.2.2+cpu.zip" && \
    unzip libtorch.zip -d /libtorch && \
    rm libtorch.zip

# Set the environment variables for CMake to find LibTorch
ENV Torch_DIR=/libtorch/share/cmake/Torch

# Copy your project files
COPY . /app
WORKDIR /app

# Build your project with Meson
RUN meson builddir && \
    ninja -C builddir

# Copy the mario.nes file to the build directory
COPY mario.nes /app/

# Start a new, final image to reduce size
FROM ubuntu:latest

# Copy the built binary and necessary libraries from the builder image
COPY --from=builder /app/builddir /app
COPY --from=builder /libtorch /libtorch

# Environment variables needed at runtime
ENV LD_LIBRARY_PATH=/libtorch/lib:$LD_LIBRARY_PATH

# Change to app directory
WORKDIR /app

# Command to run your binary
CMD ["./MedNES"]
