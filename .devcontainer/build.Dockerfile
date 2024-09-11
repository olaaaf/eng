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

#RUN wget https://ftp.gnu.org/gnu/gdb/gdb-10.1.tar.gz -O gdb.tar.gz \
#    && tar -xf gdb.tar.gz && cd gdb-10.1 && ./configure && make && make install \
#    && cd /root/

# Set environment variables for vcpkg and Clang
ENV CXX=/usr/bin/clang++
ENV CC=/usr/bin/clang

# Download and install libtorch
WORKDIR /root/
RUN wget https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.4.1%2Bcpu.zip -O libtorch.zip \
    && unzip libtorch.zip -d /opt/ \
    && rm libtorch.zip

RUN wget https://github.com/conan-io/conan/releases/download/2.6.0/conan-2.6.0-linux-x86_64.tgz \ 
    && tar -xvf conan-2.6.0-linux-x86_64.tgz && cp -R bin/* /usr/bin/

RUN mkdir -p /root/.conan2/profiles
COPY default.profile /root/.conan2/profiles/default

RUN /usr/bin/conan install --requires=drogon/1.9.6 --build=missing

# Copy the libtorch files to the appropriate system directories
RUN cp -R /opt/libtorch/include/* /usr/include/ \
    && cp -R /opt/libtorch/share/* /usr/share/ \
    && cp -R /opt/libtorch/lib/* /usr/lib/

RUN rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

CMD ["bash"]