# Learning a Thousand Tasks - Docker Image
# Base: Ubuntu 22.04 with CUDA support for GPU acceleration

FROM nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive

# Set working directory
WORKDIR /workspace

# Install system dependencies
RUN apt-get update && apt-get install -y \
    software-properties-common \
    && add-apt-repository ppa:deadsnakes/ppa \
    && apt-get update \
    && apt-get install -y \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3-pip \
    git \
    wget \
    curl \
    build-essential \
    cmake \
    libssl-dev \
    libffi-dev \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.9 as default python3
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.9 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1

# Upgrade pip and install build tools with pinned versions
RUN python3 -m pip install --no-cache-dir --upgrade \
    pip==23.2.1 \
    setuptools==68.0.0 \
    wheel==0.41.0

# Install PyTorch with CUDA 11.8 support first (critical for compatibility)
RUN pip install --no-cache-dir \
    torch==2.0.1 \
    torchvision==0.15.2 \
    --index-url https://download.pytorch.org/whl/cu118

# Install core dependencies with pinned versions
RUN pip install --no-cache-dir \
    numpy==1.24.3 \
    pandas==2.0.3 \
    scipy==1.10.1 \
    scikit-learn==1.3.0 \
    matplotlib==3.7.2 \
    tqdm==4.65.0 \
    imageio==2.31.1

# Install computer vision libraries
RUN pip install --no-cache-dir \
    opencv-python==4.8.0.74 \
    pillow==10.0.0

# Install 3D processing libraries
RUN pip install --no-cache-dir --ignore-installed blinker \
    open3d==0.17.0 \
    pyvista==0.42.3

# Install configuration and utilities
RUN pip install --no-cache-dir \
    pyyaml==6.0.1 \
    configargparse==1.7

# Install language models for retrieval
RUN pip install --no-cache-dir \
    huggingface-hub==0.16.4 \
    transformers==4.31.0

# Install torch-geometric with compatible versions
RUN pip install --no-cache-dir \
    torch-scatter==2.1.1 \
    torch-sparse==0.6.17 \
    torch-cluster==1.6.1 \
    torch-geometric==2.3.1 \
    -f https://data.pyg.org/whl/torch-2.0.1+cu118.html

# Install wandb for experiment tracking
RUN pip install --no-cache-dir wandb==0.15.8

# Install CLIP from official repo
RUN pip install --no-cache-dir git+https://github.com/openai/CLIP.git@a9b1bf5920416aaeaec965c25dd9e8f98c864f16

# Install lightning (the unified package) while explicitly keeping your torch version
RUN pip install --no-cache-dir lightning==2.1.4 --no-deps && \
    pip install --no-cache-dir lightning-utilities torchmetrics fsspec packaging typing-extensions \
    arrow beautifulsoup4 click croniter dateutils deepdiff fastapi inquirer Jinja2 lightning-cloud \
    psutil pydantic rich starlette starsessions uvicorn websocket-client websockets requests urllib3 traitlets

RUN pip uninstall -y numpy
RUN pip install --no-cache-dir numpy==1.24.3

# Copy the entire project
COPY . /workspace/

# Install the package in editable mode (--no-deps to avoid upgrading pinned versions)
RUN pip install -e . --no-deps

# Set environment variables for GPU support
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=compute,utility

# Set Python path
ENV PYTHONPATH=/workspace

# Create directory for outputs
RUN mkdir -p /workspace/outputs

# Install missing lightning dependencies and other utilities
RUN pip install --no-cache-dir \
    backoff==2.2.1 \
    beautifulsoup4==4.12.2 \
    websocket-client==1.6.1 \
    python-multipart==0.0.6 \
    pytorch-lightning==2.0.6 \
    msgpack==1.0.5 \
    msgpack-numpy==0.4.8 \
    numba==0.57.1

# Install XMem dependencies for demo preprocessing (BC training only)
# Note: Not needed for MT3 deployment, only for generating per-timestep masks
RUN pip install --no-cache-dir \
    progressbar2==4.2.0 \
    imageio-ffmpeg==0.4.9

# Set XMem path (will point to host-mounted directory)
ENV XMEM_PATH=/workspace/XMem

# Default command: run bash for interactive use
CMD ["/bin/bash"]
