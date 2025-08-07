
# syntax=docker/dockerfile:1
FROM intel/intel-optimized-tensorflow:2.15.1-idp-base

# Install tools
RUN export DEBIAN_FRONTEND=noninteractive && \
    apt-get update && \
    apt-get upgrade -y && \
    apt-get install -y --no-install-recommends --fix-missing \
    apt-utils \
    bzip2 \
    git \
    gnupg2 \
    gpg-agent \
    rsync \
    unzip \
    wget \
    vim-tiny \
    zip && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# Install dependencies and setup environment
RUN export CI=1 MAMBA_ROOT_PREFIX=/opt/conda PIP_ROOT_USER_ACTION=ignore && \
    conda config --system --set auto_update_conda false && \
    mamba update -n base -c conda-forge conda mamba

# Add preloaded weights
ARG BIFROST_WEIGHTS_DIR=/usr/local/share/bifrost/weights
# Set environment variable for weights directory
ENV BIFROST_WEIGHTS_DIR="${BIFROST_WEIGHTS_DIR}"
# Add weights directory
COPY weights ${BIFROST_WEIGHTS_DIR}/

# Download weights if not already present
RUN [ -f "${BIFROST_WEIGHTS_DIR}/urls.txt" ] && wget --no-clobber -P "${BIFROST_WEIGHTS_DIR}" -i "${BIFROST_WEIGHTS_DIR}/urls.txt"

# Copy demo data
COPY demo_dataset.tar.gz /src/bifrost/

# Copy project metadata
COPY pyproject.toml add-environment-intel.yml LICENSE.md README.md .condarc /src/bifrost/

# Install BIFROST's dependencies
ENV CONDA_DEFAULT_ENV=idp
WORKDIR /src/bifrost
RUN cp .condarc "/opt/conda/envs/${CONDA_DEFAULT_ENV}/.condarc" && \
    conda env update -n "${CONDA_DEFAULT_ENV}" -f add-environment-intel.yml && \
    conda clean -y --all

# Install BIFROST
COPY bifrost /src/bifrost/bifrost
RUN export CI=1 PIP_ROOT_USER_ACTION=ignore && \
    conda run -n "${CONDA_DEFAULT_ENV}" pip install --no-cache-dir --no-deps --no-build-isolation -e .

# Set PATH
ENV PATH="/opt/conda/envs/idp/bin:$PATH"

# Set environment variables for TensorFlow
ENV TF_ENABLE_ONEDNN_OPTS=1 \
    TF_USE_LEGACY_KERAS=1 \
    CUDA_VISIBLE_DEVICES='' \
    KMP_AFFINITY='granularity=fine,noverbose,compact,1,0' \
    KMP_BLOCKTIME=200 \
    KMP_SETTINGS=1

# Set default command
CMD printf "Started at " && date -Is >&2 && exec "$@"