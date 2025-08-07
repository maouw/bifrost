Bootstrap: docker
From: intel/intel-optimized-tensorflow:2.15.1-idp-base

%arguments
    BIFROST_WEIGHTS_DIR=/usr/local/share/bifrost/weights

%files
    bifrost /src/bifrost/
	pyproject.toml /src/bifrost/
    LICENSE.md /src/bifrost/
    README.md /src/bifrost/
    .condarc /src/bifrost/
    add-environment-intel.yml /src/bifrost/
    demo_dataset.tar.gz /src/bifrost/
    weights "{{BIFROST_WEIGHTS_DIR}}"

%post
	set -eEx
    export BIFROST_WEIGHTS_DIR="{{ BIFROST_WEIGHTS_DIR }}"

    # Install tools
    export DEBIAN_FRONTEND=noninteractive && \
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

    # Update conda
    export CI=1 MAMBA_ROOT_PREFIX="/opt/conda" && \
    conda config --system --set auto_update_conda false && \
    mamba update -n base -c conda-forge conda mamba

    # # Download weights if not already present
    [ -f "${BIFROST_WEIGHTS_DIR}/urls.txt" ] && wget --no-clobber -P "${BIFROST_WEIGHTS_DIR}" -i "${BIFROST_WEIGHTS_DIR}/urls.txt"

    # Install BIFROST's depencencies    
    export CONDA_DEFAULT_ENV=idp
    cd /src/bifrost && \
    cp .condarc "/opt/conda/envs/${CONDA_DEFAULT_ENV}/.condarc" && \
    conda env update -y -n "${CONDA_DEFAULT_ENV}" -f add-environment-intel.yml && \
    conda clean -y --all

    # Install BIFROST
    conda run -n "${CONDA_DEFAULT_ENV}" pip install --no-cache-dir --no-deps --no-build-isolation -e .

%environment
    CONDA_DEFAULT_ENV=idp
	PATH="/opt/conda/envs/idp/bin:$PATH"
    BIFROST_WEIGHTS_DIR="{{ BIFROST_WEIGHTS_DIR }}"
    TF_ENABLE_ONEDNN_OPTS=1
    TF_USE_LEGACY_KERAS=1
    CUDA_VISIBLE_DEVICES=''
    KMP_AFFINITY='granularity=fine,noverbose,compact,1,0'
    KMP_BLOCKTIME=200
    KMP_SETTINGS=1

%runscript
	# Run the provided command with the micromamba base environment activated:
    printf "Started at " && date -Is >&2
    exec "$@"
