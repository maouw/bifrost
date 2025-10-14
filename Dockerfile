
# syntax=docker/dockerfile:1
FROM ghcr.io/astral-sh/uv:debian

# Copy project metadata
COPY pyproject.toml uv.lock LICENSE.md pytest.ini README.md /app/
COPY src/ /app/src/
COPY tests/ /app/tests/

# Sync the project into a new environment, asserting the lockfile is up to date
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
RUN uv sync --frozen --no-dev 

# Set environment variables for TensorFlow
ENV TF_ENABLE_ONEDNN_OPTS=1 \
    TF_USE_LEGACY_KERAS=1 \
    KMP_AFFINITY='granularity=fine,noverbose,compact,1,0' \
    KMP_BLOCKTIME=200 \
    KMP_SETTINGS=1

# Set default command
CMD ["/bin/sh", "-c", "printf 'Started at ' && date -Is >&2 && exec \"$@\"", "--"]
