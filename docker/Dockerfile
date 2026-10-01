FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    UV_SYSTEM_PYTHON=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    libgdal-dev \
    fonts-noto-core \
    fonts-noto-cjk \
    && rm -rf /var/lib/apt/lists/*

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Copy project definition and install dependencies
COPY pyproject.toml README.md /app/
RUN uv pip install -e ".[dev]"

# Copy source tree
COPY src/ /app/src/
COPY evaluation/ /app/evaluation/
COPY training/ /app/training/
COPY tests/ /app/tests/

# Default entrypoint runs tests then CLI
CMD ["pytest", "tests/"]
