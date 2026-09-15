FROM ghcr.io/astral-sh/uv:python3.14-bookworm-slim

WORKDIR /app

# Copy dependency files first (cache layer)
COPY pyproject.toml uv.lock README.md ./

# Install dependencies (without the project itself)
RUN uv sync --frozen --no-dev --no-install-project

# Copy source
COPY src/ src/

# Install the project
RUN uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
ENV PYTHONUNBUFFERED=1

ENTRYPOINT ["plainrag"]
CMD ["--help"]
