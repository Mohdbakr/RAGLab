# Build context is the repo root (see ../../docker-compose.yml) so this
# image can also COPY the platform/common sibling package that the
# orchestrator depends on as an editable path dependency.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# docker compose CLI only — no daemon runs in this image. At runtime the
# host's /var/run/docker.sock is mounted in (see the root docker-compose.yml)
# so every `docker compose ...` this process runs controls the HOST's
# Docker, which is what lets it start/stop sibling project containers.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl gnupg \
    && install -m 0755 -d /etc/apt/keyrings \
    && curl -fsSL https://download.docker.com/linux/debian/gpg -o /etc/apt/keyrings/docker.asc \
    && chmod a+r /etc/apt/keyrings/docker.asc \
    && echo "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.asc] https://download.docker.com/linux/debian bookworm stable" \
        > /etc/apt/sources.list.d/docker.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        docker-ce-cli docker-compose-plugin docker-buildx-plugin \
    && rm -rf /var/lib/apt/lists/*

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=platform/orchestrator/uv.lock,target=uv.lock \
    --mount=type=bind,source=platform/orchestrator/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=platform/common,target=/common \
    uv sync --frozen --no-install-project --no-dev

COPY platform/common /common
COPY platform/orchestrator /app
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8100
CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "8100"]
