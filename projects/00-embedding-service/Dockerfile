# Build context is the repo root (see docker-compose.yml in this folder)
# so this image can also COPY the platform/common sibling package this
# service depends on as an editable path dependency.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# The pyproject.toml's `raglab-common = {path = "../../platform/common"}`
# is resolved relative to WORKDIR (/app), so the sibling package needs to
# land two directories up from there: /platform/common.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=projects/00-embedding-service/uv.lock,target=uv.lock \
    --mount=type=bind,source=projects/00-embedding-service/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=platform/common,target=/platform/common \
    uv sync --frozen --no-install-project --no-dev

COPY platform/common /platform/common
COPY projects/00-embedding-service /app
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 9100
CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "9100"]
