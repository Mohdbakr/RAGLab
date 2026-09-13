# Build context is the repo root (see docker-compose.yml in this folder)
# so this image can also COPY the platform/common sibling package this
# service depends on as an editable path dependency.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app/service
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# The pyproject.toml's `raglab-common = {path = "../../platform/common"}`
# is resolved relative to WORKDIR, and it climbs two directories because
# that's the real repo layout (projects/00-embedding-service -> repo
# root -> platform/common). WORKDIR must therefore be two levels deep
# too (/app/service, not /app), so those two ".." land back at "/" and
# then find /platform/common.
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=projects/00-embedding-service/uv.lock,target=uv.lock \
    --mount=type=bind,source=projects/00-embedding-service/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=platform/common,target=/platform/common \
    uv sync --frozen --no-install-project --no-dev

COPY platform/common /platform/common
COPY projects/00-embedding-service /app/service
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

ENV PATH="/app/service/.venv/bin:$PATH"
EXPOSE 9100
CMD ["fastapi", "run", "app/main.py", "--host", "0.0.0.0", "--port", "9100"]
