# Build context is the repo root (see ../../docker-compose.yml) so this
# image can also COPY the platform/common sibling package that the
# frontend depends on as an editable path dependency.

FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
WORKDIR /app
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=platform/frontend/uv.lock,target=uv.lock \
    --mount=type=bind,source=platform/frontend/pyproject.toml,target=pyproject.toml \
    --mount=type=bind,source=platform/common,target=/common \
    uv sync --frozen --no-install-project --no-dev

COPY platform/common /common
COPY platform/frontend /app
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev

ENV PATH="/app/.venv/bin:$PATH"
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
