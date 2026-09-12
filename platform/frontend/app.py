"""RAGLab unified launcher: pick a backend, launch it, try it.

Thin by design — all HTTP/call-sequencing logic lives in
`raglab_frontend` where it's unit tested; this file is UI wiring only.
"""

from __future__ import annotations

import streamlit as st
from raglab_common import configure_logging

from config import settings
from raglab_frontend.launch_actions import launch, reset, stop
from raglab_frontend.orchestrator_client import OrchestratorClient
from raglab_frontend.ordering import sort_basic_to_advanced

log = configure_logging(project_id="frontend")

_STATE_ICONS = {"healthy": "🟢", "starting": "🟡", "stopped": "⚪"}


@st.cache_resource
def get_client() -> OrchestratorClient:
    """Build the (cached, one-per-session) orchestrator client."""
    return OrchestratorClient(base_url=settings.orchestrator_base_url)


def page_setup() -> None:
    """Configure the page and hide Streamlit's default chrome."""
    st.set_page_config(page_title="RAGLab Launcher", page_icon="🧪", layout="wide")
    st.markdown(
        "<style>#MainMenu {visibility: hidden;} footer {visibility: hidden;}</style>",
        unsafe_allow_html=True,
    )


def _state_badge(state: str) -> str:
    return f"{_STATE_ICONS.get(state, '⚪')} {state}"


def render_sidebar(client: OrchestratorClient) -> tuple[str | None, str | None]:
    """Render the backend/vector-store pickers and lifecycle controls.

    Args:
        client: The orchestrator client to query and act through.

    Returns:
        The currently selected (backend_id, vector_store_id); either may
        be None if nothing is selectable yet.
    """
    st.sidebar.title("🧪 RAGLab")
    st.sidebar.caption("Pick a backend, launch it, try it.")

    try:
        backends = sort_basic_to_advanced(client.list_backends())
    except Exception as exc:  # noqa: BLE001 - surfaced to the user, not swallowed
        st.sidebar.error(f"Can't reach the orchestrator: {exc}")
        log.warning("failed to list backends: {}", exc)
        return None, None

    if not backends:
        st.sidebar.warning("No backends in the catalog yet.")
        return None, None

    options = {f"{b.spec.name} ({b.spec.level})": b for b in backends}
    chosen = st.sidebar.selectbox("Backend", list(options.keys()))
    backend = options[chosen].spec

    vector_store_id: str | None = None
    if backend.compatible_vector_stores:
        vector_store_id = st.sidebar.selectbox(
            "Vector store", backend.compatible_vector_stores
        )
    else:
        st.sidebar.caption("This backend doesn't use the shared vector-store slot.")

    launch_col, stop_col, reset_col = st.sidebar.columns(3)
    if launch_col.button("▶ Launch", use_container_width=True):
        with st.spinner("Starting..."):
            launch(client, backend.id, vector_store_id)
        st.rerun()
    if stop_col.button("■ Stop", use_container_width=True):
        stop(client, backend.id, vector_store_id)
        st.rerun()
    if reset_col.button("⟲ Reset", use_container_width=True):
        reset(client, backend.id, vector_store_id)
        st.rerun()

    st.sidebar.divider()
    try:
        st.sidebar.write(f"Backend: {_state_badge(client.get_backend_status(backend.id).state)}")
    except Exception as exc:  # noqa: BLE001
        st.sidebar.write("Backend: unknown")
        log.debug("status check failed for {}: {}", backend.id, exc)
    if vector_store_id:
        try:
            vs_state = client.get_vector_store_status(vector_store_id).state
            st.sidebar.write(f"Vector store: {_state_badge(vs_state)}")
        except Exception as exc:  # noqa: BLE001
            st.sidebar.write("Vector store: unknown")
            log.debug("status check failed for {}: {}", vector_store_id, exc)

    return backend.id, vector_store_id


def render_main(
    client: OrchestratorClient, backend_id: str | None, vector_store_id: str | None
) -> None:
    """Render the Try It / Benchmarks tabs for the selected backend.

    Args:
        client: The orchestrator client to query.
        backend_id: The selected backend's catalog id, or None.
        vector_store_id: The selected vector store's catalog id, or None.
    """
    try_it_tab, benchmarks_tab = st.tabs(["Try it", "Benchmarks"])

    with try_it_tab:
        if backend_id is None:
            st.info("Pick a backend from the sidebar to get started.")
        elif client.get_backend_status(backend_id).state != "healthy":
            st.info("Launch the backend from the sidebar, then come back here once it's ready.")
        else:
            st.caption(
                "Chat and file upload wired to the live backend land with each "
                "project's own weekend milestone."
            )
            prompt = st.chat_input("Ask something...")
            if prompt:
                st.chat_message("user").write(prompt)
                st.chat_message("assistant").write(
                    "(this project hasn't wired up chat yet — check back after its weekend lands)"
                )

    with benchmarks_tab:
        st.caption(
            "Latency/cost/throughput comparisons land once a few projects are "
            "shipped — see the root README roadmap."
        )
        st.info("No benchmark data yet.")


def main() -> None:
    """Entry point Streamlit runs."""
    page_setup()
    client = get_client()
    backend_id, vector_store_id = render_sidebar(client)
    render_main(client, backend_id, vector_store_id)


if __name__ == "__main__":
    main()
