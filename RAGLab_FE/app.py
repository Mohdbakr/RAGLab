from typing import Optional, Any
import requests
import time

import streamlit as st


API_BASE_URL: str = "http://localhost:8000/"


def chat_with_backend(query: str) -> Optional[str]:
    """
    Sends a chat query to the backend and returns the response.

    Args:
        query (str): The user's chat query.

    Returns:
        Optional[str]: The response from the backend, or an error message.

    Example:
        >>> chat_with_backend("What is the capital of France?")
        'The capital of France is Paris.'
    """
    pass


def upload_files(file: Any) -> str:
    """
    Uploads a file to the backend for processing.

    Args:
        file (Any): The file object to upload.

    Returns:
        str: A message indicating the upload status or an error message.
    """
    pass


def check_health_status() -> Optional[str]:
    """
    Checks the health status of the backend service.

    Returns:
        Optional[str]: The health status message, or an error message.
    """
    pass


def check_server_status(url: str) -> tuple[bool, str, float]:
    """Check server status with retry logic and latency measurement."""
    if not st.session_state["auto_refresh"]:
        return False, "Status Check Disabled", 0.0

    try:
        start_time = time.time()
        response = requests.get(url, timeout=5)
        latency = (time.time() - start_time) * 1000  # Convert to milliseconds

        if response.status_code == 200:
            return True, "Server Connected", latency

    except requests.exceptions.ConnectionError:
        return False, "Connection Failed", 0.0
    except requests.exceptions.Timeout:
        return False, "Request Timeout", 0.0
    except requests.exceptions.RequestException:
        return False, "Connection Error", 0.0

    return False, "Server Disconnected", 0.0


def show_status_indicator(
    status: bool, message: str, latency: float = None
) -> None:
    """Display an enhanced status indicator with latency and controls."""
    indicator = "🟢" if status else "🔴"
    latency_text = f" ({latency:.2f}ms)" if latency is not None else ""

    st.markdown(
        f"""
        <div style='display: flex; align-items: center;
                    justify-content: space-between; padding: 8px;
                    border-radius: 4px; margin-bottom: 8px'>
            <div style='display: flex; align-items: center; gap: 8px;'>
                <span style='font-size: 8px'>{indicator}</span>
                <span style='font-size: 12px; color: #6b7280'>
                    {message}{latency_text}
                </span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


@st.fragment(run_every="30s")
def update_status():
    """Update and display server status."""
    status, message, latency = check_server_status(API_BASE_URL)
    st.session_state.status = status
    st.session_state.status_message = message
    st.session_state.latency = latency
    show_status_indicator(status, message, latency)


def page_setup():
    st.markdown(
        """
        <style>
            .reportview-container {
                margin-top: -2em;
            }
            #MainMenu {visibility: hidden;}
            .stAppDeployButton {display:none;}
            footer {visibility: hidden;}
            #stDecoration {display:none;}
        </style>
    """,
        unsafe_allow_html=True,
    )


@st.dialog("Model Settings")
def set_model_configs():
    st.write("Set up different model configurations")

    col1, col2 = st.columns([3, 1])
    with col1:
        on = st.toggle(
            "Backend Health Check",
            value=st.session_state.get("auto_refresh", True),
            help="Checks if the Backend server is connected when ON",
            key="health_check_toggle",
        )
        st.session_state["auto_refresh"] = on

    with col2:
        st.write("Status:", "🟢 ON" if on else "🔴 OFF")


@st.dialog("How it works!")
def instructions():
    st.write("Follow these instructions")


@st.dialog("Upload Your Documents")
def file_upload_dialog():
    with st.form("my-form", clear_on_submit=True):
        files = st.file_uploader(
            "Upload files",
            type=["pdf", "txt", "docx"],
            accept_multiple_files=True,
        )
        submitted = st.form_submit_button("UPLOAD!")

        if submitted:
            if files is not None:
                with st.spinner("Uploading files..."):
                    st.write("UPLOADED!")
                    upload_files(files)
            else:
                st.warning("Please select at least one file to upload.")


def sidebar():
    # Sidebar for uploading Docs

    with st.sidebar:
        st.title("🧪 RAG Lab")

        # Status indicator at the top of sidebar
        if "auto_refresh" not in st.session_state:
            st.session_state["auto_refresh"] = True

        # update_status()

        st.divider()
        if st.button("☁️ Upload Documents", type="tertiary"):
            file_upload_dialog()
        if st.button("📖 instructions", type="tertiary"):
            set_model_configs()
        if st.button("⚙️ Settings", type="tertiary"):
            set_model_configs()

        # Push status to bottom using empty space
        st.markdown(
            '<div style="flex-grow: 1; min-height: 45vh;"></div>',
            unsafe_allow_html=True,
        )

        # Create a container for the bottom status
        status_container = st.container()
        # Status indicator at the bottom
        with status_container:
            st.divider()
            update_status()


def main():
    page_setup()
    sidebar()
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("What is up?"):
        # Display user message in chat message container
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        response = f"Echo: {prompt}"
        # Display assistant response in chat message container
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append(
            {"role": "assistant", "content": response}
        )

    async def ws_connection() -> None:
        """
        Establishes a WebSocket connection
        and receives messages from the backend.

        This function runs as an asynchronous task
        and displays real-time messages from the backend.
        """
        pass


if __name__ == "__main__":
    main()
