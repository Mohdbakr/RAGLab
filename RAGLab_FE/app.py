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


def check_server_status(url):
    """Check if the backend server is running and returns the status."""
    try:
        response = requests.get(
            url, timeout=5
        )  # Set a timeout to avoid hanging
        if response.status_code == 200:
            return True, response.json().get("message", "Server is Up")
    except requests.exceptions.ConnectionError:
        st.error(
            f"Error: Unable to connect to the backend at {url} (Connection Refused)"
        )
    except requests.exceptions.Timeout:
        st.error(f"Error: Request to {url} timed out")
    except requests.exceptions.RequestException as e:
        st.error(f"Unexpected error: {e}")
    return False, "Server is Down"


def show_status_indicator(status, message):
    """Display a red or green dot with status text on the Streamlit app."""
    if status:
        st.markdown(
            f"""
            <div style='display: flex; align-items: center;'>
                <div style='width: 10px; height: 10px; background-color: green; border-radius: 50%; margin-right: 10px;'></div>
                <p style='margin: 0; font-size: 12px;'>{message}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            f"""
            <div style='display: flex; align-items: center;'>
                <div style='width: 10px; height: 10px; background-color: red; border-radius: 50%; margin-right: 10px;'></div>
                <p style='margin: 0; font-size: 12px;'>{message}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


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
        # Status section with periodic updates
        status_placeholder = st.empty()  # Placeholder for dynamic updates

        # Initialize session state for periodic refresh
        if "last_checked" not in st.session_state:
            st.session_state.last_checked = 0

        current_time = time.time()
        if (
            current_time - st.session_state.last_checked > 30
        ):  # 30-second interval
            # Update status every 30 seconds
            status, message = check_server_status(API_BASE_URL)
            with status_placeholder:
                show_status_indicator(status, message)
            st.session_state.last_checked = current_time

        st.divider()
        if st.button("☁️ Upload Documents", type="tertiary"):
            file_upload_dialog()
        if st.button("📖 instructions", type="tertiary"):
            set_model_configs()
        if st.button("⚙️ Settings", type="tertiary"):
            set_model_configs()

        st.divider()


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
