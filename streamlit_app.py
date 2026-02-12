from pathlib import Path
import time

import streamlit as st
import inngest
from dotenv import load_dotenv
import os
import requests
from vector_db import QdrantStorage

load_dotenv()

# -------------------------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------------------------
st.set_page_config(
    page_title="RAG Support Assistant",
    page_icon="📄",
    layout="centered"
)

# -------------------------------------------------------------------
# Inngest Client
# -------------------------------------------------------------------
@st.cache_resource
def get_inngest_client() -> inngest.Inngest:
    return inngest.Inngest(app_id="rag_app", is_production=False)

# -------------------------------------------------------------------
# Document Upload Helpers
# -------------------------------------------------------------------
def save_uploaded_file(file) -> Path:
    uploads_dir = Path("uploads")
    uploads_dir.mkdir(parents=True, exist_ok=True)
    file_path = uploads_dir / file.name
    file_bytes = file.getbuffer()
    file_path.write_bytes(file_bytes)
    return file_path


def send_rag_ingest_event(document_path: Path) -> None:
    client = get_inngest_client()
    client.send_sync(
        inngest.Event(
            name="rag/ingest_document",
            data={
                "document_path": str(document_path.resolve()),
                "source_id": document_path.name,
            },
        )
    )

# -------------------------------------------------------------------
# UI — Document Ingestion
# -------------------------------------------------------------------
st.title("Upload a document to ingest")

uploaded = st.file_uploader(
    "Choose a PDF or TXT file",
    type=["pdf", "txt"],
    accept_multiple_files=False
)

if uploaded is not None:
    with st.spinner("Uploading and triggering ingestion..."):
        path = save_uploaded_file(uploaded)
        send_rag_ingest_event(path)
        time.sleep(0.3)

    st.success(f"Ingestion triggered for: {path.name}")
    st.caption("You can upload another document if you like.")

# -------------------------------------------------------------------
# Vector Database Management
# -------------------------------------------------------------------
st.divider()
st.subheader("Vector Database Management")

try:
    db = QdrantStorage()
    vector_count = db.get_collection_count()
    st.metric("Total Vectors in Database", vector_count)
except Exception as e:
    st.warning(f"Could not connect to vector database: {e}")
    vector_count = 0

col1, col2 = st.columns([1, 3])
with col1:
    if st.button("Clear Database", type="primary", disabled=vector_count == 0):
        with st.spinner("Clearing vector database..."):
            try:
                db = QdrantStorage()
                success = db.reset_collection()
                if success:
                    st.success("Vector database cleared successfully!")
                    st.rerun()
                else:
                    st.error("Failed to clear database.")
            except Exception as e:
                st.error(f"Error clearing database: {e}")

with col2:
    st.caption("This will permanently delete all vectors and documents from the database.")

# -------------------------------------------------------------------
# Divider
# -------------------------------------------------------------------
st.divider()

# -------------------------------------------------------------------
# UI — Support / Reflection Assistant
# -------------------------------------------------------------------
st.title("Talk through something on your mind")

st.caption(
    "This assistant doesn’t give direct answers. "
    "It helps you reflect by asking thoughtful follow-up questions."
)

# -------------------------------------------------------------------
# RAG Query Helpers
# -------------------------------------------------------------------
def send_rag_query_event(input_text: str, top_k: int) -> str:
    client = get_inngest_client()
    result = client.send_sync(
        inngest.Event(
            name="rag/query_pdf_ai",
            data={
                "question": input_text,
                "top_k": top_k,
            },
        )
    )
    return result[0]


def _inngest_api_base() -> str:
    return os.getenv("INNGEST_API_BASE", "http://127.0.0.1:8288/v1")


def fetch_runs(event_id: str) -> list[dict]:
    url = f"{_inngest_api_base()}/events/{event_id}/runs"
    resp = requests.get(url)
    resp.raise_for_status()
    data = resp.json()
    return data.get("data", [])


def wait_for_run_output(
    event_id: str,
    timeout_s: float = 120.0,
    poll_interval_s: float = 0.5
) -> dict:
    start = time.time()
    last_status = None

    while True:
        runs = fetch_runs(event_id)
        if runs:
            run = runs[0]
            status = run.get("status")
            last_status = status or last_status

            if status in ("Completed", "Succeeded", "Success", "Finished"):
                return run.get("output") or {}

            if status in ("Failed", "Cancelled"):
                raise RuntimeError(f"Function run {status}")

        if time.time() - start > timeout_s:
            raise TimeoutError(
                f"Timed out waiting for run output (last status: {last_status})"
            )

        time.sleep(poll_interval_s)

# -------------------------------------------------------------------
# UI — Reflection Input
# -------------------------------------------------------------------
with st.form("rag_query_form"):
    user_input = st.text_input(
        "What’s going on? You can share a concern, situation, or uncertainty."
    )

    top_k = st.number_input(
        "How many document chunks should I consider?",
        min_value=1,
        max_value=20,
        value=5,
        step=1
    )

    submitted = st.form_submit_button("Continue")

    if submitted and user_input.strip():
        with st.spinner("Thinking this through with you..."):
            event_id = send_rag_query_event(user_input.strip(), int(top_k))
            output = wait_for_run_output(event_id)

            response = output.get("answer", "")
            sources = output.get("sources", [])

        st.subheader("Response")
        st.write(response or "(No response)")

        if sources:
            st.caption("Sources considered")
            for s in sources:
                st.write(f"- {s}")
