"""Runtime configuration for clinical-rag-chat (env-driven, no secrets)."""

from __future__ import annotations

import os

# Vector Search
VS_ENDPOINT = os.environ.get(
    "DATABRICKS_VECTOR_SEARCH_ENDPOINT", "ka-3abc305d-vs-endpoint"
)
VS_INDEX = os.environ.get(
    "DATABRICKS_VECTOR_SEARCH_INDEX", "workspace.ccda_rag.document_chunks_index"
)

# Serving
SERVING_ENDPOINT = os.environ.get(
    "DATABRICKS_SERVING_ENDPOINT_NAME", "databricks-meta-llama-3-3-70b-instruct"
)

# Unity Catalog volume (docs root)
VOLUME_ROOT = os.environ.get(
    "DATABRICKS_VOLUME_PATH",
    os.environ.get("DATABRICKS_VOLUME_FILES", "/Volumes/workspace/ccda_rag/docs"),
).rstrip("/")
# Prefer explicit path; valueFrom "files" may inject a volume name rather than /Volumes/... path.
if VOLUME_ROOT and not VOLUME_ROOT.startswith("/Volumes/"):
    VOLUME_ROOT = "/Volumes/workspace/ccda_rag/docs"

# Ingest job (sidebar re-ingest hint)
INGEST_JOB_ID = os.environ.get("CCDA_RAG_INGEST_JOB_ID", "1026899989768735")
INGEST_JOB_NAME = os.environ.get("CCDA_RAG_INGEST_JOB_NAME", "ccda-rag-ingest")
DATABRICKS_PROFILE = os.environ.get("DATABRICKS_CONFIG_PROFILE", "dbc-7c3eed4c")

DOC_TYPES = ("ccda_xml", "pdf", "hl7_v2", "fhir_json", "markdown")

# Upload subfolders under VOLUME_ROOT → allowed extensions
UPLOAD_FOLDERS: dict[str, tuple[str, ...]] = {
    "ccda": (".xml", ".md"),
    "pdfs": (".pdf",),
    "hl7": (".hl7", ".txt"),
    "fhir": (".json",),
}

DISCLAIMER = (
    "AI-generated — verify against source documents before clinical use. "
    "Synthetic demo data only; not for real PHI or care decisions."
)
