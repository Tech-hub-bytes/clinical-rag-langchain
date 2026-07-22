"""LangChain RAG over Databricks Vector Search (C-CDA, PDF, HL7, FHIR)."""

from __future__ import annotations

import os
from typing import Any

from databricks.sdk import WorkspaceClient
from databricks_langchain import ChatDatabricks
from databricks_langchain.vectorstores import DatabricksVectorSearch
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage

SYSTEM_PROMPT = """You are a clinical document assistant for synthetic demo health records.
You answer using retrieved context from HL7 C-CDA, patient PDFs, HL7 v2 messages, and FHIR JSON.
Answer ONLY from the provided context blocks numbered [1], [2], ….
When citing, use those bracket numbers and the patient/doc_type shown in that same block.
If the user asks for a patient name that does not appear in the context, say you do not know that patient.
If a close match exists (e.g. ALI → Alice), say so and cite the matching blocks only.
Prefer patient-identification / demographics / PID / FHIR Patient sections for name questions.
Do not invent clinical facts. Synthetic demo data only. Keep answers concise."""

# Canonical patient labels used in chunk metadata
KNOWN_PATIENTS = (
    "EmmaTestPatient",
    "James Testpatient",
    "Alice Jones",
    "Sample Patient",
    "ALI",
    "John Smith",
    "Emily Davis",
    "David Miller",
    "Sophia Anderson",
)


def detect_patient_filter(question: str, candidates: list[str] | None = None) -> str | None:
    """Map free-text questions to patient labels without relying only on hardcoding.

    1) Known aliases (emma/james/ali/…)
    2) Token overlap against candidate patient names (from index + known list)
    """
    q = question.lower()
    q_tokens = set(_tokens(question))
    if not q_tokens:
        return None

    # Exact alias tokens
    exact_map = {
        "emma": "EmmaTestPatient",
        "emmatestpatient": "EmmaTestPatient",
        "james": "James Testpatient",
        "jamestestpatient": "James Testpatient",
        "alice": "Alice Jones",
        "newman": "Alice Jones",
        "ali": "ALI",
        "sample": "Sample Patient",
        "john": "John Smith",
        "smith": "John Smith",
        "johnsmith": "John Smith",
        "emily": "Emily Davis",
        "davis": "Emily Davis",
        "emilydavis": "Emily Davis",
        "david": "David Miller",
        "miller": "David Miller",
        "davidmiller": "David Miller",
        "sophia": "Sophia Anderson",
        "anderson": "Sophia Anderson",
        "sophiaanderson": "Sophia Anderson",
    }
    # Prefer multi-token full-name hits first
    if "sophia" in q_tokens and "anderson" in q_tokens:
        return "Sophia Anderson"
    if "david" in q_tokens and "miller" in q_tokens:
        return "David Miller"
    if "emily" in q_tokens and "davis" in q_tokens:
        return "Emily Davis"
    if "john" in q_tokens and "smith" in q_tokens:
        return "John Smith"
    if "alice" in q_tokens or "newman" in q_tokens:
        return "Alice Jones"
    if "emma" in q_tokens:
        return "EmmaTestPatient"
    if "james" in q_tokens:
        return "James Testpatient"
    if "ali" in q_tokens and "alice" not in q_tokens:
        return "ALI"

    for tok, label in exact_map.items():
        if tok in q_tokens:
            return label

    # Dynamic overlap against known + discovered patients
    pool = list(dict.fromkeys((candidates or []) + list(KNOWN_PATIENTS)))
    best_label = None
    best_score = 0
    for label in pool:
        name_tokens = set(_tokens(label))
        if not name_tokens:
            continue
        overlap = q_tokens & name_tokens
        # Require at least one strong token (>=4 chars) or two tokens
        strong = {t for t in overlap if len(t) >= 4}
        score = len(overlap) * 2 + len(strong)
        if len(overlap) >= 2 or strong:
            if score > best_score:
                best_score = score
                best_label = label
    return best_label

DEFAULT_COLUMNS = [
    "chunk_id",
    "content",
    "source_path",
    "patient",
    "doc_type",
    "section_title",
]


def env(name: str, default: str = "") -> str:
    return (os.environ.get(name) or default).strip()


def workspace_client() -> WorkspaceClient:
    return WorkspaceClient()


def volume_root() -> str:
    return env("DATABRICKS_VOLUME_PATH", "/Volumes/workspace/ccda_rag/docs").rstrip("/")


def serving_endpoint() -> str:
    return env("DATABRICKS_SERVING_ENDPOINT_NAME", "databricks-meta-llama-3-3-70b-instruct")


def vs_index() -> str:
    return env("DATABRICKS_VECTOR_SEARCH_INDEX", "workspace.ccda_rag.document_chunks_index")


def ingest_job_id() -> str:
    return env("INGEST_JOB_ID", "1026899989768735")


def vs_pipeline_id() -> str:
    return env("VS_PIPELINE_ID", "bd6baf44-ea7a-4d58-89b2-06a974da54cd")


def list_indexed_patients(limit: int = 50) -> list[str]:
    """Best-effort patient list from a broad hybrid probe (for sidebar)."""
    try:
        pairs = retrieve("patient identification demographics name PID", k=min(limit, 20))
        names = []
        seen = set()
        for doc, _ in pairs:
            p = str((doc.metadata or {}).get("patient") or "").strip()
            if p and p not in seen:
                seen.add(p)
                names.append(p)
        return names
    except Exception:
        return list(KNOWN_PATIENTS)


def trigger_ingest(wait: bool = False) -> dict[str, Any]:
    """Start the ccda-rag-ingest job so new volume files become searchable."""
    w = workspace_client()
    job_id = int(ingest_job_id())
    run = w.jobs.run_now(job_id=job_id)
    run_id = getattr(run, "run_id", None) or (run.get("run_id") if isinstance(run, dict) else None)
    result: dict[str, Any] = {"job_id": job_id, "run_id": run_id, "waited": False, "state": "TRIGGERED"}
    if wait and run_id:
        import time

        for _ in range(90):
            time.sleep(10)
            info = w.jobs.get_run(run_id)
            life = getattr(getattr(info, "state", None), "life_cycle_state", None)
            life = str(life).split(".")[-1] if life is not None else None
            result_state = getattr(getattr(info, "state", None), "result_state", None)
            result_state = str(result_state).split(".")[-1] if result_state is not None else None
            result["state"] = f"{life}/{result_state}"
            if life in ("TERMINATED", "INTERNAL_ERROR", "SKIPPED"):
                result["waited"] = True
                result["result_state"] = result_state
                break
    return result


def trigger_index_sync(*, wait: bool = True, timeout_s: int = 600) -> dict[str, Any]:
    """Kick Vector Search delta-sync and optionally wait until the pipeline completes."""
    import time

    w = workspace_client()
    pipe = vs_pipeline_id()
    method = "pipelines.start_update"
    upd_id = None
    vs_error = None

    try:
        from databricks.vector_search.client import VectorSearchClient

        vsc = VectorSearchClient(workspace_client=w)
        try:
            vsc.sync_index(index_name=vs_index())
            method = "vector_search_client.sync_index"
        except TypeError:
            vsc.get_index(index_name=vs_index()).sync()
            method = "vector_search_client.get_index.sync"
    except Exception as vs_err:
        vs_error = str(vs_err)

    # Always also ensure a pipeline update is running (VS client sync is sometimes async-only)
    try:
        # If a prior update is stuck in CREATED, stop then restart
        try:
            w.pipelines.stop(pipeline_id=pipe)
            time.sleep(2)
        except Exception:
            pass
        upd = w.pipelines.start_update(pipeline_id=pipe)
        upd_id = getattr(upd, "update_id", None) or (
            upd.get("update_id") if isinstance(upd, dict) else None
        )
        method = "pipelines.start_update"
    except Exception as pipe_err:
        if vs_error:
            return {"ok": False, "error": str(pipe_err), "vs_error": vs_error}
        return {"ok": False, "error": str(pipe_err), "method": method}

    result: dict[str, Any] = {
        "ok": True,
        "method": method,
        "pipeline_id": pipe,
        "update_id": upd_id,
        "waited": False,
        "vs_error": vs_error,
    }
    if not wait:
        return result

    deadline = time.time() + timeout_s
    while time.time() < deadline:
        time.sleep(15)
        info = w.pipelines.get(pipeline_id=pipe)
        updates = getattr(info, "latest_updates", None) or []
        if not updates and isinstance(info, dict):
            updates = info.get("latest_updates") or []
        latest = updates[0] if updates else None
        state = None
        if latest is not None:
            state = getattr(latest, "state", None) or (
                latest.get("state") if isinstance(latest, dict) else None
            )
        state_s = str(state).split(".")[-1] if state is not None else None
        result["pipeline_state"] = state_s
        if state_s in ("COMPLETED", "FAILED", "CANCELED"):
            result["waited"] = True
            result["ok"] = state_s == "COMPLETED"
            return result
    result["waited"] = True
    result["ok"] = False
    result["error"] = "timeout waiting for vector search sync"
    return result


def reindex_after_upload(*, wait_ingest: bool = True, wait_sync: bool = True) -> dict[str, Any]:
    """Upload is not enough — run ingest job then sync the VS index (wait for both)."""
    ingest = trigger_ingest(wait=wait_ingest)
    if wait_ingest and ingest.get("result_state") not in (None, "SUCCESS"):
        return {"ingest": ingest, "sync": {"ok": False, "error": "ingest did not succeed"}}
    sync = trigger_index_sync(wait=wait_sync)
    return {"ingest": ingest, "sync": sync}

def _tokens(text: str) -> list[str]:
    import re

    return [t for t in re.findall(r"[a-z0-9]+", (text or "").lower()) if len(t) >= 2]


def is_patient_name_question(question: str) -> bool:
    q = question.lower()
    return any(
        phrase in q
        for phrase in (
            "patient name",
            "patient's name",
            "who is the patient",
            "name of the patient",
            "what is the name",
            "demographics",
        )
    ) or (q.strip().startswith("patient name") or " named " in q)


def retrieval_query(question: str, patient: str | None) -> str:
    """Bias hybrid search toward identity chunks for name questions."""
    if is_patient_name_question(question) or patient:
        who = patient or "patient"
        return (
            f"{question}. Patient identification demographics legal name "
            f"PID FHIR Patient {who} given family"
        )
    return question


def rerank_pairs(
    pairs: list[tuple[Document, float | None]],
    *,
    patient: str | None,
    question: str,
) -> list[tuple[Document, float | None]]:
    """Prefer matching patient + identity sections when the question is about names."""
    name_q = is_patient_name_question(question)

    def sort_key(item: tuple[Document, float | None]):
        doc, score = item
        meta = doc.metadata or {}
        p = str(meta.get("patient") or "")
        section = str(meta.get("section_title") or "").lower()
        content = (doc.page_content or "").lower()
        patient_match = 0 if (patient and p == patient) else (1 if patient else 0)
        identity = 0
        if name_q:
            identity_hit = any(
                k in section or k in content
                for k in (
                    "patient identification",
                    "demographics",
                    "pid",
                    "fhir patient",
                    "patient name",
                    "legal name",
                )
            )
            identity = 0 if identity_hit else 1
        # Higher VS score is better; treat None as 0
        score_rank = -(float(score) if isinstance(score, (int, float)) else 0.0)
        return (patient_match, identity, score_rank)

    return sorted(pairs, key=sort_key)


def build_filters(
    patient: str | None,
    doc_types: list[str] | None,
) -> dict[str, Any] | None:
    filters: dict[str, Any] = {}
    if patient and patient != "All patients":
        filters["patient"] = patient
    if doc_types:
        if len(doc_types) == 1:
            filters["doc_type"] = doc_types[0]
        else:
            filters["doc_type"] = doc_types
    return filters or None


def get_vector_store() -> DatabricksVectorSearch:
    return DatabricksVectorSearch(
        index_name=vs_index(),
        columns=DEFAULT_COLUMNS,
        workspace_client=workspace_client(),
    )


def retrieve(
    question: str,
    *,
    patient: str | None = None,
    doc_types: list[str] | None = None,
    k: int = 8,
) -> list[tuple[Document, float | None]]:
    store = get_vector_store()
    filters = build_filters(patient, doc_types)

    # Try hybrid + score first; fall back across SDK signature variants.
    attempts: list[dict[str, Any]] = [
        {"k": k, "query_type": "HYBRID", "filters": filters},
        {"k": k, "query_type": "HYBRID", "filter": filters},
        {"k": k, "filters": filters},
        {"k": k, "filter": filters},
        {"k": k},
    ]
    last_err: Exception | None = None
    for kwargs in attempts:
        clean = {key: val for key, val in kwargs.items() if val is not None}
        try:
            scored = store.similarity_search_with_score(question, **clean)
            return [(doc, float(score) if score is not None else None) for doc, score in scored]
        except TypeError as err:
            last_err = err
            continue
        except Exception as err:
            last_err = err
            # Invalid filter key variants — try next signature
            if "filter" in clean or "filters" in clean or "query_type" in clean:
                continue
            raise
    # Last resort: plain search
    try:
        docs = store.similarity_search(question, k=k)
        return [(d, None) for d in docs]
    except Exception as err:
        raise RuntimeError(f"Vector Search retrieve failed: {err}") from (last_err or err)


def format_context(pairs: list[tuple[Document, float | None]]) -> str:
    blocks = []
    for i, (doc, score) in enumerate(pairs, start=1):
        meta = doc.metadata or {}
        score_s = f"{score:.4f}" if isinstance(score, (int, float)) else "n/a"
        blocks.append(
            f"[{i}] source={meta.get('source_path', '')} | doc_type={meta.get('doc_type', '')} "
            f"| patient={meta.get('patient', '')} | section={meta.get('section_title', '')} "
            f"| score={score_s}\n{doc.page_content}"
        )
    return "\n\n---\n\n".join(blocks) if blocks else "(no retrieved context)"


def answer_question(
    question: str,
    *,
    patient: str | None = None,
    doc_types: list[str] | None = None,
    k: int = 8,
) -> dict[str, Any]:
    # Probe without patient filter to discover names currently in the index
    probe = retrieve(question, patient=None, doc_types=doc_types, k=max(k, 10))
    candidates = []
    seen = set()
    for doc, _ in probe:
        p = str((doc.metadata or {}).get("patient") or "").strip()
        if p and p not in seen:
            seen.add(p)
            candidates.append(p)

    effective_patient = patient
    if not effective_patient or effective_patient == "All patients":
        detected = detect_patient_filter(question, candidates=candidates)
        if detected:
            effective_patient = detected

    query = retrieval_query(question, effective_patient)
    fetch_k = max(k * 2, 12) if (effective_patient or is_patient_name_question(question)) else k
    pairs = retrieve(query, patient=effective_patient, doc_types=doc_types, k=fetch_k)
    # If filtered retrieval is empty (index lag), fall back to unfiltered probe
    if not pairs and probe:
        pairs = probe
    pairs = rerank_pairs(pairs, patient=effective_patient, question=question)[:k]
    context = format_context(pairs)
    llm = ChatDatabricks(
        endpoint=serving_endpoint(),
        temperature=0.1,
        max_tokens=1024,
        workspace_client=workspace_client(),
    )
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=(
                f"Context:\n{context}\n\n"
                f"Question: {question}\n\n"
                "Answer using only the context. Cite bracket numbers that appear above."
            )
        ),
    ]
    response = llm.invoke(messages)
    text = response.content if hasattr(response, "content") else str(response)
    if isinstance(text, list):
        text = "\n".join(
            part.get("text", str(part)) if isinstance(part, dict) else str(part) for part in text
        )

    sources = []
    for doc, score in pairs:
        meta = doc.metadata or {}
        sources.append(
            {
                "source_path": meta.get("source_path", ""),
                "doc_type": meta.get("doc_type", ""),
                "patient": meta.get("patient", ""),
                "section_title": meta.get("section_title", ""),
                "score": score,
                "preview": (doc.page_content or "")[:400],
            }
        )

    return {
        "answer": text,
        "sources": sources,
        "patient_filter": effective_patient,
        "index": vs_index(),
    }


def folder_for_filename(name: str) -> str:
    lower = name.lower()
    if lower.endswith(".pdf"):
        return "pdfs"
    if lower.endswith(".json"):
        return "fhir"
    if lower.endswith((".hl7",)):
        return "hl7"
    if lower.endswith((".xml", ".md")):
        return "ccda"
    if lower.endswith(".txt"):
        return "hl7"
    return "ccda"


def upload_to_volume(filename: str, data: bytes) -> str:
    folder = folder_for_filename(filename)
    dest = f"{volume_root()}/{folder}/{filename}"
    w = workspace_client()
    from io import BytesIO

    # Ensure parent directory exists (UC Files API)
    try:
        w.files.create_directory(f"{volume_root()}/{folder}")
    except Exception:
        pass
    w.files.upload(dest, BytesIO(data), overwrite=True)
    return dest


def ensure_volume_folders() -> list[str]:
    w = workspace_client()
    created = []
    for folder in ("ccda", "pdfs", "hl7", "fhir"):
        path = f"{volume_root()}/{folder}/.keep"
        try:
            from io import BytesIO

            w.files.upload(path, BytesIO(b""), overwrite=True)
            created.append(folder)
        except Exception:
            pass
    return created
