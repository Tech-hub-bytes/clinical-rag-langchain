"""Multi-format Clinical RAG chatbot — Streamlit + LangChain + Databricks Vector Search."""

from __future__ import annotations

import streamlit as st

from rag import (
    answer_question,
    ensure_volume_folders,
    ingest_job_id,
    list_indexed_patients,
    reindex_after_upload,
    serving_endpoint,
    upload_to_volume,
    volume_root,
    vs_index,
)

st.set_page_config(
    page_title="Clinical RAG Chat",
    layout="wide",
)

DOC_TYPES = ["ccda_xml", "pdf", "hl7_v2", "fhir_json", "markdown"]

st.title("Clinical RAG Chat")
st.caption(
    "LangChain RAG over C-CDA · PDF · HL7 v2 · FHIR JSON via Databricks Vector Search"
)
st.info("AI-generated — verify against source documents. Synthetic demo data only.")
st.warning(
    "Upload alone is **not** enough. After new files click **Re-index now** and wait until it "
    "shows sync COMPLETED (often 3–8 minutes). Asking before Vector Search finishes syncing "
    "will say “I do not know that patient.”"
)

with st.sidebar:
    st.header("Retrieval")
    indexed = ["All patients"] + sorted(
        set(
            list_indexed_patients()
            + [
                "ALI",
                "EmmaTestPatient",
                "James Testpatient",
                "Alice Jones",
                "Sample Patient",
                "Sophia Anderson",
            ]
        )
    )
    # Keep All patients first
    indexed = ["All patients"] + [p for p in indexed if p != "All patients"]
    patient = st.selectbox("Patient filter", indexed, index=0)
    doc_types = st.multiselect("Document types", DOC_TYPES, default=DOC_TYPES)
    top_k = st.slider("Top-k chunks", min_value=3, max_value=16, value=8)

    st.divider()
    st.header("Upload")
    st.caption(f"Volume: `{volume_root()}`")
    uploads = st.file_uploader(
        "C-CDA (.xml/.md), PDF, HL7 (.hl7/.txt), FHIR (.json)",
        type=["xml", "md", "pdf", "hl7", "txt", "json"],
        accept_multiple_files=True,
    )
    auto_reindex = st.checkbox("After upload, run ingest + index sync", value=True)
    if uploads and st.button("Upload to volume", type="primary"):
        ensure_volume_folders()
        results = []
        for f in uploads:
            path = upload_to_volume(f.name, f.getvalue())
            results.append(path)
        st.success(f"Uploaded {len(results)} file(s)")
        for p in results:
            st.code(p, language=None)
        if auto_reindex:
            with st.spinner("Running ingest job + Vector Search sync (can take a few minutes)…"):
                try:
                    status = reindex_after_upload(wait_ingest=True)
                    st.json(status)
                    if status.get("ingest", {}).get("result_state") == "SUCCESS":
                        st.success("Re-index finished. Ask about the new patient now.")
                    else:
                        st.warning("Ingest may still be running — wait a minute, then try again.")
                except Exception as exc:
                    st.error(f"Re-index failed: {exc}")
                    st.info(
                        f"Run manually: `databricks jobs run-now {ingest_job_id()} --profile dbc-7c3eed4c`"
                    )

    st.divider()
    st.header("Re-index")
    st.caption(
        f"Job `{ingest_job_id()}` builds chunks; then index `{vs_index()}` syncs for chat."
    )
    if st.button("Re-index now", type="secondary"):
        with st.spinner("Running ingest + Vector Search sync…"):
            try:
                status = reindex_after_upload(wait_ingest=True)
                st.json(status)
                st.success("Done — new patients should be searchable shortly.")
            except Exception as exc:
                st.error(f"Re-index failed: {exc}")

    st.caption(f"LLM: `{serving_endpoint()}`")
    st.caption(f"Index: `{vs_index()}`")

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("Sources"):
                for s in msg["sources"]:
                    st.markdown(
                        f"**{s.get('section_title') or 'chunk'}** · "
                        f"`{s.get('doc_type')}` · patient={s.get('patient')} · "
                        f"score={s.get('score')}"
                    )
                    st.caption(s.get("source_path", ""))
                    st.text(s.get("preview", "")[:300])

prompt = st.chat_input("Ask about medications, allergies, labs, FHIR resources, HL7 segments…")
if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Retrieving + generating…"):
            try:
                result = answer_question(
                    prompt,
                    patient=None if patient == "All patients" else patient,
                    doc_types=doc_types or None,
                    k=top_k,
                )
                answer = result["answer"]
                sources = result["sources"]
                st.markdown(answer)
                if sources:
                    with st.expander("Sources", expanded=True):
                        for s in sources:
                            st.markdown(
                                f"**{s.get('section_title') or 'chunk'}** · "
                                f"`{s.get('doc_type')}` · patient={s.get('patient')} · "
                                f"score={s.get('score')}"
                            )
                            st.caption(s.get("source_path", ""))
                            st.text(s.get("preview", "")[:300])
                elif "do not know" in answer.lower():
                    st.info(
                        "No matching indexed chunks. If you just uploaded a file, click "
                        "**Re-index now** in the sidebar, wait for SUCCESS, then ask again."
                    )
                st.caption("AI-generated — verify. Synthetic demo data only.")
                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as exc:
                err = f"Error: {exc}"
                st.error(err)
                st.session_state.messages.append({"role": "assistant", "content": err, "sources": []})
