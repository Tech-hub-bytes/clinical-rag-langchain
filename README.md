# Multi-format Clinical RAG (LangChain + Streamlit)

Databricks App that answers questions over **C-CDA**, **PDF**, **HL7 v2**, and **FHIR JSON** using LangChain + Vector Search.

## Example UI

Example question: *is there any Sophia Anderson patient?*

![Example multi-format clinical RAG UI](docs/screenshots/example-chat.png)

## Architecture of clinical-rag-chat

```mermaid
flowchart TB
  subgraph Users
    U[User / Clinician]
  end

  subgraph App["Databricks App: clinical-rag-chat (Streamlit)"]
    UI[Chat UI + Upload + Re-index]
    RAG[LangChain RAG layer]
    UI --> RAG
  end

  subgraph Storage["Unity Catalog Volume"]
    VOL["/Volumes/workspace/ccda_rag/docs\nccda/ · pdfs/ · hl7/ · fhir/"]
  end

  subgraph Ingest["Ingest Job: ccda-rag-ingest"]
    JOB[Parse C-CDA / PDF / HL7 / FHIR\n→ chunk + patient metadata]
    DELTA[(Delta table\nworkspace.ccda_rag.document_chunks)]
    JOB --> DELTA
    JOB --> SYNC[Auto-trigger VS sync]
  end

  subgraph Search["Vector Search"]
    IDX["Index: document_chunks_index\n(hybrid, TRIGGERED)"]
    EP[Endpoint: ka-3abc305d-vs-endpoint]
    IDX --- EP
  end

  subgraph LLM["Model Serving"]
    LLAMA[databricks-meta-llama-3-3-70b-instruct\nChatDatabricks]
  end

  U -->|ask / upload| UI
  UI -->|write files| VOL
  UI -->|Re-index / post-upload| JOB
  VOL -->|read files| JOB
  DELTA -->|CDF sync| IDX
  SYNC --> IDX
  RAG -->|hybrid retrieve| IDX
  RAG -->|prompt + context| LLAMA
  LLAMA -->|answer + citations| UI
  UI -->|response| U
```

**Data flow**

1. Upload docs to `/Volumes/workspace/ccda_rag/docs` (`ccda/`, `pdfs/`, `hl7/`, `fhir/`)
2. Ingest job `ccda-rag-ingest` → `workspace.ccda_rag.document_chunks` (also auto-triggers Vector Search sync)
3. Vector Search index `workspace.ccda_rag.document_chunks_index` (hybrid, TRIGGERED)
4. Streamlit app retrieves with LangChain `DatabricksVectorSearch` and answers via `ChatDatabricks` (Llama)

## App

| Item | Value |
|------|--------|
| Name | `clinical-rag-chat` |
| Path | `app/clinical-rag-chat` |
| LLM | `databricks-meta-llama-3-3-70b-instruct` |
| VS endpoint | `ka-3abc305d-vs-endpoint` |
| Ingest job | `1026899989768735` |

## Deploy

```powershell
cd app\clinical-rag-chat
databricks apps deploy -t default --profile dbc-7c3eed4c --auto-approve
```

## Re-ingest after uploads

Upload via the app (UC Files API) or:

```powershell
# Prefer Files REST API so Spark/SQL can see the files
$token = (databricks auth token -p dbc-7c3eed4c -o json | ConvertFrom-Json).access_token
# PUT to https://<host>/api/2.0/fs/files/Volumes/workspace/ccda_rag/docs/<folder>/<file>?overwrite=true

databricks jobs run-now 1026899989768735 --profile dbc-7c3eed4c --no-wait
databricks vector-search-indexes sync-index workspace.ccda_rag.document_chunks_index --profile dbc-7c3eed4c
```

If sync is stuck (`pipeline ... CREATED`), stop the VS sync pipeline then re-sync:

```powershell
databricks pipelines stop bd6baf44-ea7a-4d58-89b2-06a974da54cd --profile dbc-7c3eed4c
databricks pipelines start-update bd6baf44-ea7a-4d58-89b2-06a974da54cd --profile dbc-7c3eed4c
```

## Local notes

Uses Databricks Apps service-principal auth (`WorkspaceClient()` / `Config()`). No tokens in source.
Synthetic demo data only — AI answers must be verified.
