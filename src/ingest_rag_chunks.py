# Databricks notebook source
# MAGIC %md
# MAGIC # C-CDA + PDF + HL7 + FHIR RAG ingest
# MAGIC Builds `workspace.ccda_rag.document_chunks` for Vector Search
# MAGIC (C-CDA XML/MD, PDFs, HL7 v2 messages, FHIR JSON).

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import types as T
import hashlib
import json
import re
import xml.etree.ElementTree as ET

VOLUME_ROOT = "/Volumes/workspace/ccda_rag/docs"
CATALOG = "workspace"
SCHEMA = "ccda_rag"
CHUNKS_TABLE = f"{CATALOG}.{SCHEMA}.document_chunks"

spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{SCHEMA}")

# COMMAND ----------

def local(tag: str) -> str:
    return tag.split("}")[-1] if "}" in tag else tag


def strip_text(elem) -> str:
    if elem is None:
        return ""
    parts = []
    if elem.text:
        parts.append(elem.text)
    for child in list(elem):
        parts.append(strip_text(child))
        if child.tail:
            parts.append(child.tail)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()


def patient_from_path(path: str, hint: str = "") -> str:
    base = path.split("/")[-1]
    base = re.sub(r"\.(xml|pdf|md|hl7|json|txt)$", "", base, flags=re.I)
    base = re.sub(r"\s*\(\d+\)\s*$", "", base)
    # Drop common document-type suffixes so PDF/CCDA share one patient label
    base = re.sub(
        r"(?i)[_\s-]*(clinical[_\s-]*summary|ccda|adt[_\s-]*a0\d|oru[_\s-]*r0\d|fhir[_\s-]*bundle|summary)$",
        "",
        base,
    )
    base = re.sub(r"[_-]+", " ", base)
    base = re.sub(r"\d{6,}", " ", base)
    cleaned = " ".join(w.capitalize() for w in base.split() if w)
    low = (path + " " + hint).lower()
    if "emma" in low:
        return "EmmaTestPatient"
    if "james" in low:
        return "James Testpatient"
    if "alice" in low or "newman" in low:
        return "Alice Jones"
    if "john" in low and "smith" in low:
        return "John Smith"
    if "emily" in low and "davis" in low:
        return "Emily Davis"
    if "david" in low and "miller" in low:
        return "David Miller"
    if "sophia" in low and "anderson" in low:
        return "Sophia Anderson"
    if "sample" in low or "merged" in low:
        return "Sample Patient"
    # Prefer structured hint (HL7/CCDA/FHIR name) when present
    if hint and len(hint.strip()) > 2:
        h = re.sub(
            r"(?i)\b(clinical\s+summary|ccda|summary)\b",
            "",
            hint.strip(),
        )
        h = re.sub(r"\s+", " ", h).strip(" ,.-")
        if h:
            return h
    return cleaned or "Unknown patient"


def chunk_text(text: str, size: int = 1200, overlap: int = 150):
    text = re.sub(r"\s+", " ", text or "").strip()
    if not text:
        return []
    out, i = [], 0
    while i < len(text):
        out.append(text[i : i + size])
        i += max(size - overlap, 1)
    return out


def rows_for_parts(path, doc_type, patient, title, parts):
    rows = []
    for i, part in enumerate(parts):
        cid = hashlib.md5(f"{path}#{title}#{i}".encode()).hexdigest()
        content = f"# {title}\nPatient: {patient}\nSource: {path}\n\n{part}"
        rows.append(
            {
                "chunk_id": cid,
                "source_path": path,
                "doc_type": doc_type,
                "patient": patient,
                "section_title": title if i == 0 else f"{title} ({i+1})",
                "content": content,
                "content_embed": f"Patient {patient}. {title}. {part}",
            }
        )
    return rows


def parse_ccda_xml(path: str, xml_text: str):
    try:
        root = ET.fromstring(xml_text)
    except Exception as e:
        return rows_for_parts(path, "ccda_xml", patient_from_path(path), "parse_error", [str(e)])

    given, family = [], []
    for el in root.iter():
        ln = local(el.tag)
        if ln == "given" and el.text:
            given.append(el.text.strip())
        if ln == "family" and el.text:
            family.append(el.text.strip())
    patient = patient_from_path(path, " ".join(given + family))
    title_el = next((el for el in root.iter() if local(el.tag) == "title"), None)
    doc_title = strip_text(title_el) if title_el is not None else path.split("/")[-1]

    rows = rows_for_parts(path, "ccda_xml", patient, "Overview", [f"Document: {doc_title}. Patient: {patient}."])
    seen = set()
    for section in root.iter():
        if local(section.tag) != "section":
            continue
        stitle, narrative = "Untitled", ""
        for child in list(section):
            if local(child.tag) == "title":
                stitle = strip_text(child) or stitle
            if local(child.tag) == "text":
                narrative = strip_text(child)
        key = stitle.lower()
        if key in seen:
            continue
        body = narrative or strip_text(section)
        if len(body) < 20:
            continue
        seen.add(key)
        rows.extend(rows_for_parts(path, "ccda_xml", patient, stitle, chunk_text(body)))
    return rows


def parse_markdown(path: str, text: str):
    patient = patient_from_path(path, text[:500])
    return rows_for_parts(path, "markdown", patient, "markdown", chunk_text(text))


def _hl7_field(segment: str, index: int) -> str:
    """1-based field index (field 0 is the segment ID)."""
    parts = segment.split("|")
    if index < 0 or index >= len(parts):
        return ""
    return parts[index].strip()


def _hl7_patient_name(pid_segment: str) -> str:
    # PID-5: family^given^middle^suffix^prefix
    name = _hl7_field(pid_segment, 5)
    if not name:
        return ""
    comps = name.split("^")
    family = comps[0].strip() if len(comps) > 0 else ""
    given = comps[1].strip() if len(comps) > 1 else ""
    return " ".join(p for p in [given, family] if p).strip()


HL7_SEGMENT_TITLES = {
    "MSH": "Message Header",
    "EVN": "Event Type",
    "PID": "Patient Identification",
    "PD1": "Patient Additional Demographics",
    "NK1": "Next of Kin",
    "PV1": "Patient Visit",
    "PV2": "Patient Visit Additional",
    "ORC": "Common Order",
    "OBR": "Observation Request",
    "OBX": "Observation / Result",
    "AL1": "Allergy",
    "DG1": "Diagnosis",
    "PR1": "Procedures",
    "RXE": "Pharmacy Encoded Order",
    "RXA": "Pharmacy Administration",
    "RXR": "Pharmacy Route",
    "IN1": "Insurance",
    "GT1": "Guarantor",
    "NTE": "Notes",
}


def parse_hl7_v2(path: str, text: str):
    """Parse HL7 v2 pipe-delimited messages into RAG chunks."""
    raw = (text or "").replace("\r\n", "\n").replace("\r", "\n")
    segments = [ln.strip() for ln in raw.split("\n") if ln.strip() and not ln.strip().startswith("#")]
    if not segments:
        return rows_for_parts(path, "hl7_v2", patient_from_path(path), "empty", ["Empty HL7 file."])

    # Support multi-message files (each starting with MSH)
    messages = []
    current = []
    for seg in segments:
        if seg.startswith("MSH") and current:
            messages.append(current)
            current = [seg]
        else:
            current.append(seg)
    if current:
        messages.append(current)

    rows = []
    for mi, msg in enumerate(messages):
        pid = next((s for s in msg if s.startswith("PID")), "")
        patient = patient_from_path(path, _hl7_patient_name(pid))
        msh = next((s for s in msg if s.startswith("MSH")), msg[0])
        msg_type = _hl7_field(msh, 8) or "UNKNOWN"
        prefix = f"HL7 msg {mi+1}" if len(messages) > 1 else "HL7"

        overview = (
            f"{prefix} message type {msg_type}. Patient: {patient}. "
            f"Segments: {len(msg)}. Source file: {path.split('/')[-1]}."
        )
        rows.extend(rows_for_parts(path, "hl7_v2", patient, f"{prefix} Overview", [overview]))

        # Group consecutive same-type segments for readable chunks
        groups = []
        g_type, g_lines = None, []
        for seg in msg:
            sid = seg.split("|", 1)[0].strip().upper()[:3]
            if g_type is None:
                g_type, g_lines = sid, [seg]
            elif sid == g_type and sid in ("OBX", "NTE", "AL1", "DG1", "NK1"):
                g_lines.append(seg)
            else:
                groups.append((g_type, g_lines))
                g_type, g_lines = sid, [seg]
        if g_lines:
            groups.append((g_type, g_lines))

        for sid, lines in groups:
            title = f"{prefix} {HL7_SEGMENT_TITLES.get(sid, sid)}"
            body = "\n".join(lines)
            # Human-readable summary line for common clinical segments
            if sid == "PID":
                body = f"Patient identification: {_hl7_patient_name(pid) or patient}\n{body}"
            elif sid == "OBX":
                summaries = []
                for ln in lines:
                    obs = _hl7_field(ln, 3)
                    val = _hl7_field(ln, 5)
                    units = _hl7_field(ln, 6)
                    if obs or val:
                        summaries.append(f"{obs}: {val} {units}".strip())
                if summaries:
                    body = "Observations:\n" + "\n".join(summaries) + "\n\n" + body
            elif sid == "DG1":
                summaries = []
                for ln in lines:
                    code = _hl7_field(ln, 3)
                    desc = _hl7_field(ln, 4)
                    if code or desc:
                        summaries.append(f"{code} {desc}".strip())
                if summaries:
                    body = "Diagnoses:\n" + "\n".join(summaries) + "\n\n" + body
            elif sid == "AL1":
                summaries = []
                for ln in lines:
                    allergen = _hl7_field(ln, 3)
                    reaction = _hl7_field(ln, 5)
                    if allergen:
                        summaries.append(f"{allergen} — {reaction}".strip(" —"))
                if summaries:
                    body = "Allergies:\n" + "\n".join(summaries) + "\n\n" + body
            rows.extend(rows_for_parts(path, "hl7_v2", patient, title, chunk_text(body, size=1400)))
    return rows


def _fhir_human_name(name_obj) -> str:
    if isinstance(name_obj, list) and name_obj:
        name_obj = name_obj[0]
    if not isinstance(name_obj, dict):
        return ""
    if name_obj.get("text"):
        return str(name_obj["text"]).strip()
    family = name_obj.get("family") or ""
    given = name_obj.get("given") or []
    if isinstance(given, list):
        given = " ".join(str(g) for g in given)
    return " ".join(p for p in [str(given).strip(), str(family).strip()] if p).strip()


def _fhir_patient_hint(obj) -> str:
    if not isinstance(obj, dict):
        return ""
    if obj.get("resourceType") == "Patient":
        return _fhir_human_name(obj.get("name"))
    if obj.get("resourceType") == "Bundle":
        for entry in obj.get("entry") or []:
            res = entry.get("resource") if isinstance(entry, dict) else None
            if isinstance(res, dict) and res.get("resourceType") == "Patient":
                return _fhir_human_name(res.get("name"))
    return ""


def _flatten_fhir_value(val, depth=0):
    if depth > 6:
        return ""
    if val is None or isinstance(val, bool):
        return str(val) if val is not None else ""
    if isinstance(val, (int, float)):
        return str(val)
    if isinstance(val, str):
        return val.strip()
    if isinstance(val, list):
        return "; ".join(p for p in (_flatten_fhir_value(v, depth + 1) for v in val) if p)
    if isinstance(val, dict):
        # Prefer readable FHIR patterns
        if "display" in val and val.get("display"):
            code = val.get("code") or ""
            return f"{val['display']}" + (f" ({code})" if code else "")
        if "text" in val and isinstance(val["text"], str):
            return val["text"]
        if "coding" in val:
            return _flatten_fhir_value(val.get("coding"), depth + 1)
        parts = []
        for k, v in val.items():
            if k in ("extension", "meta", "identifier"):
                continue
            fv = _flatten_fhir_value(v, depth + 1)
            if fv:
                parts.append(f"{k}: {fv}")
        return ", ".join(parts[:40])
    return str(val)


def _fhir_resource_text(resource: dict) -> tuple:
    rtype = resource.get("resourceType") or "Resource"
    rid = resource.get("id") or ""
    title = f"FHIR {rtype}" + (f"/{rid}" if rid else "")
    lines = [f"resourceType: {rtype}"]
    if rid:
        lines.append(f"id: {rid}")
    # Highlight common clinical fields first
    priority = (
        "name", "gender", "birthDate", "status", "code", "category", "subject",
        "patient", "encounter", "effectiveDateTime", "effectivePeriod", "issued",
        "valueQuantity", "valueString", "valueCodeableConcept", "conclusion",
        "clinicalStatus", "verificationStatus", "severity", "reaction",
        "medicationCodeableConcept", "medicationReference", "dosageInstruction",
        "reasonCode", "onsetDateTime", "abatementDateTime", "note", "text",
        "address", "telecom", "maritalStatus", "communication",
    )
    seen = set()
    for key in priority:
        if key in resource:
            seen.add(key)
            fv = _flatten_fhir_value(resource[key])
            if fv:
                lines.append(f"{key}: {fv}")
    for key, val in resource.items():
        if key in seen or key in ("resourceType", "id", "meta", "extension", "identifier"):
            continue
        fv = _flatten_fhir_value(val)
        if fv and len(fv) < 2000:
            lines.append(f"{key}: {fv}")
    return title, "\n".join(lines)


def parse_fhir_json(path: str, text: str):
    """Parse FHIR JSON (Bundle or single resource) into RAG chunks."""
    try:
        obj = json.loads(text)
    except Exception as e:
        return rows_for_parts(path, "fhir_json", patient_from_path(path), "parse_error", [str(e)])

    if not isinstance(obj, dict):
        return rows_for_parts(
            path, "fhir_json", patient_from_path(path), "unsupported",
            ["JSON root must be a FHIR resource or Bundle object."],
        )

    patient = patient_from_path(path, _fhir_patient_hint(obj))
    resources = []
    if obj.get("resourceType") == "Bundle":
        btype = obj.get("type") or "collection"
        rows = rows_for_parts(
            path, "fhir_json", patient, "FHIR Bundle Overview",
            [f"FHIR Bundle type={btype}. Patient: {patient}. Entries: {len(obj.get('entry') or [])}."],
        )
        for entry in obj.get("entry") or []:
            if isinstance(entry, dict) and isinstance(entry.get("resource"), dict):
                resources.append(entry["resource"])
    elif obj.get("resourceType"):
        rows = rows_for_parts(
            path, "fhir_json", patient, "FHIR Document Overview",
            [f"FHIR {obj.get('resourceType')} document. Patient: {patient}."],
        )
        resources.append(obj)
    else:
        # Not clearly FHIR — still index as searchable JSON text
        flat = json.dumps(obj, indent=2)[:50000]
        return rows_for_parts(path, "fhir_json", patient, "json", chunk_text(flat, size=1400))

    for res in resources:
        title, body = _fhir_resource_text(res)
        rows.extend(rows_for_parts(path, "fhir_json", patient, title, chunk_text(body, size=1400)))
    return rows


def looks_like_hl7(text: str) -> bool:
    t = (text or "").lstrip("\ufeff \t\r\n")
    return t.startswith("MSH|") or t.startswith("MSH^")


def looks_like_fhir(text: str) -> bool:
    t = (text or "").lstrip("\ufeff \t\r\n")
    if not t.startswith("{") and not t.startswith("["):
        return False
    try:
        obj = json.loads(t)
    except Exception:
        return False
    if isinstance(obj, dict) and (obj.get("resourceType") or "entry" in obj):
        return True
    return False


def extract_text_from_parsed_variant(parsed_obj) -> str:
    """Best-effort flatten of ai_parse_document VARIANT / dict / string."""
    if parsed_obj is None:
        return ""
    if isinstance(parsed_obj, str):
        # May be JSON string
        try:
            parsed_obj = json.loads(parsed_obj)
        except Exception:
            return parsed_obj
    if isinstance(parsed_obj, dict):
        # Common shapes: pages[].elements[].content / text
        parts = []
        pages = parsed_obj.get("document", {}).get("pages") if isinstance(parsed_obj.get("document"), dict) else None
        if pages is None:
            pages = parsed_obj.get("pages")
        if isinstance(pages, list):
            for p in pages:
                for el in (p.get("elements") or p.get("lines") or []):
                    if isinstance(el, dict):
                        t = el.get("content") or el.get("text") or el.get("value")
                        if t:
                            parts.append(str(t))
                    elif isinstance(el, str):
                        parts.append(el)
        # Fallback: walk strings
        if not parts:

            def walk(o):
                if isinstance(o, str) and len(o.strip()) > 2:
                    parts.append(o)
                elif isinstance(o, dict):
                    for v in o.values():
                        walk(v)
                elif isinstance(o, list):
                    for v in o:
                        walk(v)

            walk(parsed_obj)
        return re.sub(r"\s+", " ", " ".join(parts)).strip()
    return str(parsed_obj)

# COMMAND ----------

# Load recursively then filter by extension (brace globs are unreliable on some volume backends)
_ALLOWED_EXT = ("pdf", "xml", "md", "hl7", "json", "txt")
files_df = (
    spark.read.format("binaryFile")
    .option("recursiveFileLookup", "true")
    .load(VOLUME_ROOT)
    .select(
        F.col("path").alias("source_path"),
        F.col("content").alias("raw_bytes"),
        F.lower(F.element_at(F.split(F.col("path"), "\\."), -1)).alias("ext"),
    )
    .filter(F.col("ext").isin(*_ALLOWED_EXT))
)

print("Files found:")
files_df.select("source_path", "ext", F.length("raw_bytes").alias("nbytes")).show(100, False)

# COMMAND ----------

# PDF: ai_parse_document → string → chunk on driver (layout-agnostic)
pdf_rows = []
pdf_df = files_df.filter(F.col("ext") == "pdf")
if pdf_df.count() > 0:
    try:
        pdf_parsed = pdf_df.select(
            "source_path",
            F.expr("ai_parse_document(raw_bytes, map('version','2.0'))").alias("parsed"),
        ).select(
            "source_path",
            F.expr("CAST(parsed AS STRING)").alias("parsed_str"),
        )
        for r in pdf_parsed.collect():
            path = r["source_path"]
            patient = patient_from_path(path)
            text = extract_text_from_parsed_variant(r["parsed_str"])
            if len(text) < 40:
                text = (r["parsed_str"] or "")[:20000]
            pdf_rows.extend(rows_for_parts(path, "pdf", patient, "pdf_content", chunk_text(text, size=1400)))
        print(f"Parsed {len(pdf_rows)} PDF chunk rows via ai_parse_document")
    except Exception as e:
        print(f"ai_parse_document failed ({e}); PDF files will be skipped for this run")

# COMMAND ----------

# XML / Markdown / HL7 v2 / FHIR JSON on driver
text_rows = []
for r in files_df.filter(F.col("ext").isin("xml", "md", "hl7", "json", "txt")).collect():
    path = r["source_path"]
    raw = bytes(r["raw_bytes"])
    text = raw.decode("utf-8", errors="ignore")
    ext = (r["ext"] or "").lower()
    if ext == "xml":
        text_rows.extend(parse_ccda_xml(path, text))
    elif ext == "md":
        text_rows.extend(parse_markdown(path, text))
    elif ext == "hl7" or (ext == "txt" and looks_like_hl7(text)):
        text_rows.extend(parse_hl7_v2(path, text))
    elif ext == "json" or (ext == "txt" and looks_like_fhir(text)):
        text_rows.extend(parse_fhir_json(path, text))
    elif ext == "txt":
        # Generic text fallback
        text_rows.extend(
            rows_for_parts(path, "markdown", patient_from_path(path, text[:500]), "text", chunk_text(text))
        )

print(f"Text/XML/HL7/FHIR chunk rows: {len(text_rows)}")

schema = T.StructType(
    [
        T.StructField("chunk_id", T.StringType()),
        T.StructField("source_path", T.StringType()),
        T.StructField("doc_type", T.StringType()),
        T.StructField("patient", T.StringType()),
        T.StructField("section_title", T.StringType()),
        T.StructField("content", T.StringType()),
        T.StructField("content_embed", T.StringType()),
    ]
)

all_rows = text_rows + pdf_rows
if not all_rows:
    raise RuntimeError("No chunks produced — check volume contents")

all_chunks = (
    spark.createDataFrame(all_rows, schema=schema)
    .dropDuplicates(["chunk_id"])
    .withColumn("updated_at", F.current_timestamp())
)

print(f"Total chunks: {all_chunks.count()}")
all_chunks.groupBy("doc_type", "patient").count().orderBy("doc_type", "patient").show(50, False)

# COMMAND ----------

(
    all_chunks.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(CHUNKS_TABLE)
)

spark.sql(
    f"ALTER TABLE {CHUNKS_TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)"
)

spark.sql(
    f"SELECT doc_type, patient, count(*) AS n FROM {CHUNKS_TABLE} GROUP BY 1,2 ORDER BY 1,2"
).show(50, False)
print(f"Wrote {CHUNKS_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Auto-sync Vector Search (TRIGGERED indexes)
# MAGIC This workspace does not support CONTINUOUS Vector Search pipelines.
# MAGIC Sync here so every successful ingest makes new patients searchable without a second manual step.

# COMMAND ----------

import time
from databricks.sdk import WorkspaceClient

VS_ENDPOINT = "ka-3abc305d-vs-endpoint"
VS_INDEX = "workspace.ccda_rag.document_chunks_index"
VS_PIPELINE_ID = "bd6baf44-ea7a-4d58-89b2-06a974da54cd"

w = WorkspaceClient()
sync_ok = False
try:
    from databricks.vector_search.client import VectorSearchClient

    vsc = VectorSearchClient()
    vsc.get_index(endpoint_name=VS_ENDPOINT, index_name=VS_INDEX).sync()
    print(f"Triggered Vector Search sync for {VS_INDEX}")
    sync_ok = True
except Exception as e:
    print(f"VectorSearchClient.sync failed ({e}); trying pipelines.start_update")

if not sync_ok:
    try:
        try:
            w.pipelines.stop(pipeline_id=VS_PIPELINE_ID)
            time.sleep(2)
        except Exception:
            pass
        upd = w.pipelines.start_update(pipeline_id=VS_PIPELINE_ID)
        print(f"Started VS pipeline update: {upd}")
        sync_ok = True
    except Exception as e:
        print(f"WARNING: could not sync Vector Search index: {e}")

if sync_ok:
    # Best-effort wait so job SUCCESS usually means the index is updating / done
    deadline = time.time() + 480
    last = None
    while time.time() < deadline:
        time.sleep(15)
        info = w.pipelines.get(pipeline_id=VS_PIPELINE_ID)
        updates = getattr(info, "latest_updates", None) or []
        latest = updates[0] if updates else None
        state = getattr(latest, "state", None) if latest is not None else None
        last = str(state).split(".")[-1] if state is not None else None
        print(f"VS pipeline state: {last}")
        if last in ("COMPLETED", "FAILED", "CANCELED"):
            break
    if last != "COMPLETED":
        print(f"WARNING: VS sync ended with state={last} (chat may lag until sync finishes)")
    else:
        print("Vector Search sync COMPLETED — new patients are searchable")
