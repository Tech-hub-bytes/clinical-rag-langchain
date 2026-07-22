"""Upload clinical documents into the UC volume folder layout used by ingest."""

from __future__ import annotations

import io
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import PurePosixPath

from databricks.sdk import WorkspaceClient

from config import UPLOAD_FOLDERS, VOLUME_ROOT


@dataclass(frozen=True)
class UploadResult:
    folder: str
    volume_path: str
    filename: str


def _safe_filename(name: str) -> str:
    base = PurePosixPath(name.replace("\\", "/")).name
    cleaned = re.sub(r"[^\w.\- ()]+", "_", base).strip("._ ")
    return cleaned or "upload.bin"


def resolve_upload_folder(filename: str, preferred: str | None = None) -> str:
    """Map a filename to ccda/pdfs/hl7/fhir. Prefer explicit folder when provided."""
    ext = PurePosixPath(filename).suffix.lower()
    if preferred:
        if preferred not in UPLOAD_FOLDERS:
            raise ValueError(f"Unknown folder '{preferred}'")
        allowed = UPLOAD_FOLDERS[preferred]
        if ext not in allowed:
            raise ValueError(
                f"File '{filename}' ({ext or 'no extension'}) does not belong in "
                f"{preferred}/ (allowed: {', '.join(allowed)})"
            )
        return preferred
    for folder, exts in UPLOAD_FOLDERS.items():
        if ext in exts:
            return folder
    allowed = ", ".join(
        f"{folder}/ ({', '.join(exts)})" for folder, exts in UPLOAD_FOLDERS.items()
    )
    raise ValueError(f"Unsupported file type '{ext or '(none)'}'. Use: {allowed}")


@lru_cache(maxsize=1)
def get_workspace_client() -> WorkspaceClient:
    """App SP / workspace env credentials — no hardcoded tokens."""
    return WorkspaceClient()


def ensure_upload_folders(client: WorkspaceClient | None = None) -> None:
    w = client or get_workspace_client()
    for folder in UPLOAD_FOLDERS:
        path = f"{VOLUME_ROOT}/{folder}"
        try:
            w.files.create_directory(path)
        except Exception:
            # Directory may already exist; list to confirm reachability.
            try:
                next(iter(w.files.list_directory_contents(path)), None)
            except Exception as exc:
                raise RuntimeError(f"Cannot access volume folder {path}: {exc}") from exc


def upload_bytes(
    filename: str,
    data: bytes,
    *,
    folder: str | None = None,
    overwrite: bool = True,
    client: WorkspaceClient | None = None,
) -> UploadResult:
    """Write file bytes into the appropriate volume subfolder."""
    w = client or get_workspace_client()
    target_folder = resolve_upload_folder(filename, folder)
    safe_name = _safe_filename(filename)
    volume_path = f"{VOLUME_ROOT}/{target_folder}/{safe_name}"

    ensure_upload_folders(w)
    w.files.upload(volume_path, io.BytesIO(data), overwrite=overwrite)
    return UploadResult(folder=target_folder, volume_path=volume_path, filename=safe_name)


def upload_streamlit_file(
    uploaded_file,
    *,
    folder: str | None = None,
    overwrite: bool = True,
) -> UploadResult:
    """Upload a Streamlit UploadedFile into the UC volume."""
    return upload_bytes(
        uploaded_file.name,
        uploaded_file.getvalue(),
        folder=folder,
        overwrite=overwrite,
    )
