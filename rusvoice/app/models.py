"""Download model assets with visible progress, timeout, mirror fallback and safe extraction.

Model archives and weights retain their original third-party licenses.
"""
from __future__ import annotations

import json
import logging
import shutil
import tarfile
import tempfile
import time
import urllib.request
import zipfile
from pathlib import Path, PurePosixPath
from typing import Callable

LOG = logging.getLogger("rusvoice.models")
Progress = Callable[..., None]
VOSK_NAME = "vosk-model-small-ru-0.22"
VOSK_SOURCES = (
    f"https://alphacephei.com/vosk/models/{VOSK_NAME}.zip",
    f"https://huggingface.co/rhasspy/vosk-models/resolve/main/ru/{VOSK_NAME}.zip",
)
SUPERTONIC_NAME = "sherpa-onnx-supertonic-3-tts-int8-2026-05-11"
SUPERTONIC_HF = f"https://huggingface.co/csukuangfj2/{SUPERTONIC_NAME}/resolve/main/"
SUPERTONIC_FILES = (
    "duration_predictor.int8.onnx", "text_encoder.int8.onnx",
    "vector_estimator.int8.onnx", "vocoder.int8.onnx", "tts.json",
    "unicode_indexer.bin", "voice.bin",
)
SUPERTONIC_GITHUB = f"https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/{SUPERTONIC_NAME}.tar.bz2"
MAX_ARCHIVE_BYTES = 300 * 1024 * 1024
MAX_UNPACKED_BYTES = 600 * 1024 * 1024

# Minimum plausible sizes: prevent a CDN HTML error page from being treated as model weights.
SUPERTONIC_MIN_BYTES = {
    "duration_predictor.int8.onnx": 1_000_000,
    "text_encoder.int8.onnx": 10_000_000,
    "vector_estimator.int8.onnx": 10_000_000,
    "vocoder.int8.onnx": 5_000_000,
    "tts.json": 1_000,
    "unicode_indexer.bin": 50_000,
    "voice.bin": 100_000,
}


def supertonic_installed(target: Path) -> bool:
    return all((target / f).is_file() and (target / f).stat().st_size >= minimum
               for f, minimum in SUPERTONIC_MIN_BYTES.items())




def report(callback: Progress, **kw):
    if callback:
        callback(**kw)


def fetch(url: str, dest: Path, callback: Progress | None = None, name: str = "model", max_bytes: int = MAX_ARCHIVE_BYTES) -> None:
    """HTTP download, with per-read timeout, bounded size, and resumable .part file."""
    part = dest.with_name(dest.name + ".part")
    part.parent.mkdir(parents=True, exist_ok=True)
    existing = part.stat().st_size if part.exists() else 0
    headers = {"User-Agent": "RusVoice-HA/0.2 (+https://github.com/ruslan200065/RusVoice-HA)"}
    if existing:
        headers["Range"] = f"bytes={existing}-"
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=20) as response:
        # If the CDN ignores Range, discard the old partial data.
        mode = "ab" if existing and response.status == 206 else "wb"
        if mode == "wb":
            existing = 0
        content_len = response.headers.get("Content-Length")
        total = existing + int(content_len) if content_len and content_len.isdecimal() else 0
        if total and total > max_bytes:
            raise ValueError("Model download exceeds size limit")
        done = existing
        started = time.monotonic()
        with part.open(mode) as output:
            while True:
                chunk = response.read(256 * 1024)
                if not chunk:
                    break
                done += len(chunk)
                if done > max_bytes:
                    raise ValueError("Model download exceeds size limit")
                output.write(chunk)
                report(callback, phase="downloading", file=name, received=done,
                       total=total, speed=int((done - existing) / max(time.monotonic() - started, .01)))
        if total and done != total:
            raise IOError(f"Incomplete download: {done} of {total} bytes")
    part.replace(dest)


def _safe_members(members, root_name: str, limit: int = MAX_UNPACKED_BYTES):
    total = 0
    for member in members:
        # Do not let untrusted archives escape their extraction directory.
        pathname = member.filename if hasattr(member, "filename") else member.name
        path = PurePosixPath(pathname)
        if path.is_absolute() or ".." in path.parts or not path.parts or path.parts[0] != root_name:
            raise ValueError(f"Unexpected path in archive: {pathname}")
        if hasattr(member, "issym") and (member.issym() or member.islnk()):
            raise ValueError("Archive links are not allowed")
        if hasattr(member, "isfile") and not (member.isfile() or member.isdir()):
            raise ValueError("Unexpected archive member")
        total += member.file_size if hasattr(member, "file_size") else member.size
        if total > limit:
            raise ValueError("Unpacked archive exceeds size limit")
    return members


def vosk_installed(folder: Path) -> bool:
    """Vosk supports either a compact HCLr+Gr graph or a monolithic HCLG graph.

    The official vosk-model-small-ru-0.22 ships with HCLr.fst + Gr.fst.
    Reject missing/incomplete graphs, not an otherwise valid compact model.
    """
    required = ("am/final.mdl", "conf/model.conf")
    if not all((folder / name).is_file() and (folder / name).stat().st_size > 0
               for name in required):
        return False

    def exists(name: str) -> bool:
        path = folder / "graph" / name
        return path.is_file() and path.stat().st_size > 0

    return exists("HCLG.fst") or (exists("HCLr.fst") and exists("Gr.fst"))


def ensure_vosk(root: Path, callback: Progress | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    target = root / VOSK_NAME
    if vosk_installed(target):
        report(callback, phase="installed", file=VOSK_NAME)
        return
    error = None
    for idx, source in enumerate(VOSK_SOURCES, 1):
        try:
            report(callback, phase="connecting", source=f"mirror {idx}/{len(VOSK_SOURCES)}", file=VOSK_NAME)
            with tempfile.TemporaryDirectory(dir=root, prefix=".vosk-") as tmp:
                tmpdir = Path(tmp)
                archive = root / f".{VOSK_NAME}.zip"
                fetch(source, archive, callback, name=VOSK_NAME)
                report(callback, phase="extracting", file=VOSK_NAME)
                with zipfile.ZipFile(archive) as z:
                    members = _safe_members(z.infolist(), VOSK_NAME)
                    z.extractall(tmpdir, members=members)
                unpacked = tmpdir / VOSK_NAME
                if not vosk_installed(unpacked):
                    raise ValueError("Downloaded Vosk model does not contain expected files")
                if target.exists():
                    shutil.rmtree(target)
                unpacked.rename(target)
                archive.unlink(missing_ok=True)
                report(callback, phase="installed", file=VOSK_NAME)
                return
        except Exception as exc:
            LOG.warning("Vosk mirror %s failed: %s", idx, exc)
            error = exc
            # Delete corrupt archives and partials before trying the next mirror.
            (root / f".{VOSK_NAME}.zip").unlink(missing_ok=True)
            (root / f".{VOSK_NAME}.zip.part").unlink(missing_ok=True)
    raise RuntimeError(f"No Vosk download mirror succeeded: {error}")


def ensure_supertonic(root: Path, callback: Progress | None = None) -> None:
    root.mkdir(parents=True, exist_ok=True)
    target = root / SUPERTONIC_NAME
    if supertonic_installed(target):
        report(callback, phase="installed", file=SUPERTONIC_NAME)
        return
    target.mkdir(parents=True, exist_ok=True)
    errors = []
    # Download individual, resumable weights from Hugging Face; this avoids
    # re-downloading ~145MB after a disconnect or addon restart.
    for i, name in enumerate(SUPERTONIC_FILES, 1):
        path = target / name
        if path.is_file() and path.stat().st_size >= SUPERTONIC_MIN_BYTES[name]:
            continue
        try:
            report(callback, phase="connecting", file=name, file_index=i, file_count=len(SUPERTONIC_FILES))
            fetch(SUPERTONIC_HF + name + "?download=true", path, callback, name=name)
        except Exception as exc:
            LOG.warning("Supertonic HF download failed: %s", exc)
            errors.append(str(exc))
            break
    if supertonic_installed(target):
        # Quick sanity check to reject login HTML disguised as a model JSON.
        json.loads((target / "tts.json").read_text(encoding="utf-8"))
        report(callback, phase="installed", file=SUPERTONIC_NAME)
        return
    # Alternative official GitHub release archive.
    report(callback, phase="connecting", file="Supertonic GitHub release (fallback)")
    archive = root / f".{SUPERTONIC_NAME}.tar.bz2"
    try:
        fetch(SUPERTONIC_GITHUB, archive, callback, name=SUPERTONIC_NAME)
        report(callback, phase="extracting", file=SUPERTONIC_NAME)
        with tempfile.TemporaryDirectory(dir=root, prefix=".supertonic-") as tmp:
            tmpdir = Path(tmp)
            with tarfile.open(archive, "r:bz2") as t:
                members = _safe_members(t.getmembers(), SUPERTONIC_NAME)
                t.extractall(tmpdir, members=members, filter="data")
            unpacked = tmpdir / SUPERTONIC_NAME
            if not supertonic_installed(unpacked):
                raise ValueError("Incomplete Supertonic archive")
            if target.exists():
                shutil.rmtree(target)
            unpacked.rename(target)
            report(callback, phase="installed", file=SUPERTONIC_NAME)
    except Exception as exc:
        errors.append(str(exc))
        raise RuntimeError("Supertonic download failed: " + "; ".join(errors)) from exc
    finally:
        archive.unlink(missing_ok=True)


def ensure_piper(root: Path, voice: str, callback: Progress | None = None) -> None:
    """Explicit Piper downloads with progress and retry, instead of invisible startup."""
    root.mkdir(parents=True, exist_ok=True)
    parts = voice.split("-")
    if len(parts) != 3 or parts[0] != "ru_RU":
        raise ValueError("Unexpected Piper voice name")
    base = f"https://huggingface.co/rhasspy/piper-voices/resolve/main/ru/ru_RU/{parts[1]}/{parts[2]}/{voice}"
    for suffix in (".onnx", ".onnx.json"):
        path = root / (voice + suffix)
        minimum = 1_000_000 if suffix == ".onnx" else 300
        if path.is_file() and path.stat().st_size >= minimum:
            continue
        path.unlink(missing_ok=True)
        report(callback, phase="connecting", file=path.name)
        fetch(base + suffix + "?download=true", path, callback, name=path.name)
    json.loads((root / (voice + ".onnx.json")).read_text(encoding="utf-8"))
    report(callback, phase="installed", file=voice)
