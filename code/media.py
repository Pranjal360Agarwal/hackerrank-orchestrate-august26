"""Local, privacy-preserving media enrichment for router inputs.

Image OCR is optional: the router remains runnable on a clean machine without
Tesseract, but uses it automatically when the executable is available.  Media
never leaves the machine and no generated transcript is persisted.
"""

from __future__ import annotations

import csv
import shutil
import subprocess
from pathlib import Path


def _read_mapping(path: Path, id_column: str) -> dict[str, Path]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return {
            row[id_column]: path.parent / row["file_path"]
            for row in csv.DictReader(handle)
        }


class MediaAnalyzer:
    """Enrich messages with local image text while retaining their original fields."""

    def __init__(self, dataset: Path) -> None:
        self.images = _read_mapping(dataset / "images.csv", "image_id")
        self.voice_notes = _read_mapping(dataset / "voice_notes.csv", "voice_note_id")
        self.tesseract = shutil.which("tesseract")
        self._image_text_cache: dict[str, str] = {}

    def enrich(self, message: dict[str, str]) -> dict[str, str]:
        enriched = dict(message)
        media_type = message.get("media_type")
        media_id = message.get("media_id", "")
        if media_type == "image":
            image_text = self.image_text(media_id)
            if image_text:
                enriched["message_text"] = f"{message.get('message_text', '')}\n[image text] {image_text}"
        # Voice-note files are intentionally resolved here even if no local ASR
        # executable is installed. This validates the dataset reference and keeps
        # an extension point for an offline ASR adapter without network calls.
        elif media_type == "voice":
            enriched["media_available"] = str(self.voice_notes.get(media_id, Path()).is_file())
        return enriched

    def image_text(self, image_id: str) -> str:
        if image_id in self._image_text_cache:
            return self._image_text_cache[image_id]
        image_path = self.images.get(image_id)
        if not image_path or not image_path.is_file() or not self.tesseract:
            self._image_text_cache[image_id] = ""
            return ""
        try:
            completed = subprocess.run(
                [self.tesseract, str(image_path), "stdout"],
                capture_output=True,
                timeout=25,
                check=False,
            )
            text = completed.stdout.decode("utf-8", errors="replace").replace("\x00", " ").strip()
        except (OSError, subprocess.SubprocessError):
            text = ""
        self._image_text_cache[image_id] = text
        return text
