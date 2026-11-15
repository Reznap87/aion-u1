from __future__ import annotations

import base64
from pathlib import Path
from typing import Any


def multimodal_content(
    prompt: str,
    image_path: str,
    root: Path,
    mime_type: str = "image/png",
) -> list[dict[str, Any]]:
    """Baut einen OpenRouter-kompatiblen Multimodal-Content-Block."""

    path = Path(root) / image_path

    if not path.is_file():
        raise FileNotFoundError(f"Bild fehlt: {path}")

    encoded = base64.b64encode(
        path.read_bytes()
    ).decode("ascii")

    return [
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{encoded}"
            },
        },
    ]
