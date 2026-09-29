"""Google Cloud Storage credentials without ever writing the key to disk or logs."""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_ENV = "CUSTOM_KEY"
"""Environment variable Code Ocean exposes the attached service-account key under."""


def gcs_token(*, env: str = DEFAULT_ENV, path: Path | None = None) -> dict | str:
    """Service-account credentials for ``gcsfs``: key JSON from ``env``, else a key file path.

    Raises
    ------
    RuntimeError
        If neither source is available (the message never includes key material).
    """
    if path is not None:
        return str(path)
    content = os.environ.get(env, "").strip()
    if content:
        try:
            return json.loads(content)
        except json.JSONDecodeError:
            if Path(content).is_file():
                return content
            raise RuntimeError(f"${env} is set but is neither key JSON nor a key file") from None
    if os.environ.get("GOOGLE_APPLICATION_CREDENTIALS"):
        return os.environ["GOOGLE_APPLICATION_CREDENTIALS"]
    raise RuntimeError(f"No GCS credentials: set ${env} (key JSON) or pass a key file")
