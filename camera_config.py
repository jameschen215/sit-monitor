import os
from pathlib import Path

_ENV_PATH = Path(__file__).parent / ".env"


def _load_dotenv():
    if not _ENV_PATH.exists():
        return
    for line in _ENV_PATH.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        os.environ.setdefault(key.strip(), value.strip())


_load_dotenv()


def get_rtsp_url() -> str:
    url = os.environ.get("RTSP_URL")
    if not url:
        raise RuntimeError(
            "RTSP_URL is not set. Copy .env.example to .env and fill in your "
            "camera credentials, or export RTSP_URL before running."
        )
    return url
