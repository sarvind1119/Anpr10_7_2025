from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent
os.environ.setdefault("YOLO_CONFIG_DIR", str((BASE_DIR / r"storage\ultralytics").resolve()))


@dataclass(frozen=True)
class Settings:
    source_type: str
    source_value: str
    camera_name: str
    db_path: Path
    roi_config_path: Path
    snapshot_dir: Path
    log_dir: Path
    vehicle_model_path: Path
    host: str
    port: int
    timezone: str
    ocr_confidence_threshold: float
    line_y_ratio: float
    tesseract_cmd: str
    enable_ollama_fallback: bool
    ollama_url: str
    ollama_model: str
    max_events: int
    reconnect_seconds: int
    vehicle_classes: tuple[int, ...] = (2, 5, 7)


def _env(name: str, default: str) -> str:
    return os.getenv(name, default).strip()


def _env_bool(name: str, default: str) -> bool:
    return _env(name, default).lower() in {"1", "true", "yes", "on"}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    source_type = _env("ANPR_SOURCE_TYPE", "video").lower()
    if source_type not in {"video", "rtsp"}:
        raise ValueError("ANPR_SOURCE_TYPE must be 'video' or 'rtsp'")

    db_path = (BASE_DIR / _env("ANPR_DB_PATH", r"storage\anpr.db")).resolve()
    roi_config_path = (BASE_DIR / _env("ANPR_ROI_CONFIG_PATH", r"storage\roi.json")).resolve()
    snapshot_dir = (BASE_DIR / _env("ANPR_SNAPSHOT_DIR", r"storage\snapshots")).resolve()
    log_dir = (BASE_DIR / _env("ANPR_LOG_DIR", "logs")).resolve()

    return Settings(
        source_type=source_type,
        source_value=_env("ANPR_SOURCE_VALUE", str(BASE_DIR / "carslbs.mp4")),
        camera_name=_env("ANPR_CAMERA_NAME", "main_gate"),
        db_path=db_path,
        roi_config_path=roi_config_path,
        snapshot_dir=snapshot_dir,
        log_dir=log_dir,
        vehicle_model_path=(BASE_DIR / _env("ANPR_VEHICLE_MODEL_PATH", "yolov8n.pt")).resolve(),
        host=_env("ANPR_HOST", "0.0.0.0"),
        port=int(_env("ANPR_PORT", "8000")),
        timezone=_env("ANPR_TIMEZONE", "Asia/Calcutta"),
        ocr_confidence_threshold=float(_env("ANPR_OCR_CONFIDENCE_THRESHOLD", "65")),
        line_y_ratio=float(_env("ANPR_LINE_Y_RATIO", "0.55")),
        tesseract_cmd=_env("ANPR_TESSERACT_CMD", r"C:\Program Files\Tesseract-OCR\tesseract.EXE"),
        enable_ollama_fallback=_env_bool("ANPR_ENABLE_OLLAMA_FALLBACK", "true"),
        ollama_url=_env("ANPR_OLLAMA_URL", "http://127.0.0.1:11434"),
        ollama_model=_env("ANPR_OLLAMA_MODEL", "gemma3:4b"),
        max_events=int(_env("ANPR_MAX_EVENTS", "500")),
        reconnect_seconds=int(_env("ANPR_RECONNECT_SECONDS", "5")),
    )
