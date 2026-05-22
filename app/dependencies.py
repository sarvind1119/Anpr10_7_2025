from __future__ import annotations

from collections.abc import Generator

from sqlalchemy.orm import Session

from app.config import get_settings
from app.database import create_session_factory
from app.logging_config import configure_logging
from app.ollama_fallback import OllamaPlateFallback
from app.ocr import PlateRecognizer
from app.repository import EventRepository
from app.roi import ROIStore
from app.storage import SnapshotStorage


SETTINGS = get_settings()
configure_logging(SETTINGS.log_dir)
SESSION_FACTORY = create_session_factory(SETTINGS.db_path)
STORAGE = SnapshotStorage(SETTINGS.snapshot_dir)
ROI_STORE = ROIStore(SETTINGS.roi_config_path)
RECOGNIZER = PlateRecognizer(
    tesseract_cmd=SETTINGS.tesseract_cmd,
    confidence_threshold=SETTINGS.ocr_confidence_threshold,
)
OLLAMA_FALLBACK = (
    OllamaPlateFallback(base_url=SETTINGS.ollama_url, model=SETTINGS.ollama_model)
    if SETTINGS.enable_ollama_fallback
    else None
)


def get_session() -> Generator[Session, None, None]:
    session = SESSION_FACTORY()
    try:
        yield session
    finally:
        session.close()


def get_repository(session: Session) -> EventRepository:
    return EventRepository(session)
