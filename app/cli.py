from __future__ import annotations

import argparse
import logging
import threading
from pathlib import Path

import cv2
import uvicorn

from app.config import get_settings
from app.dependencies import OLLAMA_FALLBACK, RECOGNIZER, ROI_STORE, SESSION_FACTORY, STORAGE
from app.main import app
from app.pipeline import IngestWorker
from app.repository import EventRepository


LOGGER = logging.getLogger(__name__)


def _repository_factory() -> EventRepository:
    return EventRepository(SESSION_FACTORY())


def _create_worker() -> IngestWorker:
    return IngestWorker(
        settings=get_settings(),
        repository_factory=_repository_factory,
        storage=STORAGE,
        recognizer=RECOGNIZER,
        ollama_fallback=OLLAMA_FALLBACK,
        roi_provider=ROI_STORE.get,
    )


def run_api() -> None:
    settings = get_settings()
    uvicorn.run(app, host=settings.host, port=settings.port)


def run_worker() -> None:
    worker = _create_worker()
    try:
        worker.run_forever()
    except KeyboardInterrupt:
        LOGGER.info("Worker interrupted by user.")
        worker.stop()


def run_dev() -> None:
    worker = _create_worker()
    thread = threading.Thread(target=worker.run_forever, daemon=True)
    thread.start()
    try:
        run_api()
    finally:
        worker.stop()
        thread.join(timeout=5)


def run_backfill(limit: int = 100) -> None:
    settings = get_settings()
    repository = _repository_factory()
    pending_events = list(repository.list_pending_without_plate(limit=limit))
    LOGGER.info("Backfilling %s pending events.", len(pending_events))
    for event in pending_events:
        image_path = STORAGE.root / event.snapshot_path
        if not image_path.exists():
            LOGGER.warning("Snapshot missing for event %s: %s", event.id, image_path)
            continue

        frame = cv2.imread(str(image_path))
        if frame is None:
            LOGGER.warning("Unable to read snapshot for event %s", event.id)
            continue

        result = None
        if OLLAMA_FALLBACK is not None:
            result = OLLAMA_FALLBACK.read_plate(frame)
        if result is None or not result.plate_number_final:
            continue

        plate_crop_path = event.plate_crop_path
        if result.plate_crop is not None:
            plate_name = Path(event.snapshot_path).stem + "_plate.jpg"
            plate_path = (STORAGE.root / Path(event.snapshot_path).parent / plate_name)
            plate_path.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(plate_path), result.plate_crop)
            plate_crop_path = str(plate_path.relative_to(STORAGE.root))

        repository.update_ocr_result(
            event.id,
            plate_number_raw=result.plate_number_raw,
            plate_number_final=result.plate_number_final,
            ocr_confidence=result.confidence,
            plate_crop_path=plate_crop_path,
            review_status="pending",
        )
        LOGGER.info(
            "Backfilled event id=%s plate=%s confidence=%.2f",
            event.id,
            result.plate_number_final,
            result.confidence,
        )
    repository.session.close()


def main() -> None:
    parser = argparse.ArgumentParser(description="Academy ANPR runtime")
    parser.add_argument("mode", choices=["api", "worker", "dev", "backfill"], help="Runtime mode")
    parser.add_argument("--limit", type=int, default=100, help="Maximum pending events to backfill")
    args = parser.parse_args()

    if args.mode == "api":
        run_api()
        return
    if args.mode == "worker":
        run_worker()
        return
    if args.mode == "backfill":
        run_backfill(limit=args.limit)
        return
    run_dev()


if __name__ == "__main__":
    main()
