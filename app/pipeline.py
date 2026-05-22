from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable
from zoneinfo import ZoneInfo

import cv2
import numpy as np
from ultralytics import YOLO

from app.config import Settings
from app.ollama_fallback import OllamaPlateFallback
from app.ocr import OCRResult, PlateRecognizer
from app.repository import EventRepository
from app.roi import ROIBox
from app.storage import SnapshotStorage
from app.tracker import CentroidTracker


LOGGER = logging.getLogger(__name__)


@dataclass
class TrackState:
    last_center_y: float
    emitted: bool = False
    best_area: int = 0
    best_result: OCRResult | None = None
    best_frame: np.ndarray | None = None
    last_ocr_area: int = 0


class IngestWorker:
    def __init__(
        self,
        *,
        settings: Settings,
        repository_factory: Callable[[], EventRepository],
        storage: SnapshotStorage,
        recognizer: PlateRecognizer,
        ollama_fallback: OllamaPlateFallback | None = None,
        status_callback: Callable[[str, dict], None] | None = None,
        roi_provider: Callable[[], ROIBox | None] | None = None,
    ) -> None:
        self.settings = settings
        self.repository_factory = repository_factory
        self.storage = storage
        self.recognizer = recognizer
        self.ollama_fallback = ollama_fallback
        self.status_callback = status_callback
        self.roi_provider = roi_provider
        self.model = YOLO(str(settings.vehicle_model_path))
        self.tracker = CentroidTracker()
        self._track_states: dict[int, TrackState] = {}
        self._stop = False
        self._tz = ZoneInfo(settings.timezone)

    def stop(self) -> None:
        self._stop = True

    def run_forever(self) -> None:
        while not self._stop:
            completed = self.run_once()
            if self.settings.source_type == "video":
                break
            if not completed and not self._stop:
                LOGGER.warning("Camera read failed. Reconnecting in %s seconds.", self.settings.reconnect_seconds)
                self._report("reconnect_wait", {"seconds": self.settings.reconnect_seconds})
                time.sleep(self.settings.reconnect_seconds)

    def run_once(self) -> bool:
        if self.settings.source_type == "video" and not Path(self.settings.source_value).exists():
            LOGGER.error("Sample video not found: %s", self.settings.source_value)
            self._report("source_error", {"message": f"Sample video not found: {self.settings.source_value}"})
            return False
        cap = cv2.VideoCapture(self.settings.source_value)
        if not cap.isOpened():
            LOGGER.error("Unable to open source: %s", self.settings.source_value)
            self._report("source_error", {"message": f"Unable to open source: {self.settings.source_value}"})
            return False

        self._report("connected", {"source_type": self.settings.source_type})
        completed = True
        frame_count = 0
        try:
            while not self._stop:
                ok, frame = cap.read()
                if not ok:
                    completed = self.settings.source_type == "video"
                    if not completed:
                        self._report("source_error", {"message": "Frame read failed from live source."})
                    break
                frame_count += 1
                if frame_count % 30 == 0:
                    self._report("heartbeat", {"frame_count": frame_count})
                if frame_count % 2 == 1:
                    continue

                line_y = int(frame.shape[0] * self.settings.line_y_ratio)
                roi = self.roi_provider() if self.roi_provider is not None else None
                results = self.model(
                    frame,
                    verbose=False,
                    classes=list(self.settings.vehicle_classes),
                )
                if not results:
                    continue

                result = results[0]
                boxes = getattr(result, "boxes", None)
                if boxes is None or len(boxes) == 0:
                    continue

                detections = [
                    tuple(int(value) for value in coords)
                    for coords in boxes.xyxy.cpu().numpy()
                ]
                if roi is not None:
                    detections = [
                        detection
                        for detection in detections
                        if roi.contains_bbox_center(
                            detection,
                            frame_width=frame.shape[1],
                            frame_height=frame.shape[0],
                        )
                    ]
                    if not detections:
                        continue
                for tracked in self.tracker.update(detections):
                    self._process_track(frame, tracked.track_id, np.array(tracked.bbox), line_y)
        finally:
            cap.release()
            cv2.destroyAllWindows()
        return completed

    def _process_track(self, frame: np.ndarray, track_id: int, coords: np.ndarray, line_y: int) -> None:
        x1, y1, x2, y2 = [int(value) for value in coords]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(frame.shape[1], x2)
        y2 = min(frame.shape[0], y2)
        if x2 <= x1 or y2 <= y1:
            return

        center_y = (y1 + y2) / 2
        vehicle_crop = frame[y1:y2, x1:x2].copy()
        current_area = max(0, (x2 - x1) * (y2 - y1))
        state = self._track_states.get(track_id)
        if state is None:
            state = TrackState(last_center_y=center_y)
            self._track_states[track_id] = state
        else:
            crossed_line = state.last_center_y < line_y <= center_y
            self._track_states[track_id].last_center_y = center_y

            self._update_best_candidate(state, frame, vehicle_crop, current_area)
            if state.emitted or not crossed_line:
                return

            self._emit_event(track_id, state)
            state.emitted = True
            return

        self._update_best_candidate(state, frame, vehicle_crop, current_area)

    def _update_best_candidate(
        self,
        state: TrackState,
        frame: np.ndarray,
        vehicle_crop: np.ndarray,
        area: int,
    ) -> None:
        should_run_ocr = (
            state.best_result is None
            or not state.best_result.plate_number_final
            or area >= max(int(state.last_ocr_area * 1.08), state.last_ocr_area + 2500)
        )
        if not should_run_ocr:
            return

        state.last_ocr_area = area
        ocr_result = self.recognizer.read_plate(frame, vehicle_crop)
        candidate_score = self._candidate_score(ocr_result, area)
        best_score = self._candidate_score(state.best_result, state.best_area)
        if candidate_score <= best_score:
            return

        state.best_area = area
        state.best_result = ocr_result
        state.best_frame = frame.copy()

    def _candidate_score(self, result: OCRResult | None, area: int) -> float:
        if result is None:
            return 0.0
        normalized = result.plate_number_final
        length_bonus = len(normalized) * 6
        confidence_score = result.confidence * 2
        crop_bonus = 8 if result.plate_crop is not None else 0
        area_bonus = min(area / 25000.0, 12)
        return confidence_score + length_bonus + crop_bonus + area_bonus

    def _emit_event(self, track_id: int, state: TrackState) -> None:
        entry_time = datetime.now(self._tz).replace(tzinfo=None)
        result = state.best_result or OCRResult("", "", 0.0, None)
        frame = state.best_frame
        if frame is None:
            LOGGER.warning("Skipping event for track=%s because no frame candidate was buffered.", track_id)
            return
        if self._should_use_ollama(result):
            fallback_result = self.ollama_fallback.read_plate(frame, result.plate_crop) if self.ollama_fallback else None
            if fallback_result and self._candidate_score(fallback_result, state.best_area) > self._candidate_score(result, state.best_area):
                result = fallback_result

        snapshot_path, plate_crop_path = self.storage.save_event_images(
            camera_name=self.settings.camera_name,
            entry_time=entry_time,
            frame=frame,
            plate_crop=result.plate_crop,
        )
        review_status = self.recognizer.review_status_for(result)

        repository = self.repository_factory()
        try:
            repository.create_event(
                camera_name=self.settings.camera_name,
                plate_number_raw=result.plate_number_raw,
                plate_number_final=result.plate_number_final,
                ocr_confidence=result.confidence,
                entry_time=entry_time,
                snapshot_path=snapshot_path,
                plate_crop_path=plate_crop_path,
                review_status=review_status,
            )
        finally:
            repository.session.close()
        state.emitted = True
        LOGGER.info(
            "Recorded event track=%s plate=%s confidence=%.2f status=%s",
            track_id,
            result.plate_number_final or result.plate_number_raw or "UNREADABLE",
            result.confidence,
            review_status,
        )
        self._report(
            "event_recorded",
            {
                "track_id": track_id,
                "plate": result.plate_number_final or result.plate_number_raw or "UNREADABLE",
                "confidence": result.confidence,
                "entry_time": entry_time.strftime("%Y-%m-%d %H:%M:%S"),
                "review_status": review_status,
            },
        )

    def _should_use_ollama(self, result: OCRResult) -> bool:
        if self.ollama_fallback is None:
            return False
        if result.plate_number_final and result.confidence >= self.settings.ocr_confidence_threshold:
            return False
        return True

    def _report(self, event: str, payload: dict) -> None:
        if self.status_callback is not None:
            self.status_callback(event, payload)
