from __future__ import annotations

import csv
import io
import logging
import re
import threading
from urllib.parse import quote_plus
from dataclasses import replace
from datetime import datetime
from pathlib import Path

import cv2
from fastapi import Depends, FastAPI, File, Form, Query, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, RedirectResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.config import get_settings
from app.dependencies import OLLAMA_FALLBACK, RECOGNIZER, ROI_STORE, SESSION_FACTORY, STORAGE, get_repository, get_session
from app.ocr import normalize_plate_text
from app.pipeline import IngestWorker
from app.roi import ROIBox
from app.schemas import EventResponse, EventUpdateRequest


settings = get_settings()
app = FastAPI(title="Academy ANPR")
templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))
app.mount("/assets", StaticFiles(directory=str(Path(__file__).resolve().parent / "static")), name="assets")
LOGGER = logging.getLogger(__name__)


def _camera_slug(value: str) -> str:
    slug = re.sub(r"[^a-z0-9_]+", "_", value.strip().lower())
    slug = re.sub(r"_+", "_", slug).strip("_")
    return slug or "test_video"


class TestVideoRunner:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.is_running = False
        self.status = "idle"
        self.message = "No test video has been run yet."
        self.current_video = ""
        self.last_started_at = ""
        self.last_finished_at = ""

    def snapshot(self) -> dict[str, str | bool]:
        with self.lock:
            return {
                "is_running": self.is_running,
                "status": self.status,
                "message": self.message,
                "current_video": self.current_video,
                "last_started_at": self.last_started_at,
                "last_finished_at": self.last_finished_at,
            }

    def start(self, *, video_path: Path, camera_name: str) -> bool:
        with self.lock:
            if self.is_running:
                self.status = "running"
                self.message = "A test video run is already in progress."
                return False
            self.is_running = True
            self.status = "running"
            self.message = f"Processing {video_path.name}"
            self.current_video = video_path.name
            self.last_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.last_finished_at = ""

        thread = threading.Thread(
            target=self._run_video,
            kwargs={"video_path": video_path, "camera_name": camera_name},
            daemon=True,
        )
        thread.start()
        return True

    def _run_video(self, *, video_path: Path, camera_name: str) -> None:
        try:
            worker_settings = replace(
                settings,
                source_type="video",
                source_value=str(video_path),
                camera_name=camera_name,
            )
            worker = IngestWorker(
                settings=worker_settings,
                repository_factory=lambda: get_repository(SESSION_FACTORY()),
                storage=STORAGE,
                recognizer=RECOGNIZER,
                ollama_fallback=OLLAMA_FALLBACK,
                roi_provider=ROI_STORE.get,
            )
            completed = worker.run_once()
            with self.lock:
                self.status = "completed" if completed else "failed"
                self.message = (
                    f"Finished processing {video_path.name}"
                    if completed
                    else f"Unable to process {video_path.name}"
                )
        except Exception as exc:
            LOGGER.exception("Test video processing failed.")
            with self.lock:
                self.status = "failed"
                self.message = f"Failed: {exc}"
        finally:
            with self.lock:
                self.is_running = False
                self.last_finished_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")


TEST_VIDEO_RUNNER = TestVideoRunner()


class LiveFeedRunner:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.thread: threading.Thread | None = None
        self.worker: IngestWorker | None = None
        self.is_running = False
        self.status = "idle"
        self.message = "Live feed is not running."
        self.last_started_at = ""
        self.last_stopped_at = ""
        self.last_frame_at = ""
        self.last_event_at = ""
        self.events_recorded = 0
        self.frames_seen = 0
        self.last_plate = ""
        self.last_error = ""

    def snapshot(self) -> dict[str, str | bool]:
        with self.lock:
            return {
                "is_running": self.is_running,
                "status": self.status,
                "message": self.message,
                "last_started_at": self.last_started_at,
                "last_stopped_at": self.last_stopped_at,
                "last_frame_at": self.last_frame_at,
                "last_event_at": self.last_event_at,
                "events_recorded": self.events_recorded,
                "frames_seen": self.frames_seen,
                "last_plate": self.last_plate,
                "last_error": self.last_error,
                "camera_name": settings.camera_name,
                "source_type": settings.source_type,
                "source_value_masked": _mask_source_value(settings.source_value),
            }

    def start(self) -> bool:
        with self.lock:
            if self.is_running:
                self.status = "running"
                self.message = "Live feed is already running."
                return False
            if settings.source_type != "rtsp":
                self.status = "failed"
                self.message = "Live feed start is only available when ANPR_SOURCE_TYPE=rtsp."
                return False
            self.is_running = True
            self.status = "running"
            self.message = "Connecting to live RTSP feed..."
            self.last_started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            self.last_stopped_at = ""
            self.last_frame_at = ""
            self.last_event_at = ""
            self.events_recorded = 0
            self.frames_seen = 0
            self.last_plate = ""
            self.last_error = ""

        worker = IngestWorker(
            settings=settings,
            repository_factory=lambda: get_repository(SESSION_FACTORY()),
            storage=STORAGE,
            recognizer=RECOGNIZER,
            ollama_fallback=OLLAMA_FALLBACK,
            status_callback=self._handle_worker_event,
            roi_provider=ROI_STORE.get,
        )
        thread = threading.Thread(target=self._run_worker, args=(worker,), daemon=True)
        with self.lock:
            self.worker = worker
            self.thread = thread
        thread.start()
        return True

    def stop(self) -> bool:
        with self.lock:
            if not self.is_running or self.worker is None:
                self.status = "idle"
                self.message = "Live feed is not running."
                return False
            self.status = "stopping"
            self.message = "Stopping live feed..."
            worker = self.worker
        worker.stop()
        return True

    def _run_worker(self, worker: IngestWorker) -> None:
        try:
            worker.run_forever()
            with self.lock:
                if self.status == "stopping":
                    self.status = "idle"
                    self.message = "Live feed stopped."
                else:
                    self.status = "completed"
                    self.message = "Live feed worker exited."
        except Exception as exc:
            LOGGER.exception("Live feed worker failed.")
            with self.lock:
                self.status = "failed"
                self.message = f"Live feed failed: {exc}"
        finally:
            with self.lock:
                self.is_running = False
                self.last_stopped_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                self.worker = None
                self.thread = None

    def _handle_worker_event(self, event: str, payload: dict) -> None:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self.lock:
            if event == "connected":
                self.status = "running"
                self.message = "Connected to live feed. Waiting for vehicle events..."
                return
            if event == "heartbeat":
                self.frames_seen = int(payload.get("frame_count", self.frames_seen))
                self.last_frame_at = now
                self.status = "running"
                self.message = "Receiving frames from live feed."
                return
            if event == "event_recorded":
                self.events_recorded += 1
                self.last_event_at = payload.get("entry_time", now)
                self.last_plate = str(payload.get("plate", ""))
                self.status = "running"
                self.message = "Live feed is running and detections are being recorded."
                return
            if event == "source_error":
                self.last_error = str(payload.get("message", "Source error"))
                self.message = self.last_error
                self.status = "running"
                return
            if event == "reconnect_wait":
                seconds = payload.get("seconds", settings.reconnect_seconds)
                self.message = f"Live feed disconnected. Reconnecting in {seconds} seconds..."
                self.status = "running"


LIVE_FEED_RUNNER = LiveFeedRunner()


def _video_choices() -> list[Path]:
    roots = [
        Path.cwd(),
        settings.db_path.parent / "uploads",
        Path(settings.source_value).parent if settings.source_type == "video" else settings.db_path.parent,
    ]
    seen: set[Path] = set()
    videos: list[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for pattern in ("*.mp4", "*.avi", "*.mov", "*.mkv"):
            for video in sorted(root.glob(pattern)):
                resolved = video.resolve()
                if resolved in seen:
                    continue
                seen.add(resolved)
                videos.append(resolved)
    return videos


def _mask_source_value(value: str) -> str:
    if "@" not in value:
        return value
    prefix, suffix = value.split("@", 1)
    if "://" not in prefix:
        return f"***@{suffix}"
    scheme, credentials = prefix.split("://", 1)
    username = credentials.split(":", 1)[0]
    return f"{scheme}://{username}:****@{suffix}"


def _capture_preview_frame(source_value: str) -> bytes | None:
    cap = cv2.VideoCapture(source_value)
    if not cap.isOpened():
        return None

    frame = None
    try:
        for _ in range(20):
            ok, candidate = cap.read()
            if ok and candidate is not None and candidate.size > 0:
                frame = candidate
                break
    finally:
        cap.release()

    if frame is None:
        return None

    ok, encoded = cv2.imencode(".jpg", frame)
    if not ok:
        return None
    return encoded.tobytes()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@app.get("/", response_class=HTMLResponse)
def dashboard(
    request: Request,
    q: str = Query(default=""),
    review_status: str = Query(default=""),
    event_date: str = Query(default=""),
    roi_message: str = Query(default=""),
    roi_error: str = Query(default=""),
    session: Session = Depends(get_session),
):
    repo = get_repository(session)
    events = repo.list_events(
        search=q,
        review_status=review_status,
        event_date=event_date,
        limit=settings.max_events,
    )
    return templates.TemplateResponse(
        request,
        "dashboard.html",
        {
            "events": events,
            "request": request,
            "query": q,
            "review_status": review_status,
            "event_date": event_date,
            "runner": TEST_VIDEO_RUNNER.snapshot(),
            "live_runner": LIVE_FEED_RUNNER.snapshot(),
            "video_choices": _video_choices(),
            "roi": ROI_STORE.snapshot(),
            "roi_message": roi_message,
            "roi_error": roi_error,
        },
    )


@app.get("/events")
def list_events(
    q: str = Query(default=""),
    review_status: str = Query(default=""),
    event_date: str = Query(default=""),
    session: Session = Depends(get_session),
):
    repo = get_repository(session)
    events = repo.list_events(
        search=q,
        review_status=review_status,
        event_date=event_date,
        limit=settings.max_events,
    )
    return [EventResponse.from_model(event) for event in events]


@app.get("/events/{event_id}")
def get_event(event_id: int, session: Session = Depends(get_session)):
    repo = get_repository(session)
    event = repo.get_event(event_id)
    if event is None:
        return JSONResponse(status_code=404, content={"detail": "Event not found"})
    return EventResponse.from_model(event)


@app.patch("/events/{event_id}")
def update_event(
    event_id: int,
    payload: EventUpdateRequest,
    session: Session = Depends(get_session),
):
    repo = get_repository(session)
    event = repo.update_plate_number(event_id, normalize_plate_text(payload.plate_number_final))
    if event is None:
        return JSONResponse(status_code=404, content={"detail": "Event not found"})
    return EventResponse.from_model(event)


@app.post("/events/{event_id}/correct")
def correct_event(
    event_id: int,
    plate_number_final: str = Form(...),
    session: Session = Depends(get_session),
):
    repo = get_repository(session)
    event = repo.update_plate_number(event_id, normalize_plate_text(plate_number_final))
    if event is None:
        return JSONResponse(status_code=404, content={"detail": "Event not found"})
    return RedirectResponse(url="/", status_code=303)


@app.post("/test-video/run")
async def run_test_video(
    selected_video: str = Form(default=""),
    camera_name: str = Form(default="test_video"),
    upload_video: UploadFile | None = File(default=None),
):
    if LIVE_FEED_RUNNER.snapshot()["is_running"]:
        TEST_VIDEO_RUNNER.status = "failed"
        TEST_VIDEO_RUNNER.message = "Stop the live feed before running a test video."
        return RedirectResponse(url="/", status_code=303)

    upload_dir = settings.db_path.parent / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)

    video_path: Path | None = None
    if upload_video is not None and upload_video.filename:
        safe_name = Path(upload_video.filename).name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        video_path = upload_dir / f"{timestamp}_{safe_name}"
        with video_path.open("wb") as handle:
            while chunk := await upload_video.read(1024 * 1024):
                handle.write(chunk)
    elif selected_video:
        candidate = Path(selected_video).resolve()
        if candidate.exists():
            video_path = candidate

    if video_path is None:
        TEST_VIDEO_RUNNER.status = "failed"
        TEST_VIDEO_RUNNER.message = "Select an existing video or upload a new one first."
        return RedirectResponse(url="/", status_code=303)

    TEST_VIDEO_RUNNER.start(video_path=video_path, camera_name=_camera_slug(camera_name))
    return RedirectResponse(url="/", status_code=303)


@app.post("/live-feed/start")
def start_live_feed():
    if TEST_VIDEO_RUNNER.snapshot()["is_running"]:
        return RedirectResponse(url="/", status_code=303)
    LIVE_FEED_RUNNER.start()
    return RedirectResponse(url="/", status_code=303)


@app.post("/live-feed/stop")
def stop_live_feed():
    LIVE_FEED_RUNNER.stop()
    return RedirectResponse(url="/", status_code=303)


@app.get("/roi/preview.jpg")
def roi_preview_image():
    image_bytes = _capture_preview_frame(settings.source_value)
    if image_bytes is None:
        return JSONResponse(status_code=503, content={"detail": "Unable to capture preview frame from source"})
    return Response(
        content=image_bytes,
        media_type="image/jpeg",
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.post("/roi/update")
def update_roi(
    x_min: float = Form(...),
    y_min: float = Form(...),
    x_max: float = Form(...),
    y_max: float = Form(...),
):
    try:
        ROI_STORE.set(
            ROIBox.normalized(
                x_min=x_min,
                y_min=y_min,
                x_max=x_max,
                y_max=y_max,
            )
        )
    except ValueError as exc:
        message = quote_plus(str(exc))
        return RedirectResponse(url=f"/?roi_error={message}", status_code=303)
    return RedirectResponse(url=f"/?roi_message={quote_plus('ROI saved successfully.')}", status_code=303)


@app.post("/roi/clear")
def clear_roi():
    ROI_STORE.clear()
    message = quote_plus("ROI cleared. Full frame detection is active.")
    return RedirectResponse(url=f"/?roi_message={message}", status_code=303)


@app.get("/events/export.csv")
def export_events_csv(
    q: str = Query(default=""),
    review_status: str = Query(default=""),
    event_date: str = Query(default=""),
    session: Session = Depends(get_session),
):
    repo = get_repository(session)
    events = repo.list_events(
        search=q,
        review_status=review_status,
        event_date=event_date,
        limit=settings.max_events,
    )
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(
        [
            "id",
            "camera_name",
            "plate_number_raw",
            "plate_number_final",
            "ocr_confidence",
            "entry_time",
            "review_status",
            "snapshot_path",
            "plate_crop_path",
        ]
    )
    for event in events:
        writer.writerow(
            [
                event.id,
                event.camera_name,
                event.plate_number_raw,
                event.plate_number_final,
                event.ocr_confidence,
                event.entry_time.isoformat(sep=" "),
                event.review_status,
                event.snapshot_path,
                event.plate_crop_path,
            ]
        )
    return StreamingResponse(
        iter([buffer.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=vehicle_events.csv"},
    )


@app.get("/files/{relative_path:path}")
def serve_snapshot(relative_path: str):
    candidate = (STORAGE.root / relative_path).resolve()
    if not str(candidate).startswith(str(STORAGE.root.resolve())):
        return JSONResponse(status_code=400, content={"detail": "Invalid file path"})
    if not candidate.exists():
        return JSONResponse(status_code=404, content={"detail": "File not found"})
    return FileResponse(candidate)


@app.exception_handler(Exception)
async def unhandled_error_handler(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": str(exc)})
