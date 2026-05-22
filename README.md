# Academy ANPR MVP

Offline single-gate ANPR system for academy vehicle entry logging.

This codebase replaces the earlier YOLO-only prototype with a small local application that:

- reads a recorded video file or RTSP camera stream
- tracks vehicles and creates one event per line crossing
- runs offline OCR with Tesseract
- falls back to local Ollama vision OCR for weak Indian-plate reads
- stores `plate_number`, `entry_time`, and snapshot paths in SQLite
- lets you define a rectangular ROI from the dashboard so off-area vehicles are ignored
- exposes a local FastAPI dashboard for search, review, correction, and CSV export

The older prototype scripts are still present as reference material:

- `app.py`
- `app2.py`
- `vision.py`
- `vision_utils.py`

## Project layout

```text
app/
  cli.py
  config.py
  database.py
  dependencies.py
  logging_config.py
  main.py
  models.py
  ocr.py
  pipeline.py
  repository.py
  schemas.py
  static/
  storage.py
  templates/
tests/
.env.example
pyproject.toml
```

## Requirements

- Windows machine on the academy network
- Python available through the local virtual environment
- Tesseract installed at `C:\Program Files\Tesseract-OCR\tesseract.EXE`
- YOLO vehicle model file available locally, default `yolov8n.pt`

## Setup

1. Activate the virtual environment:

   ```powershell
   .\.venv\Scripts\Activate.ps1
   ```

   If your environment folder is named `venv` instead, use:

   ```powershell
   .\venv\Scripts\Activate.ps1
   ```

2. Optional: install the project in editable mode if `pip install -e .` works on your machine:

   ```powershell
   .\venv\Scripts\pip.exe install -e .
   ```

   If editable install is blocked by local Windows permissions, use the included PowerShell launchers instead.

3. Copy `.env.example` to `.env` and adjust:

   - `ANPR_SOURCE_TYPE=video` for sample testing
   - `ANPR_SOURCE_VALUE` to your sample file or RTSP URL
   - `ANPR_VEHICLE_MODEL_PATH` to the local YOLO model
   - `ANPR_ROI_CONFIG_PATH` if you want ROI settings stored somewhere else
   - `ANPR_LINE_Y_RATIO` to match your camera angle; the default `0.55` works better for the included sample footage than the earlier prototype value
   - `ANPR_ENABLE_OLLAMA_FALLBACK=true` to enable local multimodal fallback
   - `ANPR_OLLAMA_MODEL=gemma3:4b` if you want the current Indian-plate fallback path

## Run

Run the API only:

```powershell
.\anpr-api.ps1
```

Run the worker only:

```powershell
.\anpr-worker.ps1
```

Run both together for local development:

```powershell
.\anpr-dev.ps1
```

Then open `http://127.0.0.1:8010`.

You can also run the module directly:

```powershell
.\venv\Scripts\python.exe -m app.cli api
.\venv\Scripts\python.exe -m app.cli worker
.\venv\Scripts\python.exe -m app.cli dev
.\venv\Scripts\python.exe -m app.cli backfill --limit 20
```

`backfill` updates existing pending rows from saved snapshots without rerunning the whole video and creating duplicate events.

## ROI setup

- Open the dashboard and use the `Detection Region` panel.
- Click and drag on the preview frame to draw the ROI rectangle.
- Click `Save ROI` to persist it immediately.
- The live worker will ignore vehicle detections whose bounding-box center falls outside the saved ROI.
- `Clear ROI` returns the system to full-frame detection.

## Phase 1 behavior

- one gate camera only
- entry events only
- passive logging only
- offline OCR only
- low-confidence reads go to a manual review queue
- local Ollama fallback can populate reviewable Indian-plate candidates even when Tesseract is blank

## Current limitations

- plate localization is heuristic, not a dedicated plate detector model
- `yolov8n.pt` must already exist locally; the app does not download models
- live RTSP validation still depends on your actual camera details and network reachability

## Tests

Run the current unit tests with:

```powershell
.\venv\Scripts\python.exe -m unittest discover -s tests
```
