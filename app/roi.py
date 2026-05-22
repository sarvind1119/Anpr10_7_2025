from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path


LOGGER = logging.getLogger(__name__)


def _clamp(value: float, minimum: float = 0.0, maximum: float = 1.0) -> float:
    return max(minimum, min(maximum, value))


@dataclass(frozen=True)
class ROIBox:
    x_min: float
    y_min: float
    x_max: float
    y_max: float

    @classmethod
    def normalized(cls, *, x_min: float, y_min: float, x_max: float, y_max: float) -> "ROIBox":
        left = _clamp(min(x_min, x_max))
        right = _clamp(max(x_min, x_max))
        top = _clamp(min(y_min, y_max))
        bottom = _clamp(max(y_min, y_max))
        if right - left < 0.01 or bottom - top < 0.01:
            raise ValueError("ROI must have non-zero width and height.")
        return cls(x_min=left, y_min=top, x_max=right, y_max=bottom)

    def contains_bbox_center(self, bbox: tuple[int, int, int, int], *, frame_width: int, frame_height: int) -> bool:
        x1, y1, x2, y2 = bbox
        center_x = (x1 + x2) / 2.0
        center_y = (y1 + y2) / 2.0
        left = self.x_min * frame_width
        right = self.x_max * frame_width
        top = self.y_min * frame_height
        bottom = self.y_max * frame_height
        return left <= center_x <= right and top <= center_y <= bottom

    def to_dict(self) -> dict[str, float]:
        return {
            "x_min": round(self.x_min, 6),
            "y_min": round(self.y_min, 6),
            "x_max": round(self.x_max, 6),
            "y_max": round(self.y_max, 6),
        }


class ROIStore:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._roi: ROIBox | None = None
        self._mtime_ns: int | None = None
        self._updated_at: str = ""
        self._reload_if_changed()

    def get(self) -> ROIBox | None:
        with self._lock:
            self._reload_if_changed()
            return self._roi

    def set(self, roi: ROIBox) -> ROIBox:
        with self._lock:
            updated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            payload = {"enabled": True, "updated_at": updated_at, **roi.to_dict()}
            self.path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
            self._mtime_ns = self.path.stat().st_mtime_ns
            self._roi = roi
            self._updated_at = updated_at
            return roi

    def clear(self) -> None:
        with self._lock:
            if self.path.exists():
                self.path.unlink()
            self._mtime_ns = None
            self._roi = None
            self._updated_at = ""

    def snapshot(self) -> dict[str, float | bool]:
        roi = self.get()
        if roi is None:
            return {
                "enabled": False,
                "x_min": 0.0,
                "y_min": 0.0,
                "x_max": 1.0,
                "y_max": 1.0,
                "updated_at": "",
            }
        return {"enabled": True, "updated_at": self._updated_at, **roi.to_dict()}

    def _reload_if_changed(self) -> None:
        if not self.path.exists():
            self._roi = None
            self._mtime_ns = None
            return

        current_mtime = self.path.stat().st_mtime_ns
        if self._mtime_ns == current_mtime:
            return

        try:
            payload = json.loads(self.path.read_text(encoding="utf-8"))
            if not payload.get("enabled", True):
                self._roi = None
                self._updated_at = ""
            else:
                self._roi = ROIBox.normalized(
                    x_min=float(payload["x_min"]),
                    y_min=float(payload["y_min"]),
                    x_max=float(payload["x_max"]),
                    y_max=float(payload["y_max"]),
                )
                self._updated_at = str(payload.get("updated_at", ""))
            self._mtime_ns = current_mtime
        except Exception:
            LOGGER.exception("Unable to load ROI configuration from %s", self.path)
