from __future__ import annotations

import base64
import io
import logging
import re
from dataclasses import dataclass

import cv2
import numpy as np
import requests
from PIL import Image

from app.ocr import OCRResult, normalize_plate_text


LOGGER = logging.getLogger(__name__)

INDIAN_STATE_CODES = {
    "AN", "AP", "AR", "AS", "BH", "BR", "CG", "CH", "DD", "DL", "DN", "GA", "GJ",
    "HP", "HR", "JH", "JK", "KA", "KL", "LA", "LD", "MH", "ML", "MN", "MP", "MZ",
    "NL", "OD", "PB", "PY", "RJ", "SK", "TN", "TR", "TS", "UK", "UP", "WB",
}
INDIAN_PLATE_PATTERN = re.compile(r"^[A-Z]{2}[0-9]{1,2}[A-Z]{1,3}[0-9]{3,4}$")


def score_indian_plate(candidate: str) -> float:
    if not candidate:
        return 0.0
    score = 0.0
    if len(candidate) in {9, 10}:
        score += 10.0
    elif len(candidate) in {8, 11}:
        score += 4.0
    if candidate[:2] in INDIAN_STATE_CODES:
        score += 20.0
    if INDIAN_PLATE_PATTERN.match(candidate):
        score += 30.0
    return score


def select_best_indian_candidate(*values: str) -> str:
    candidates = [normalize_plate_text(value) for value in values if value]
    if not candidates:
        return ""
    scored = sorted(
        ((score_indian_plate(value), value) for value in candidates),
        key=lambda item: (item[0], len(item[1])),
        reverse=True,
    )
    best_score, best_value = scored[0]
    return best_value if best_score > 0 else candidates[0]


@dataclass
class OllamaPlateFallback:
    base_url: str
    model: str
    timeout_seconds: int = 60

    def read_plate(self, frame: np.ndarray, vehicle_crop: np.ndarray | None = None) -> OCRResult:
        vehicle_guess = self._query_model(vehicle_crop) if vehicle_crop is not None and vehicle_crop.size > 0 else ""
        frame_guess = self._query_model(frame)
        best = select_best_indian_candidate(frame_guess, vehicle_guess)
        if not best or best == "NOTVISIBLE":
            return OCRResult("", "", 0.0, vehicle_crop)

        confidence = min(55.0, 25.0 + score_indian_plate(best))
        return OCRResult(best, best, confidence, vehicle_crop)

    def _query_model(self, image: np.ndarray) -> str:
        try:
            encoded_image = self._encode_image(image)
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "This is an image of an Indian vehicle. Extract only the registration number "
                            "from the number plate. Reply with only the registration number, with no spaces "
                            "or hyphens. If unreadable, reply NOT_VISIBLE."
                        ),
                        "images": [encoded_image],
                    }
                ],
                "stream": False,
            }
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
            content = response.json().get("message", {}).get("content", "")
            return normalize_plate_text(content)
        except Exception as exc:
            LOGGER.warning("Ollama fallback failed: %s", exc)
            return ""

    def _encode_image(self, image: np.ndarray) -> str:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb)
        pil_image.thumbnail((768, 768))
        buffer = io.BytesIO()
        pil_image.save(buffer, format="JPEG", quality=90)
        return base64.b64encode(buffer.getvalue()).decode("utf-8")
