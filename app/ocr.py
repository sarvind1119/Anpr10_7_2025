from __future__ import annotations

import re
from dataclasses import dataclass

import cv2
import numpy as np
import pytesseract


PLATE_TEXT_PATTERN = re.compile(r"[^A-Z0-9]")


@dataclass
class OCRResult:
    plate_number_raw: str
    plate_number_final: str
    confidence: float
    plate_crop: np.ndarray | None


def normalize_plate_text(text: str) -> str:
    return PLATE_TEXT_PATTERN.sub("", text.upper())


class PlateRecognizer:
    def __init__(self, *, tesseract_cmd: str, confidence_threshold: float) -> None:
        pytesseract.pytesseract.tesseract_cmd = tesseract_cmd
        self.confidence_threshold = confidence_threshold

    def read_plate(self, frame: np.ndarray, vehicle_crop: np.ndarray) -> OCRResult:
        candidate = self._locate_plate(vehicle_crop)
        search_regions = [candidate] if candidate is not None else []
        search_regions.extend(self._fallback_regions(vehicle_crop))

        best_text = ""
        best_confidence = 0.0
        best_crop = candidate

        for region in search_regions:
            if region is None or region.size == 0:
                continue
            processed = self._prepare_for_ocr(region)
            text, confidence = self._run_tesseract(processed)
            normalized = normalize_plate_text(text)
            if len(normalized) < 4:
                continue
            if confidence > best_confidence:
                best_text = text.strip()
                best_confidence = confidence
                best_crop = region

        normalized_text = normalize_plate_text(best_text)
        return OCRResult(
            plate_number_raw=best_text,
            plate_number_final=normalized_text,
            confidence=best_confidence,
            plate_crop=best_crop,
        )

    def review_status_for(self, result: OCRResult) -> str:
        if not result.plate_number_final:
            return "pending"
        if result.confidence < self.confidence_threshold:
            return "pending"
        return "auto_accepted"

    def _locate_plate(self, vehicle_crop: np.ndarray) -> np.ndarray | None:
        gray = cv2.cvtColor(vehicle_crop, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)

        rect_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (17, 5))
        square_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, rect_kernel)
        light = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, square_kernel)
        light = cv2.threshold(light, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

        grad_x = cv2.Sobel(blackhat, ddepth=cv2.CV_32F, dx=1, dy=0, ksize=-1)
        grad_x = np.absolute(grad_x)
        min_val = float(np.min(grad_x))
        max_val = float(np.max(grad_x))
        if max_val > min_val:
            grad_x = ((grad_x - min_val) / (max_val - min_val) * 255).astype("uint8")
        else:
            grad_x = np.zeros_like(gray)

        grad_x = cv2.morphologyEx(grad_x, cv2.MORPH_CLOSE, rect_kernel)
        grad_x = cv2.threshold(grad_x, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        grad_x = cv2.erode(grad_x, None, iterations=1)
        grad_x = cv2.dilate(grad_x, None, iterations=1)
        candidate_mask = cv2.bitwise_and(grad_x, grad_x, mask=light)

        contours, _ = cv2.findContours(candidate_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        h_total, w_total = vehicle_crop.shape[:2]
        best_region = None
        best_score = 0.0
        for contour in contours:
            x, y, w, h = cv2.boundingRect(contour)
            if h == 0 or w == 0:
                continue
            aspect_ratio = w / float(h)
            area = w * h
            if area < 1800:
                continue
            if not 2.0 <= aspect_ratio <= 7.5:
                continue

            # Favor regions in the lower-middle portion of the vehicle where the plate usually sits.
            vertical_score = (y + h / 2) / h_total
            horizontal_center = (x + w / 2) / w_total
            horizontal_score = 1.0 - abs(horizontal_center - 0.6)
            score = area * max(vertical_score, 0.2) * max(horizontal_score, 0.2)
            if score > best_score:
                pad_x = max(int(w * 0.12), 8)
                pad_y = max(int(h * 0.35), 10)
                x1 = max(0, x - pad_x)
                y1 = max(0, y - pad_y)
                x2 = min(w_total, x + w + pad_x)
                y2 = min(h_total, y + h + pad_y)
                best_region = vehicle_crop[y1:y2, x1:x2]
                best_score = score

        return best_region

    def _fallback_regions(self, vehicle_crop: np.ndarray) -> list[np.ndarray]:
        h, w = vehicle_crop.shape[:2]
        return [
            vehicle_crop[int(h * 0.48): int(h * 0.82), int(w * 0.35): int(w * 0.95)],
            vehicle_crop[int(h * 0.50):, int(w * 0.25):],
            vehicle_crop[int(h * 0.58): int(h * 0.86), int(w * 0.18): int(w * 0.92)],
            vehicle_crop[h // 2:, :],
        ]

    def _prepare_for_ocr(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.resize(gray, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
        gray = cv2.bilateralFilter(gray, 7, 25, 25)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        return gray

    def _run_tesseract(self, image: np.ndarray) -> tuple[str, float]:
        processed_variants = [
            image,
            cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1],
            cv2.threshold(image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1],
        ]
        psm_modes = (7, 8, 6, 13)

        best_text = ""
        best_confidence = 0.0
        for variant in processed_variants:
            for psm in psm_modes:
                config = f"--psm {psm} -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
                data = pytesseract.image_to_data(
                    variant,
                    config=config,
                    output_type=pytesseract.Output.DICT,
                )
                tokens = []
                confidences: list[float] = []
                for text, conf in zip(data["text"], data["conf"]):
                    token = text.strip()
                    if not token:
                        continue
                    try:
                        confidence = float(conf)
                    except ValueError:
                        continue
                    if confidence < 0:
                        continue
                    tokens.append(token)
                    confidences.append(confidence)

                if not tokens:
                    continue

                combined_text = "".join(tokens)
                normalized = normalize_plate_text(combined_text)
                if len(normalized) < 4:
                    continue
                average_conf = sum(confidences) / len(confidences)
                length_bonus = min(len(normalized), 10) * 1.5
                score = average_conf + length_bonus
                if score > best_confidence:
                    best_text = combined_text
                    best_confidence = average_conf

        return best_text, best_confidence
