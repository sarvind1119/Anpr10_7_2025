from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field


class EventUpdateRequest(BaseModel):
    plate_number_final: str = Field(min_length=1, max_length=50)


class EventResponse(BaseModel):
    id: int
    camera_name: str
    plate_number_raw: str
    plate_number_final: str
    ocr_confidence: float
    entry_time: datetime
    snapshot_path: str
    plate_crop_path: str
    review_status: str
    created_at: datetime

    @classmethod
    def from_model(cls, event) -> "EventResponse":
        return cls(
            id=event.id,
            camera_name=event.camera_name,
            plate_number_raw=event.plate_number_raw,
            plate_number_final=event.plate_number_final,
            ocr_confidence=event.ocr_confidence,
            entry_time=event.entry_time,
            snapshot_path=event.snapshot_path,
            plate_crop_path=event.plate_crop_path,
            review_status=event.review_status,
            created_at=event.created_at,
        )

