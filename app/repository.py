from __future__ import annotations

from datetime import datetime
from typing import Iterable

from sqlalchemy import and_, desc, func, or_, select
from sqlalchemy.orm import Session

from app.models import VehicleEvent


class EventRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_event(
        self,
        *,
        camera_name: str,
        plate_number_raw: str,
        plate_number_final: str,
        ocr_confidence: float,
        entry_time: datetime,
        snapshot_path: str,
        plate_crop_path: str,
        review_status: str,
    ) -> VehicleEvent:
        event = VehicleEvent(
            camera_name=camera_name,
            plate_number_raw=plate_number_raw,
            plate_number_final=plate_number_final,
            ocr_confidence=ocr_confidence,
            entry_time=entry_time,
            snapshot_path=snapshot_path,
            plate_crop_path=plate_crop_path,
            review_status=review_status,
        )
        self.session.add(event)
        self.session.commit()
        self.session.refresh(event)
        return event

    def get_event(self, event_id: int) -> VehicleEvent | None:
        return self.session.get(VehicleEvent, event_id)

    def list_events(
        self,
        *,
        search: str = "",
        review_status: str = "",
        event_date: str = "",
        limit: int = 100,
    ) -> Iterable[VehicleEvent]:
        stmt = select(VehicleEvent)
        filters = []
        normalized_search = search.strip().upper()
        if normalized_search:
            like_value = f"%{normalized_search}%"
            filters.append(
                or_(
                    func.upper(VehicleEvent.plate_number_final).like(like_value),
                    func.upper(VehicleEvent.plate_number_raw).like(like_value),
                )
            )
        if review_status:
            filters.append(VehicleEvent.review_status == review_status)
        if event_date:
            try:
                start = datetime.fromisoformat(event_date)
            except ValueError:
                start = None
            if start is not None:
                end = start.replace(hour=23, minute=59, second=59, microsecond=999999)
                filters.append(
                    and_(VehicleEvent.entry_time >= start, VehicleEvent.entry_time <= end)
                )
        if filters:
            stmt = stmt.where(*filters)
        stmt = stmt.order_by(desc(VehicleEvent.entry_time)).limit(limit)
        return self.session.scalars(stmt).all()

    def update_plate_number(self, event_id: int, plate_number_final: str) -> VehicleEvent | None:
        event = self.get_event(event_id)
        if event is None:
            return None
        event.plate_number_final = plate_number_final
        event.review_status = "corrected"
        self.session.commit()
        self.session.refresh(event)
        return event

    def update_ocr_result(
        self,
        event_id: int,
        *,
        plate_number_raw: str,
        plate_number_final: str,
        ocr_confidence: float,
        plate_crop_path: str,
        review_status: str,
    ) -> VehicleEvent | None:
        event = self.get_event(event_id)
        if event is None:
            return None
        event.plate_number_raw = plate_number_raw
        event.plate_number_final = plate_number_final
        event.ocr_confidence = ocr_confidence
        event.plate_crop_path = plate_crop_path
        event.review_status = review_status
        self.session.commit()
        self.session.refresh(event)
        return event

    def list_pending_without_plate(self, limit: int = 100) -> Iterable[VehicleEvent]:
        stmt = (
            select(VehicleEvent)
            .where(
                VehicleEvent.review_status == "pending",
                VehicleEvent.plate_number_final == "",
            )
            .order_by(VehicleEvent.entry_time.asc())
            .limit(limit)
        )
        return self.session.scalars(stmt).all()
