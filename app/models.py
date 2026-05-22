from __future__ import annotations

from datetime import datetime

from sqlalchemy import DateTime, Float, Integer, String
from sqlalchemy.orm import Mapped, mapped_column

from app.database import Base


class VehicleEvent(Base):
    __tablename__ = "vehicle_events"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    camera_name: Mapped[str] = mapped_column(String(100), nullable=False, index=True)
    plate_number_raw: Mapped[str] = mapped_column(String(50), default="", nullable=False)
    plate_number_final: Mapped[str] = mapped_column(String(50), default="", nullable=False, index=True)
    ocr_confidence: Mapped[float] = mapped_column(Float, default=0.0, nullable=False)
    entry_time: Mapped[datetime] = mapped_column(DateTime(timezone=False), nullable=False, index=True)
    snapshot_path: Mapped[str] = mapped_column(String(255), nullable=False)
    plate_crop_path: Mapped[str] = mapped_column(String(255), default="", nullable=False)
    review_status: Mapped[str] = mapped_column(String(20), default="pending", nullable=False, index=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=False), default=datetime.now, nullable=False)
