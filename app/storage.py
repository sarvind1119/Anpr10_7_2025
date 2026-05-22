from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


class SnapshotStorage:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)

    def save_event_images(
        self,
        *,
        camera_name: str,
        entry_time: datetime,
        frame: np.ndarray,
        plate_crop: np.ndarray | None,
    ) -> Tuple[str, str]:
        day_dir = self.root / entry_time.strftime("%Y-%m-%d")
        day_dir.mkdir(parents=True, exist_ok=True)
        timestamp = entry_time.strftime("%H%M%S_%f")[:-3]

        snapshot_name = f"{camera_name}_{timestamp}.jpg"
        snapshot_path = day_dir / snapshot_name
        cv2.imwrite(str(snapshot_path), frame)

        plate_rel_path = ""
        if plate_crop is not None and plate_crop.size > 0:
            plate_name = f"{camera_name}_{timestamp}_plate.jpg"
            plate_path = day_dir / plate_name
            cv2.imwrite(str(plate_path), plate_crop)
            plate_rel_path = str(plate_path.relative_to(self.root))

        return str(snapshot_path.relative_to(self.root)), plate_rel_path

