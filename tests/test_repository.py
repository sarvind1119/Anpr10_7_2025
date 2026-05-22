import unittest
from datetime import datetime
from pathlib import Path

from app.database import create_session_factory
from app.repository import EventRepository


class RepositoryTests(unittest.TestCase):
    def setUp(self):
        temp_root = Path("storage") / "test_tmp"
        temp_root.mkdir(parents=True, exist_ok=True)
        self.db_path = temp_root / "repository_test.db"
        if self.db_path.exists():
            self.db_path.unlink()
        db_path = self.db_path
        self.session_factory = create_session_factory(db_path)
        self.session = self.session_factory()
        self.repo = EventRepository(self.session)

    def tearDown(self):
        self.session.close()
        try:
            if self.db_path.exists():
                self.db_path.unlink()
        except PermissionError:
            pass

    def test_create_and_update_event(self):
        event = self.repo.create_event(
            camera_name="main_gate",
            plate_number_raw="MP09AB1234",
            plate_number_final="MP09AB1234",
            ocr_confidence=82.5,
            entry_time=datetime(2026, 1, 1, 10, 30, 0),
            snapshot_path="2026-01-01/frame.jpg",
            plate_crop_path="2026-01-01/plate.jpg",
            review_status="auto_accepted",
        )
        self.assertEqual(event.id, 1)

        updated = self.repo.update_plate_number(event.id, "UP11BW2324")
        self.assertIsNotNone(updated)
        self.assertEqual(updated.plate_number_final, "UP11BW2324")
        self.assertEqual(updated.review_status, "corrected")


if __name__ == "__main__":
    unittest.main()
