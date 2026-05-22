from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from app.roi import ROIBox, ROIStore


class ROIBoxTests(unittest.TestCase):
    def test_normalized_sorts_and_clamps_coordinates(self) -> None:
        roi = ROIBox.normalized(x_min=0.8, y_min=-0.2, x_max=0.2, y_max=1.2)
        self.assertEqual(roi.to_dict(), {"x_min": 0.2, "y_min": 0.0, "x_max": 0.8, "y_max": 1.0})

    def test_contains_bbox_center(self) -> None:
        roi = ROIBox.normalized(x_min=0.25, y_min=0.25, x_max=0.75, y_max=0.75)
        self.assertTrue(roi.contains_bbox_center((40, 40, 80, 80), frame_width=100, frame_height=100))
        self.assertFalse(roi.contains_bbox_center((0, 0, 20, 20), frame_width=100, frame_height=100))


class ROIStoreTests(unittest.TestCase):
    def test_store_persists_and_reloads_roi(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ROIStore(Path(temp_dir) / "roi.json")
            self.assertFalse(store.snapshot()["enabled"])

            store.set(ROIBox.normalized(x_min=0.1, y_min=0.2, x_max=0.8, y_max=0.9))
            snapshot = store.snapshot()
            self.assertTrue(snapshot["enabled"])
            self.assertEqual(snapshot["x_min"], 0.1)
            self.assertEqual(snapshot["y_max"], 0.9)

            reloaded = ROIStore(Path(temp_dir) / "roi.json")
            self.assertTrue(reloaded.snapshot()["enabled"])
            self.assertEqual(reloaded.get().to_dict()["x_max"], 0.8)

    def test_clear_disables_roi(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = ROIStore(Path(temp_dir) / "roi.json")
            store.set(ROIBox.normalized(x_min=0.2, y_min=0.2, x_max=0.7, y_max=0.7))
            store.clear()
            self.assertFalse(store.snapshot()["enabled"])


if __name__ == "__main__":
    unittest.main()
