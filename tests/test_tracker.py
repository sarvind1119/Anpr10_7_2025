import unittest

from app.tracker import CentroidTracker


class CentroidTrackerTests(unittest.TestCase):
    def test_reuses_track_id_for_nearby_detection(self):
        tracker = CentroidTracker(max_distance=50, max_missed_frames=2)
        first = tracker.update([(10, 10, 30, 30)])
        second = tracker.update([(14, 12, 34, 32)])

        self.assertEqual(len(first), 1)
        self.assertEqual(len(second), 1)
        self.assertEqual(first[0].track_id, second[0].track_id)


if __name__ == "__main__":
    unittest.main()
