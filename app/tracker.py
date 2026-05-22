from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrackedDetection:
    track_id: int
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]


@dataclass
class _Track:
    track_id: int
    bbox: tuple[int, int, int, int]
    center: tuple[float, float]
    missed_frames: int = 0


class CentroidTracker:
    def __init__(self, *, max_distance: float = 90.0, max_missed_frames: int = 12) -> None:
        self.max_distance = max_distance
        self.max_missed_frames = max_missed_frames
        self._next_track_id = 1
        self._tracks: dict[int, _Track] = {}

    def update(self, detections: list[tuple[int, int, int, int]]) -> list[TrackedDetection]:
        centers = [((x1 + x2) / 2.0, (y1 + y2) / 2.0) for x1, y1, x2, y2 in detections]
        unmatched_tracks = set(self._tracks.keys())
        assignments: dict[int, int] = {}
        used_detection_indexes: set[int] = set()

        pairs: list[tuple[float, int, int]] = []
        for track_id, track in self._tracks.items():
            for index, center in enumerate(centers):
                distance = ((track.center[0] - center[0]) ** 2 + (track.center[1] - center[1]) ** 2) ** 0.5
                pairs.append((distance, track_id, index))
        pairs.sort(key=lambda item: item[0])

        for distance, track_id, detection_index in pairs:
            if distance > self.max_distance:
                continue
            if track_id not in unmatched_tracks or detection_index in used_detection_indexes:
                continue
            assignments[track_id] = detection_index
            unmatched_tracks.remove(track_id)
            used_detection_indexes.add(detection_index)

        for track_id, detection_index in assignments.items():
            bbox = detections[detection_index]
            center = centers[detection_index]
            track = self._tracks[track_id]
            track.bbox = bbox
            track.center = center
            track.missed_frames = 0

        for track_id in list(unmatched_tracks):
            track = self._tracks[track_id]
            track.missed_frames += 1
            if track.missed_frames > self.max_missed_frames:
                del self._tracks[track_id]

        for index, bbox in enumerate(detections):
            if index in used_detection_indexes:
                continue
            center = centers[index]
            track_id = self._next_track_id
            self._next_track_id += 1
            self._tracks[track_id] = _Track(track_id=track_id, bbox=bbox, center=center)

        return [
            TrackedDetection(track_id=track.track_id, bbox=track.bbox, center=track.center)
            for track in self._tracks.values()
        ]
