"""
Unit Tests — Inference Pipeline
=================================
Tests for SAHI predictor, batch predictor, and WBF merging.
"""

import numpy as np


class TestWBF:
    """Weighted Box Fusion unit tests."""

    def test_wbf_single_box(self):
        from src.inference.sahi_predictor import weighted_box_fusion
        boxes = [np.array([[0.1, 0.1, 0.5, 0.5]])]
        scores = [np.array([0.9])]
        labels = [np.array([0])]
        fb, fs, fl = weighted_box_fusion(boxes, scores, labels, iou_thr=0.5)
        assert len(fb) == 1
        assert abs(fs[0] - 0.9) < 1e-4

    def test_wbf_merges_overlapping_boxes(self):
        from src.inference.sahi_predictor import weighted_box_fusion
        # Two nearly identical boxes → should merge into one
        boxes = [
            np.array([[0.1, 0.1, 0.5, 0.5]]),
            np.array([[0.11, 0.11, 0.49, 0.49]]),
        ]
        scores = [np.array([0.9]), np.array([0.85])]
        labels = [np.array([0]), np.array([0])]
        fb, fs, fl = weighted_box_fusion(boxes, scores, labels, iou_thr=0.5)
        assert len(fb) == 1

    def test_wbf_keeps_different_classes_separate(self):
        from src.inference.sahi_predictor import weighted_box_fusion
        # Same location but different classes → should NOT merge
        boxes = [
            np.array([[0.1, 0.1, 0.5, 0.5]]),
            np.array([[0.1, 0.1, 0.5, 0.5]]),
        ]
        scores = [np.array([0.9]), np.array([0.85])]
        labels = [np.array([0]), np.array([1])]  # class 0 and class 1
        fb, fs, fl = weighted_box_fusion(boxes, scores, labels, iou_thr=0.5)
        assert len(fb) == 2

    def test_wbf_empty_input(self):
        from src.inference.sahi_predictor import weighted_box_fusion
        fb, fs, fl = weighted_box_fusion([], [], [], iou_thr=0.5)
        assert len(fb) == 0

    def test_wbf_filters_low_confidence(self):
        from src.inference.sahi_predictor import weighted_box_fusion
        boxes = [np.array([[0.1, 0.1, 0.5, 0.5]])]
        scores = [np.array([0.005])]   # Below skip_box_thr=0.01
        labels = [np.array([0])]
        fb, fs, fl = weighted_box_fusion(boxes, scores, labels, skip_box_thr=0.01)
        assert len(fb) == 0

    def test_compute_iou_perfect_overlap(self):
        from src.inference.sahi_predictor import compute_iou_batch
        box = np.array([0.1, 0.1, 0.5, 0.5])
        boxes = np.array([[0.1, 0.1, 0.5, 0.5]])
        ious = compute_iou_batch(box, boxes)
        assert abs(ious[0] - 1.0) < 1e-5

    def test_compute_iou_no_overlap(self):
        from src.inference.sahi_predictor import compute_iou_batch
        box = np.array([0.0, 0.0, 0.2, 0.2])
        boxes = np.array([[0.8, 0.8, 1.0, 1.0]])
        ious = compute_iou_batch(box, boxes)
        assert ious[0] == 0.0

    def test_compute_iou_partial_overlap(self):
        from src.inference.sahi_predictor import compute_iou_batch
        # Two squares sharing half their area
        box = np.array([0.0, 0.0, 0.4, 0.4])
        boxes = np.array([[0.2, 0.0, 0.6, 0.4]])
        ious = compute_iou_batch(box, boxes)
        # Intersection = 0.2×0.4=0.08, Union = 0.16+0.16-0.08=0.24
        expected_iou = 0.08 / 0.24
        assert abs(ious[0] - expected_iou) < 1e-4


class TestTileCoordGeneration:
    """Test SAHI tile coordinate generation."""

    def test_tiles_cover_full_image(self):
        from src.inference.sahi_predictor import SAHIPredictor
        # Instantiate with a fake model path to test geometry only
        # We only test the _generate_slices method which doesn't need the model
        class FakePredictor:
            tile_size = 640
            overlap = 0.2
            _generate_slices = SAHIPredictor._generate_slices

        fp = FakePredictor()
        slices = fp._generate_slices(fp, img_h=1280, img_w=1280)

        # All pixels should be within at least one tile
        covered = np.zeros((1280, 1280), dtype=bool)
        for (x1, y1, x2, y2) in slices:
            covered[y1:y2, x1:x2] = True
        assert covered.all()

    def test_single_tile_for_small_image(self):
        from src.inference.sahi_predictor import SAHIPredictor

        class FakePredictor:
            tile_size = 640
            overlap = 0.2
            _generate_slices = SAHIPredictor._generate_slices

        fp = FakePredictor()
        slices = fp._generate_slices(fp, img_h=320, img_w=320)
        assert len(slices) == 1
        assert slices[0] == (0, 0, 320, 320)


class TestGeoUtils:
    """Geospatial utility tests."""

    def test_pixel_to_latlon_top_left(self):
        from src.utils.geo_utils import pixel_to_latlon
        lon, lat = pixel_to_latlon(
            px=0, py=0,
            img_width=1000, img_height=1000,
            top_left_lon=10.0, top_left_lat=50.0,
            bottom_right_lon=11.0, bottom_right_lat=49.0,
        )
        assert abs(lon - 10.0) < 1e-6
        assert abs(lat - 50.0) < 1e-6

    def test_pixel_to_latlon_bottom_right(self):
        from src.utils.geo_utils import pixel_to_latlon
        lon, lat = pixel_to_latlon(
            px=1000, py=1000,
            img_width=1000, img_height=1000,
            top_left_lon=10.0, top_left_lat=50.0,
            bottom_right_lon=11.0, bottom_right_lat=49.0,
        )
        assert abs(lon - 11.0) < 1e-6
        assert abs(lat - 49.0) < 1e-6

    def test_haversine_zero_distance(self):
        from src.utils.geo_utils import haversine_distance
        d = haversine_distance(10.0, 50.0, 10.0, 50.0)
        assert d < 1e-3  # Should be ~0 meters

    def test_haversine_known_distance(self):
        from src.utils.geo_utils import haversine_distance
        # Paris to London ~ 340 km
        d = haversine_distance(2.3522, 48.8566, -0.1278, 51.5074)
        assert 330_000 < d < 360_000

    def test_compute_gsd(self):
        from src.utils.geo_utils import compute_gsd
        # Approximation: 700km altitude, 70mm focal length, 10mm sensor width
        # Expected GSD ~10m
        gsd = compute_gsd(
            altitude_m=700_000,
            focal_length_mm=70,
            sensor_width_mm=10,
            image_width_px=10_000,
        )
        assert 9 < gsd < 11

    def test_pixel_bbox_to_realworld(self):
        from src.utils.geo_utils import pixel_bbox_to_realworld
        result = pixel_bbox_to_realworld(0, 0, 20, 10, gsd_m_per_px=0.5)
        assert result["width_m"] == 10.0   # 20px * 0.5m
        assert result["height_m"] == 5.0   # 10px * 0.5m
        assert result["area_m2"] == 50.0
        assert result["length_m"] == 10.0

    def test_classify_ship_size(self):
        from src.utils.geo_utils import classify_ship_size
        assert classify_ship_size(10) == "small-craft"
        assert classify_ship_size(250) == "very-large-vessel"
        assert classify_ship_size(350) == "ultra-large-vessel"

    def test_detection_density_shape(self):
        from src.utils.geo_utils import compute_detection_density
        detections = [
            {"x1": 100, "y1": 100, "x2": 200, "y2": 200},
            {"x1": 800, "y1": 800, "x2": 900, "y2": 900},
        ]
        density = compute_detection_density(detections, 1000, 1000, grid_size=10)
        assert density.shape == (10, 10)
        assert density.max() <= 1.0
        assert density.min() >= 0.0
