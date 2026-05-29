"""
Unit Tests — FastAPI Endpoints
================================
Integration tests for the /health, /info, /predict, /predict_batch endpoints.
Uses TestClient to run without a live server.
"""

import io
import numpy as np
import pytest
from unittest.mock import patch, MagicMock


def make_dummy_jpeg(width: int = 640, height: int = 480) -> bytes:
    """Create a minimal JPEG image as bytes."""
    import cv2
    img = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    _, buf = cv2.imencode(".jpg", img)
    return buf.tobytes()


def make_mock_yolo_result(num_boxes: int = 3):
    """Create a mock Ultralytics YOLO result."""
    mock_result = MagicMock()
    if num_boxes > 0:
        mock_boxes = MagicMock()
        mock_boxes.xyxy.cpu().numpy.return_value = np.array([
            [100, 100, 200, 200],
            [300, 300, 400, 350],
            [50, 50, 80, 80],
        ][:num_boxes], dtype=np.float32)
        mock_boxes.conf.cpu().numpy.return_value = np.array(
            [0.91, 0.78, 0.65][:num_boxes], dtype=np.float32
        )
        mock_boxes.cls.cpu().numpy.return_value = np.array(
            [0, 1, 10][:num_boxes], dtype=np.float32
        )
        mock_result.boxes = mock_boxes
    else:
        mock_result.boxes = None
    return [mock_result]


@pytest.fixture
def mock_yolo_model():
    """Mock YOLO model for API tests (no GPU/weights required)."""
    with patch("src.api.main._model") as mock_model:
        mock_model.names = {
            0: "plane", 1: "ship", 9: "large-vehicle", 10: "small-vehicle"
        }
        mock_model.predict.return_value = make_mock_yolo_result(3)
        yield mock_model


@pytest.fixture
def client(mock_yolo_model):
    """FastAPI TestClient with mocked model."""
    from fastapi.testclient import TestClient
    from src.api.main import app
    with TestClient(app) as c:
        yield c


# ── Health & Info Tests ────────────────────────────────────────────────────────

class TestHealthEndpoint:

    def test_health_returns_200(self, client):
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_health_response_schema(self, client):
        resp = client.get("/health")
        data = resp.json()
        assert "status" in data
        assert "model_loaded" in data
        assert "device" in data
        assert "model_version" in data

    def test_health_status_ok(self, client):
        resp = client.get("/health")
        assert resp.json()["status"] == "ok"


class TestInfoEndpoint:

    def test_info_returns_200(self, client):
        resp = client.get("/info")
        assert resp.status_code == 200

    def test_info_has_class_names(self, client):
        resp = client.get("/info")
        data = resp.json()
        assert "class_names" in data
        assert "num_classes" in data
        assert data["num_classes"] > 0

    def test_info_has_architecture(self, client):
        resp = client.get("/info")
        data = resp.json()
        assert "architecture" in data
        assert "YOLOv8" in data["architecture"]


# ── Predict Endpoint Tests ─────────────────────────────────────────────────────

class TestPredictEndpoint:

    def test_predict_returns_200(self, client):
        img_bytes = make_dummy_jpeg()
        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_predict_response_schema(self, client):
        img_bytes = make_dummy_jpeg()
        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        data = resp.json()
        required_keys = [
            "filename", "image_width", "image_height",
            "num_detections", "detections", "inference_ms", "model_version"
        ]
        for key in required_keys:
            assert key in data, f"Missing key: {key}"

    def test_predict_returns_detections(self, client):
        img_bytes = make_dummy_jpeg()
        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        data = resp.json()
        assert data["num_detections"] == 3
        assert len(data["detections"]) == 3

    def test_predict_detection_schema(self, client):
        img_bytes = make_dummy_jpeg()
        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        det = resp.json()["detections"][0]
        assert "bbox" in det
        assert "confidence" in det
        assert "class_id" in det
        assert "class_name" in det
        bbox = det["bbox"]
        assert all(k in bbox for k in ["x1", "y1", "x2", "y2"])

    def test_predict_confidence_range(self, client):
        img_bytes = make_dummy_jpeg()
        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        for det in resp.json()["detections"]:
            assert 0.0 <= det["confidence"] <= 1.0

    def test_predict_with_conf_param(self, client):
        img_bytes = make_dummy_jpeg()
        resp = client.post(
            "/predict?conf=0.5",
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        assert resp.status_code == 200

    def test_predict_invalid_conf_param(self, client):
        img_bytes = make_dummy_jpeg()
        resp = client.post(
            "/predict?conf=1.5",  # > 1.0, invalid
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        assert resp.status_code == 422  # Unprocessable Entity

    def test_predict_empty_image_returns_zero_detections(self, client, mock_yolo_model):
        mock_yolo_model.predict.return_value = make_mock_yolo_result(0)
        img_bytes = make_dummy_jpeg()
        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        assert resp.status_code == 200
        assert resp.json()["num_detections"] == 0

    def test_predict_unsupported_file_type(self, client):
        resp = client.post(
            "/predict",
            files={"file": ("test.txt", io.BytesIO(b"not an image"), "text/plain")},
        )
        assert resp.status_code == 400

    def test_predict_returns_image_dimensions(self, client):
        img_bytes = make_dummy_jpeg(640, 480)
        resp = client.post(
            "/predict",
            files={"file": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")},
        )
        data = resp.json()
        assert data["image_width"] == 640
        assert data["image_height"] == 480


# ── Batch Predict Tests ────────────────────────────────────────────────────────

class TestPredictBatchEndpoint:

    def test_batch_predict_single_image(self, client):
        img_bytes = make_dummy_jpeg()
        resp = client.post(
            "/predict_batch",
            files=[("files", ("img1.jpg", io.BytesIO(img_bytes), "image/jpeg"))],
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_images"] == 1

    def test_batch_predict_multiple_images(self, client):
        files = [
            ("files", (f"img{i}.jpg", io.BytesIO(make_dummy_jpeg()), "image/jpeg"))
            for i in range(3)
        ]
        resp = client.post("/predict_batch", files=files)
        assert resp.status_code == 200
        data = resp.json()
        assert data["total_images"] == 3
        assert len(data["results"]) == 3

    def test_batch_predict_too_many_images(self, client):
        files = [
            ("files", (f"img{i}.jpg", io.BytesIO(make_dummy_jpeg()), "image/jpeg"))
            for i in range(21)  # > 20 limit
        ]
        resp = client.post("/predict_batch", files=files)
        assert resp.status_code == 400

    def test_batch_response_schema(self, client):
        img_bytes = make_dummy_jpeg()
        resp = client.post(
            "/predict_batch",
            files=[("files", ("img1.jpg", io.BytesIO(img_bytes), "image/jpeg"))],
        )
        data = resp.json()
        assert "total_images" in data
        assert "total_detections" in data
        assert "results" in data

    def test_batch_total_detections_sum(self, client):
        files = [
            ("files", (f"img{i}.jpg", io.BytesIO(make_dummy_jpeg()), "image/jpeg"))
            for i in range(2)
        ]
        resp = client.post("/predict_batch", files=files)
        data = resp.json()
        computed_total = sum(r["num_detections"] for r in data["results"])
        assert data["total_detections"] == computed_total
