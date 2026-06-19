"""
SatDet Interactive Demo — Streamlit Application
=================================================
Portfolio-grade interactive demonstration of the satellite detection system.

Run with:
    streamlit run src/visualization/streamlit_app.py
"""

from collections import Counter
import io
import sys
import time
from pathlib import Path
from urllib.parse import urlparse

ROOT_DIR = Path(__file__).resolve().parents[2]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import cv2
import httpx
import numpy as np
import streamlit as st
from PIL import Image, UnidentifiedImageError

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SatDet — Satellite Object Detection",
    page_icon="🛰️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #1a1a2e 0%, #16213e 100%);
        border: 1px solid #0f3460;
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        color: white;
    }
    .metric-card h2 { color: #e94560; margin: 0; font-size: 2rem; }
    .metric-card p  { color: #a8b2d8; margin: 4px 0 0; font-size: 0.85rem; }
    .stButton > button {
        background: linear-gradient(135deg, #e94560, #0f3460);
        color: white; border: none; width: 100%;
        font-weight: bold; padding: 12px;
    }
    .detection-badge {
        display: inline-block; background: #e94560;
        color: white; border-radius: 4px; padding: 2px 8px;
        margin: 2px; font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/satellite.png", width=80)
    st.title("🛰️ SatDet")
    st.caption("Aerospace Object Detection System")
    st.divider()

    st.subheader("⚙️ Model Settings")
    weights_path = st.text_input("Model Weights", value="models/weights/best.pt")
    conf_thresh = st.slider("Confidence Threshold", 0.05, 0.95, 0.25, 0.05)
    iou_thresh = st.slider("NMS IoU Threshold", 0.1, 0.9, 0.45, 0.05)

    st.divider()
    st.subheader("🔬 Inference Mode")
    use_sahi = st.checkbox("SAHI Sliced Inference", value=False,
                           help="Use for high-res images (>1280px). Slower but detects more small objects.")
    if use_sahi:
        tile_size = st.select_slider("Tile Size", options=[320, 416, 512, 640, 768], value=640)
        overlap = st.slider("Tile Overlap", 0.1, 0.4, 0.2, 0.05)
    else:
        tile_size, overlap = 640, 0.2

    show_gradcam = st.checkbox("Show Grad-CAM Heatmap", value=False,
                                help="Visualize model attention regions (slower)")

    st.divider()
    st.caption("SatDet v1.0 | YOLOv8s | DOTA v1.0")


# ── Load Model ────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model(weights: str):
    """Load YOLO model (cached across reruns)."""
    try:
        from ultralytics import YOLO
        model = YOLO(weights if Path(weights).exists() else "yolov8s.pt")
        return model, None
    except Exception as e:
        return None, str(e)


PALETTE = [
    (255, 56, 56), (255, 157, 151), (255, 112, 31), (255, 178, 29),
    (207, 210, 49), (72, 249, 10), (146, 204, 23), (61, 219, 134),
    (26, 147, 52), (0, 212, 187), (44, 153, 168), (0, 194, 255),
    (52, 69, 147), (100, 115, 255), (0, 24, 236), (132, 56, 255),
]


def decode_image_bytes(image_bytes: bytes, content_type: str = ""):
    if not image_bytes:
        raise ValueError("Cannot decode image: empty response")

    img_array = np.frombuffer(image_bytes, np.uint8)
    img_bgr = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    if img_bgr is not None:
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        return img_bgr, img_rgb

    try:
        pil_img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        content_label = f" ({content_type})" if content_type else ""
        raise ValueError(
            f"Cannot decode image{content_label}. Use a direct JPEG, PNG, or WebP image URL."
        ) from exc

    img_rgb = np.array(pil_img)
    img_bgr = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)
    return img_bgr, img_rgb


def image_request_headers(url: str):
    parsed = urlparse(url)
    referer = f"{parsed.scheme}://{parsed.netloc}/" if parsed.scheme and parsed.netloc else ""
    if parsed.netloc.endswith("wikimedia.org"):
        referer = "https://commons.wikimedia.org/"

    return {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0 Safari/537.36"
        ),
        "Accept": "image/avif,image/webp,image/apng,image/*,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": referer,
    }


def run_detection(img_bgr):
    t0 = time.time()

    if use_sahi:
        from src.inference.sahi_predictor import SAHIPredictor
        predictor = SAHIPredictor(
            weights_path if Path(weights_path).exists() else "yolov8s.pt",
            conf=conf_thresh, iou=iou_thresh,
            tile_size=tile_size, overlap=overlap,
        )
        result = predictor.predict(img_bgr)
        boxes = result["boxes"]
        scores = result["scores"]
        labels = result["labels"]
        class_names = result["class_names"]
        num_tiles = result.get("num_tiles", 1)
    else:
        results = model.predict(
            img_bgr, conf=conf_thresh, iou=iou_thresh, verbose=False
        )
        r = results[0]
        has_boxes = r.boxes is not None and len(r.boxes) > 0
        boxes = r.boxes.xyxy.cpu().numpy() if has_boxes else np.array([]).reshape(0, 4)
        scores = r.boxes.conf.cpu().numpy() if has_boxes else np.array([])
        labels = r.boxes.cls.cpu().numpy().astype(int) if has_boxes else np.array([])
        class_names = model.names
        num_tiles = 1

    inf_ms = (time.time() - t0) * 1000
    return boxes, scores, labels, class_names, num_tiles, inf_ms


def draw_detections(img_bgr, boxes, scores, labels, class_names):
    vis_img = img_bgr.copy()
    for box, score, label in zip(boxes, scores, labels):
        x1, y1, x2, y2 = map(int, box)
        color = PALETTE[int(label) % len(PALETTE)]
        cls_name = class_names.get(int(label), str(label))
        cv2.rectangle(vis_img, (x1, y1), (x2, y2), color, 2)
        label_txt = f"{cls_name} {score:.2f}"
        cv2.putText(
            vis_img, label_txt, (x1, max(y1 - 5, 10)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2,
        )
    return vis_img


def show_detection_results(
    img_bgr,
    img_rgb,
    filename,
    boxes,
    scores,
    labels,
    class_names,
    num_tiles,
    inf_ms,
):
    vis_img = draw_detections(img_bgr, boxes, scores, labels, class_names)
    vis_rgb = cv2.cvtColor(vis_img, cv2.COLOR_BGR2RGB)

    col_img, col_stats = st.columns([2, 1])

    with col_img:
        st.image(vis_rgb, caption=f"Detections: {len(scores)}", width="stretch")
        _, buf_enc = cv2.imencode(".jpg", vis_img)
        st.download_button(
            "💾 Download Result",
            data=buf_enc.tobytes(),
            file_name=f"satdet_{filename}",
            mime="image/jpeg",
        )

    with col_stats:
        st.subheader("📊 Detection Results")
        st.metric("Objects Detected", len(scores))
        st.metric("Inference Time", f"{inf_ms:.0f} ms")
        if use_sahi:
            st.metric("Tiles Processed", num_tiles)

        st.divider()

        if len(scores) > 0:
            class_counts = Counter(
                class_names.get(int(label), str(label)) for label in labels
            )
            st.write("**Detected Classes:**")
            for cls_name, count in class_counts.most_common():
                st.markdown(
                    f'<span class="detection-badge">{cls_name}: {count}</span>',
                    unsafe_allow_html=True,
                )

            st.divider()
            st.write("**Top Detections:**")
            sorted_idxs = np.argsort(-scores)[:10]
            for idx in sorted_idxs:
                cls = class_names.get(int(labels[idx]), str(labels[idx]))
                conf = scores[idx]
                st.caption(f"• {cls}: {conf:.3f}")
        else:
            st.info("No detections above confidence threshold")

    if show_gradcam:
        st.subheader("🔥 Grad-CAM Attention Map")
        with st.spinner("Generating attention heatmap..."):
            try:
                from src.visualization.gradcam import YOLOGradCAM
                gcam = YOLOGradCAM(model)
                try:
                    _, overlay = gcam.generate(img_bgr)
                finally:
                    gcam.remove_hooks()
                overlay_rgb = cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB)
                col_orig, col_heat = st.columns(2)
                with col_orig:
                    st.image(img_rgb, caption="Original", width="stretch")
                with col_heat:
                    st.image(overlay_rgb, caption="Grad-CAM Attention", width="stretch")
            except Exception as e:
                st.warning(f"Grad-CAM failed: {e}")


def process_image(img_bgr, img_rgb, filename):
    with st.spinner("Running inference..."):
        boxes, scores, labels, class_names, num_tiles, inf_ms = run_detection(img_bgr)
    show_detection_results(
        img_bgr, img_rgb, filename,
        boxes, scores, labels, class_names, num_tiles, inf_ms,
    )


# ── Main App ──────────────────────────────────────────────────────────────────
st.title("🛰️ SatDet — Satellite Object Detection")
st.caption("Detect aircraft, ships, vehicles, and infrastructure in satellite/aerial imagery using YOLOv8")

# Load model
model, model_error = load_model(weights_path)
if model_error:
    st.error(f"⚠️ Model loading failed: {model_error}")
    st.stop()

col_status1, col_status2, col_status3 = st.columns(3)
with col_status1:
    st.markdown('<div class="metric-card"><h2>✅</h2><p>Model Loaded</p></div>', unsafe_allow_html=True)
with col_status2:
    st.markdown(f'<div class="metric-card"><h2>{len(model.names)}</h2><p>Object Classes</p></div>', unsafe_allow_html=True)
with col_status3:
    mode = "SAHI" if use_sahi else "Standard"
    st.markdown(f'<div class="metric-card"><h2>{mode}</h2><p>Inference Mode</p></div>', unsafe_allow_html=True)

st.divider()

# ── Upload Area ───────────────────────────────────────────────────────────────
tab_upload, tab_url, tab_info = st.tabs(["📁 Upload Image", "🔗 From URL", "ℹ️ Model Info"])

with tab_upload:
    uploaded_file = st.file_uploader(
        "Upload satellite or aerial image",
        type=["jpg", "jpeg", "png", "tif", "tiff"],
        help="Supports JPEG, PNG, and GeoTIFF formats"
    )

    if uploaded_file:
        try:
            img_bytes = uploaded_file.read()
            img_bgr, img_rgb = decode_image_bytes(img_bytes, uploaded_file.type)
            h, w = img_bgr.shape[:2]
            st.caption(f"📐 Image: {uploaded_file.name} — {w}×{h}px — {len(img_bytes)/1e6:.1f} MB")

            if st.button("🚀 Run Detection", type="primary", key="upload_detection"):
                process_image(img_bgr, img_rgb, uploaded_file.name)
        except ValueError as e:
            st.error(str(e))


with tab_url:
    image_url = st.text_input("Image URL")

    if st.button("🚀 Run Detection", type="primary", key="url_detection"):
        url = image_url.strip()
        if not url:
            st.warning("Image URL is required")
        else:
            try:
                with st.spinner("Downloading image..."):
                    response = httpx.get(
                        url,
                        headers=image_request_headers(url),
                        follow_redirects=True,
                        timeout=httpx.Timeout(
                            connect=10.0,
                            read=60.0,
                            write=10.0,
                            pool=10.0,
                        ),
                    )
                    response.raise_for_status()

                content_type = response.headers.get("content-type", "").split(";")[0].lower()
                if content_type and not (
                    content_type.startswith("image/")
                    or content_type == "application/octet-stream"
                ):
                    raise ValueError(f"URL did not return an image (Content-Type: {content_type})")

                img_bgr, img_rgb = decode_image_bytes(response.content, content_type)
                h, w = img_bgr.shape[:2]
                st.caption(f"📐 Image: URL — {w}×{h}px — {len(response.content)/1e6:.1f} MB")
                process_image(img_bgr, img_rgb, "url_image.jpg")
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 403:
                    st.error(
                        "Failed to load image: the server blocked direct download "
                        "(403 Forbidden). Try another direct image URL or upload the image file."
                    )
                else:
                    st.error(f"Failed to load image: {e}")
            except (httpx.HTTPError, ValueError) as e:
                st.error(f"Failed to load image: {e}")


with tab_info:
    st.subheader("📋 Model Information")
    st.json({
        "architecture": "YOLOv8s",
        "dataset": "DOTA v1.0",
        "num_classes": len(model.names),
        "classes": {str(k): v for k, v in model.names.items()},
        "input_size": 640,
        "inference_modes": ["Standard (640×640)", "SAHI (sliced, any resolution)"],
        "export_formats": ["PyTorch (.pt)", "ONNX (.onnx)", "TensorRT (.engine)"],
    })

    st.subheader("🗺️ SAHI vs Standard Inference")
    st.markdown("""
    | Mode | Speed | Small Object Detection | Use Case |
    |------|-------|----------------------|----------|
    | Standard | Fast (5-15ms) | Limited | Pre-tiled imagery |
    | SAHI | Slower (50-500ms) | Excellent | Raw satellite imagery |
    """)
