# ============================================================
# SmartVision AI - Intelligent Multi-Class Object Recognition
# ============================================================

from pathlib import Path
import time

import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications.resnet50 import preprocess_input
from ultralytics import YOLO
from streamlit_webrtc import VideoProcessorBase, webrtc_streamer
import av

st.set_page_config(page_title="SmartVision AI",page_icon="👁️",layout="wide",initial_sidebar_state="expanded",)

APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = Path(__file__).resolve().parent
MODEL_DIR = PROJECT_DIR / "models"

CLASSIFIER_PATH = MODEL_DIR / "resnet50_finetuned_best.keras"
DETECTOR_PATH = MODEL_DIR / "best.pt"
IMAGE_SIZE = (224, 224)

# Alphabetical class order used by image_dataset_from_directory.
CLASS_NAMES = ["airplane", "bed", "bench", "bicycle", "bird", "bottle", "bowl","bus", "cake", "car", "cat", "chair", "couch", "cow", "cup","dog", "elephant", "horse", "motorcycle", "person", "pizza","potted plant", "stop sign", "traffic light", "train", "truck",]

# Latest recorded evaluation of the saved fine-tuned ResNet50 checkpoint.
TEST_ACCURACY = 0.4000
MACRO_PRECISION = 0.4468
MACRO_RECALL = 0.4000
MACRO_F1 = 0.4056


@st.cache_resource(show_spinner="Loading ResNet50 classifier...")
def load_classifier(model_path: str):
    return tf.keras.models.load_model(model_path, compile=False)


@st.cache_resource(show_spinner="Loading YOLO detector...")
def load_detector(model_path: str):
    return YOLO(model_path)


def prepare_for_resnet(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB").resize(IMAGE_SIZE)
    array = np.asarray(image, dtype=np.float32)
    array = np.expand_dims(array, axis=0)
    return preprocess_input(array)


def classify_image(image: Image.Image, model):
    probabilities = model.predict(prepare_for_resnet(image), verbose=0)[0]

    if len(probabilities) != len(CLASS_NAMES):
        raise ValueError(
            f"Model returned {len(probabilities)} outputs but "
            f"{len(CLASS_NAMES)} class names are configured."
        )

    top_indices = np.argsort(probabilities)[::-1][:5]
    return [
        {"class": CLASS_NAMES[int(i)], "confidence": float(probabilities[i])}
        for i in top_indices
    ]


def detect_objects(image: Image.Image, model, confidence: float):
    result = model.predict(
        source=image.convert("RGB"), conf=confidence, verbose=False
    )[0]
    annotated = result.plot()[:, :, ::-1]  # BGR to RGB

    detections = []
    if result.boxes is not None:
        for box in result.boxes:
            class_id = int(box.cls.item())
            x1, y1, x2, y2 = [float(v) for v in box.xyxy[0].tolist()]
            detections.append({
                "Object": result.names.get(class_id, str(class_id)),
                "Confidence": float(box.conf.item()),
                "X1": round(x1, 1),
                "Y1": round(y1, 1),
                "X2": round(x2, 1),
                "Y2": round(y2, 1),
            })

    return annotated, detections


class LiveYOLOProcessor(VideoProcessorBase):
    """Run YOLO on each incoming webcam frame and return an annotated frame."""

    def __init__(self, model, confidence: float = 0.25):
        self.model = model
        self.confidence = confidence

    def recv(self, frame: av.VideoFrame) -> av.VideoFrame:
        image = frame.to_ndarray(format="rgb24")
        result = self.model.predict(
            source=image,
            conf=self.confidence,
            verbose=False,
            imgsz=640,
        )[0]
        annotated = result.plot()[:, :, ::-1]  # BGR to RGB
        return av.VideoFrame.from_ndarray(annotated, format="rgb24")


with st.sidebar:
    st.title("👁️ SmartVision AI")
    st.caption("Intelligent Multi-Class Object Recognition")
    st.divider()
    page = st.radio(
        "Navigation",
        ["Overview", "Image Classification", "Object Detection", "Real-Time Webcam", "Model Performance"],
    )
    st.divider()
    st.caption("Models are loaded from the project's models/ directory.")


if page == "Overview":
    st.title("SmartVision AI")
    st.subheader("Intelligent Multi-Class Object Recognition System")
    st.write(
        "Classify an image using fine-tuned ResNet50, detect multiple objects "
        "with YOLO, or run real-time webcam detection."
    )

    col1, col2, col3 = st.columns(3)
    col1.metric("Classification classes", "26")
    col2.metric("Test images evaluated", "390")
    col3.metric("ResNet50 test accuracy", "40.00%")

    left, right = st.columns(2)
    with left:
        st.markdown("### 🧠 Image Classification")
        st.write("Predicts one of 26 classes and displays the top-five scores.")
    with right:
        st.markdown("### 🎯 Object Detection")
        st.write("Detects multiple objects and draws bounding boxes.")

    st.markdown("### 📹 Real-Time Webcam")
    st.write("Run YOLO on a live webcam stream with annotated video frames.")

    st.info(
        "The displayed accuracy is a measured result for the saved checkpoint. "
        "Performance can vary across classes and image conditions."
    )


elif page == "Image Classification":
    st.title("Image Classification")
    st.write("Upload an image from one of the 26 trained classes.")

    uploaded_file = st.file_uploader(
        "Choose an image", type=["jpg", "jpeg", "png", "webp"],
        key="classification_upload",
    )

    if uploaded_file:
        image = Image.open(uploaded_file).convert("RGB")
        left, right = st.columns(2)
        with left:
            st.image(image, caption="Uploaded image", use_container_width=True)

        if not CLASSIFIER_PATH.exists():
            st.error(f"Classifier model not found: {CLASSIFIER_PATH}")
        else:
            try:
                model = load_classifier(str(CLASSIFIER_PATH))
                with st.spinner("Classifying image..."):
                    start = time.perf_counter()
                    predictions = classify_image(image, model)
                    elapsed = time.perf_counter() - start

                with right:
                    best = predictions[0]
                    st.markdown("### Prediction")
                    st.success(f"Predicted class: **{best['class']}**")
                    
            except Exception as exc:
                st.error(f"Classification failed: {exc}")


elif page == "Object Detection":
    st.title("Object Detection")
    st.write("Upload an image to detect objects and visualize bounding boxes.")

    uploaded_file = st.file_uploader(
        "Choose an image",
        type=["jpg", "jpeg", "png", "webp"],
        key="detection_upload",
    )
    confidence = st.slider(
        "Detection confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")

        if not DETECTOR_PATH.exists():
            st.error(f"YOLO model not found: {DETECTOR_PATH}")
        else:
            try:
                detector = load_detector(str(DETECTOR_PATH))

                with st.spinner("Detecting objects..."):
                    start = time.perf_counter()
                    annotated, detections = detect_objects(
                        image, detector, confidence
                    )
                    elapsed = time.perf_counter() - start

                st.markdown("### Original image")
                st.image(image, use_container_width=False, width=650)

                st.markdown("### Detected objects")
                st.image(annotated, use_container_width=False, width=650)

                if detections:
                    st.markdown("### Predictions")
                    for index, item in enumerate(detections, start=1):
                        st.write(f"{index}. **{item['Object']}**")
                else:
                    st.warning(
                        "No objects detected at this confidence threshold. "
                        "Try lowering the threshold."
                    )

            except Exception as exc:
                st.error(f"Object detection failed: {exc}")



elif page == "Real-Time Webcam":
    st.title("Real-Time Webcam Detection")
    st.write(
        "Start your camera to run live YOLO object detection. "
        "Allow camera access when your browser asks."
    )

    live_confidence = st.slider(
        "Live detection confidence threshold",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05,
        key="live_detection_confidence",
    )

    if not DETECTOR_PATH.exists():
        st.error(f"YOLO model not found: {DETECTOR_PATH}")
    else:
        try:
            detector = load_detector(str(DETECTOR_PATH))
            webrtc_streamer(
                key="smartvision-live-yolo",
                video_processor_factory=lambda: LiveYOLOProcessor(detector, live_confidence),
                media_stream_constraints={"video": True, "audio": False},
                async_processing=True,)
            st.caption(
                "Tip: good lighting and a stable internet connection help. "
                "If video is slow, try a lower camera resolution."
            )
        except Exception as exc:
            st.error(f"Could not start live webcam detection: {exc}")


elif page == "Model Performance":
    st.title("Model Performance")
    st.write("Recorded evaluation metrics for the saved fine-tuned ResNet50.")

    c1, c2 = st.columns(2)
    c1.metric("Test accuracy", f"{TEST_ACCURACY:.2%}")
    c2.metric("Macro F1-score", f"{MACRO_F1:.2%}")
    c3, c4 = st.columns(2)
    c3.metric("Macro precision", f"{MACRO_PRECISION:.2%}")
    c4.metric("Macro recall", f"{MACRO_RECALL:.2%}")

    st.markdown("### Evaluation details")
    st.write("- Test split: 390 images")
    st.write("- Classes: 26")
    st.write("- Input size: 224 × 224 RGB")
    st.write("- Architecture: fine-tuned ResNet50")

    st.warning(
        "Verify that the model's training preprocessing and class order match "
        "this application before treating these metrics as final benchmarks."
    )

    report_path = PROJECT_DIR / "metrics" / "resnet50_classification_report.csv"
    if report_path.exists():
        st.download_button(
            "Download classification report (CSV)",
            data=report_path.read_bytes(),
            file_name="resnet50_classification_report.csv",
            mime="text/csv",
        )
