import streamlit as st
from ultralytics import YOLO
from PIL import Image
import pandas as pd
import io
import os

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="AI PCB Defect Inspection",
    page_icon="🔍",
    layout="wide"
)

# ============================================================
# MODEL
# ============================================================

MODEL_PATH = "runs/detect/results/pcb_yolo_5ep/weights/best.pt"

@st.cache_resource
def load_model():
    return YOLO(MODEL_PATH)

model = load_model()

# ============================================================
# HEADER
# ============================================================

st.title("🔍 AI-Based PCB Defect Detection")
st.markdown(
    "Computer-vision-based PCB inspection using YOLO11n "
    "for defect classification and localization."
)

st.divider()

# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header("⚙️ Inspection Settings")

    confidence = st.slider(
        "Detection Confidence",
        min_value=0.10,
        max_value=0.90,
        value=0.25,
        step=0.05
    )

    st.markdown("---")

    st.markdown("### 🤖 Model")
    st.write("YOLO11n")

    st.markdown("### 💻 Device")
    st.write("CPU")

    st.markdown("### 🖼️ Image Size")
    st.write("512 × 512")

# ============================================================
# IMAGE UPLOAD
# ============================================================

uploaded_file = st.file_uploader(
    "📤 Upload a PCB Image",
    type=["jpg", "jpeg", "png"]
)

# ============================================================
# PREDICTION
# ============================================================

if uploaded_file is not None:

    image = Image.open(uploaded_file).convert("RGB")

    with st.spinner("🔎 Analyzing PCB for manufacturing defects..."):

        results = model.predict(
            source=image,
            imgsz=512,
            conf=confidence,
            device="cpu",
            verbose=False
        )

    result = results[0]

    # ========================================================
    # DETECTION EXTRACTION
    # ========================================================

    detections = []

    if result.boxes is not None and len(result.boxes) > 0:

        names = model.names

        for box in result.boxes:

            class_id = int(box.cls[0])
            conf = float(box.conf[0])

            detections.append({
                "Defect": names[class_id],
                "Confidence": conf
            })

    # ========================================================
    # ANNOTATED IMAGE
    # ========================================================

    annotated_bgr = result.plot()

    # Convert BGR → RGB
    annotated_rgb = annotated_bgr[:, :, ::-1]

    annotated_image = Image.fromarray(annotated_rgb)

    # ========================================================
    # SUMMARY METRICS
    # ========================================================

    st.divider()
    st.subheader("📊 Inspection Summary")

    if detections:

        total_defects = len(detections)

        average_confidence = sum(
            d["Confidence"] for d in detections
        ) / total_defects

        unique_defects = len(
            set(d["Defect"] for d in detections)
        )

        high_confidence = sum(
            1 for d in detections
            if d["Confidence"] >= 0.50
        )

        metric1, metric2, metric3, metric4 = st.columns(4)

        with metric1:
            st.metric(
                "Total Defects",
                total_defects
            )

        with metric2:
            st.metric(
                "Avg. Confidence",
                f"{average_confidence:.1%}"
            )

        with metric3:
            st.metric(
                "Defect Types",
                unique_defects
            )

        with metric4:
            st.metric(
                "≥ 50% Confidence",
                high_confidence
            )

    else:

        st.info(
            "No defects detected above the selected "
            "confidence threshold."
        )

    # ========================================================
    # IMAGE COMPARISON
    # ========================================================

    st.divider()

    col1, col2 = st.columns(2)

    with col1:

        st.subheader("🖼️ Original PCB")

        st.image(
            image,
            use_container_width=True
        )

    with col2:

        st.subheader("🎯 Detected Defects")

        st.image(
            annotated_image,
            use_container_width=True
        )

    # ========================================================
    # DETECTION TABLE
    # ========================================================

    if detections:

        st.divider()

        st.subheader("📋 Detection Details")

        table_data = []

        for index, detection in enumerate(
            detections,
            start=1
        ):

            table_data.append({
                "Detection": index,
                "Defect": detection["Defect"],
                "Confidence": f"{detection['Confidence']:.2%}"
            })

        df = pd.DataFrame(table_data)

        st.dataframe(
            df,
            use_container_width=True,
            hide_index=True
        )

        # ====================================================
        # DEFECT COUNTS
        # ====================================================

        st.subheader("📈 Defect Distribution")

        defect_counts = (
            df["Defect"]
            .value_counts()
            .reset_index()
        )

        defect_counts.columns = [
            "Defect",
            "Count"
        ]

        st.bar_chart(
            defect_counts.set_index("Defect")
        )

        # ====================================================
        # DOWNLOAD ANNOTATED IMAGE
        # ====================================================

        st.divider()

        st.subheader("📥 Export Results")

        image_buffer = io.BytesIO()

        annotated_image.save(
            image_buffer,
            format="JPEG"
        )

        image_buffer.seek(0)

        st.download_button(
            label="🖼️ Download Annotated Image",
            data=image_buffer,
            file_name="pcb_defect_detection.jpg",
            mime="image/jpeg"
        )

        # ====================================================
        # INSPECTION REPORT
        # ====================================================

        report_lines = []

        report_lines.append(
            "AI-BASED PCB DEFECT INSPECTION REPORT"
        )
        report_lines.append(
            "=" * 45
        )
        report_lines.append("")

        report_lines.append(
            f"Input Image: {uploaded_file.name}"
        )

        report_lines.append(
            "Model: YOLO11n"
        )

        report_lines.append(
            f"Confidence Threshold: {confidence:.2f}"
        )

        report_lines.append("")

        report_lines.append(
            f"Total Defects Detected: {total_defects}"
        )

        report_lines.append(
            f"Average Confidence: {average_confidence:.2%}"
        )

        report_lines.append(
            f"Unique Defect Types: {unique_defects}"
        )

        report_lines.append("")

        report_lines.append(
            "DETECTION DETAILS"
        )

        report_lines.append(
            "-" * 45
        )

        for index, detection in enumerate(
            detections,
            start=1
        ):

            report_lines.append(
                f"{index}. "
                f"{detection['Defect']} - "
                f"{detection['Confidence']:.2%}"
            )

        report_lines.append("")
        report_lines.append(
            "Generated by AI-Based PCB Defect Detection System"
        )

        report_text = "\n".join(report_lines)

        st.download_button(
            label="📄 Download Inspection Report",
            data=report_text,
            file_name="pcb_inspection_report.txt",
            mime="text/plain"
        )

        # ====================================================
        # STATUS
        # ====================================================

        st.success(
            f"✅ Inspection completed — "
            f"{total_defects} defect(s) detected."
        )

    else:

        st.info(
            "No defects detected above the selected "
            "confidence threshold."
        )

else:

    st.info(
        "👆 Upload a PCB image above to begin inspection."
    )