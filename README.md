# AI-Based PCB Defect Detection and Automated Inspection System

A computer-vision-based PCB inspection system using YOLO11n, OpenCV, and Streamlit to detect, localize, and classify manufacturing defects from PCB images.

## 📌 Project Overview

Manual PCB inspection can be time-consuming and may be affected by human error. This project develops an AI-based PCB inspection system that analyzes PCB images and identifies manufacturing defects using object detection.

The trained YOLO11n model detects and localizes eight PCB defect classes and provides confidence scores for each detection.

An interactive Streamlit dashboard allows users to upload PCB images, visualize detected defects, view inspection statistics, and export inspection results.

## 🎯 Objectives

- Detect manufacturing defects from PCB images.
- Localize defects using bounding boxes.
- Classify defects into predefined categories.
- Display confidence scores for detected defects.
- Provide an interactive PCB inspection dashboard.
- Generate annotated inspection images.
- Generate downloadable inspection reports.

## 🔍 Detected Defect Classes

The system detects eight PCB defect categories:

1. Mouse Bite
2. Missing Copper
3. Scratch
4. Spurious Copper
5. Copper Burr
6. Stain
7. Short
8. Open

## 🏗️ System Architecture

```text
PCB Image
     ↓
Image Preprocessing
     ↓
YOLO11n Object Detection
     ↓
Defect Classification & Localization
     ↓
Confidence Filtering
     ↓
Annotated PCB Image
     ↓
Streamlit Inspection Dashboard
     ↓
Inspection Results & Report
```

## 🛠️ Technologies Used

- Python
- YOLO11n
- Ultralytics
- OpenCV
- NumPy
- Pandas
- Matplotlib
- Scikit-learn
- Streamlit
- Pillow
- Git
- GitHub

## 📊 Dataset

The project uses the **PCB-IND** dataset.

### Dataset Distribution

| Split | Images |
|---|---:|
| Training | 3,833 |
| Validation | 478 |
| Testing | 478 |
| **Total** | **4,789** |

The dataset contains YOLO-format annotations for eight PCB defect classes.

## 🧠 Model Training

### Model

**YOLO11n**

### Training Configuration

| Parameter | Value |
|---|---|
| Model | YOLO11n |
| Epochs | 5 |
| Image Size | 512 × 512 |
| Batch Size | 4 |
| Device | CPU |

### Final Validation Results

| Metric | Score |
|---|---:|
| Precision | 63.9% |
| Recall | 54.7% |
| mAP@50 | 55.7% |
| mAP@50-95 | 37.1% |

These metrics are from the final validation run after five training epochs.

## 🖥️ Streamlit Inspection Dashboard

The application provides:

- PCB image upload
- Adjustable detection confidence threshold
- Original PCB image visualization
- Annotated PCB image visualization
- Total defect count
- Average detection confidence
- Number of detected defect types
- Detection details table
- Defect distribution chart
- Annotated image download
- Inspection report download

## 📸 Dashboard Screenshots

### Inspection Dashboard

The Streamlit dashboard provides PCB image upload, model configuration, inspection statistics, and defect visualization.

![PCB Inspection Dashboard](screenshots/dashboard-overview.png)

### PCB Defect Detection

The system displays the original PCB image alongside the AI-detected defect with its bounding box and confidence score.

![PCB Defect Detection](screenshots/pcb-detection.png)

### Inspection Results

The dashboard provides detection details, defect distribution, and downloadable inspection results.

![Inspection Results](screenshots/inspection-results.png)

## 🔄 Inspection Workflow

```text
Upload PCB Image
       ↓
AI Model Analysis
       ↓
Defect Detection
       ↓
Bounding Box + Confidence
       ↓
Inspection Summary
       ↓
Export Results
```

## ▶️ How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/DIVAKAR-516-RAGHU/AI-PCB-Defect-Detection.git
cd AI-PCB-Defect-Detection
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

### 3. Activate the Virtual Environment

Windows:

```bash
.venv\Scripts\activate
```

### 4. Install Dependencies

```bash
pip install -r requirements.txt
```

### 5. Dataset and Model

The PCB-IND dataset and trained YOLO11n model weights are **not included in this repository** because of their size.

The dataset is expected in the following structure:

```text
dataset/
└── YOLO/
    ├── data.yaml
    ├── images/
    │   ├── train/
    │   ├── val/
    │   └── test/
    └── labels/
        ├── train/
        ├── val/
        └── test/
```

The trained model is expected at:

```text
runs/detect/results/pcb_yolo_5ep/weights/best.pt
```

### 6. Validate the Dataset

After preparing the dataset:

```bash
python src/check_dataset.py
```

A valid dataset should pass the project validation checks before training or inference.

### 7. Run the Streamlit Application

```bash
streamlit run app.py
```

Open the local Streamlit URL shown in the terminal, normally:

```text
http://localhost:8501
```

## 📁 Project Structure

```text
AI-PCB-Defect-Detection/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── src/
│   └── check_dataset.py
│
├── screenshots/
│   ├── dashboard-overview.png
│   ├── pcb-detection.png
│   └── inspection-results.png
│
├── dataset/        # Excluded from GitHub
├── runs/           # Excluded from GitHub
├── results/        # Excluded from GitHub
└── .venv/          # Excluded from GitHub
```

## 📈 Inspection Output

For an uploaded PCB image, the system provides:

- Detected defect class
- Bounding-box localization
- Detection confidence
- Annotated PCB image
- Total number of detected defects
- Average confidence
- Defect-type distribution
- Downloadable annotated image
- Downloadable inspection report

## 🔬 Example Detection

The trained model was tested through the Streamlit dashboard with PCB images containing different defect types.

Example detections included:

- Missing Copper
- Copper Burr
- Spurious Copper
- Short
- Mouse Bite
- Stain

The dashboard supports both single-defect and multiple-defect detections in an image.

## 🚀 Future Improvements

- Improve model performance through further experimentation.
- Perform detailed evaluation on the held-out test set.
- Export the trained model to ONNX for optimized CPU inference.
- Add inspection history and result storage.
- Add CSV and PDF report generation.
- Explore deployment on edge-computing hardware.
- Develop a production-oriented automated PCB inspection workflow.

## 👨‍💻 Author

**Divakar R**

B.E. Electronics and Communication Engineering  
Adithya Institute of Technology, Coimbatore

## 📄 License

This project is developed for educational, academic, and portfolio purposes.
