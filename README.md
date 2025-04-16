# Traffic Sign Detection using YOLOv8

This repository contains the code, trained model weights, dataset, and results for detecting various Indian traffic signs using the YOLOv8 object detection model provided by Ultralytics.


<!-- Replace the image path above with a link to one of your best sample detection output images -->
<!-- Or add multiple images: -->
<!-- ![Sample 1](path/to/sample1.jpg) -->
<!-- ![Sample 2](path/to/sample2.jpg) -->


## Table of Contents

*   [Project Goal](#project-goal)
*   [Dataset](#dataset)
*   [Model](#model)
*   [Technology Stack](#technology-stack)
*   [Setup](#setup)
*   [Usage](#usage)
    *   [Training](#training)
    *   [Evaluation](#evaluation)
    *   [Inference (Prediction)](#inference-prediction)
*   [Results](#results)
*   [File Structure](#file-structure)
*   [License](#license)
*   [Acknowledgements](#acknowledgements)

## Project Goal

The primary goal of this project is to train an accurate object detection model capable of identifying and localizing various traffic signs commonly found on Indian roads from images or video streams.

## Dataset

*   **Source:** The dataset used is "Indian-Traffic-Signboards--5" obtained from Roboflow.
    <!-- Optional: Add link if your dataset version is public on Roboflow -->
    <!-- [Link to Roboflow Dataset](https://app.roboflow.com/...) -->
*   **Format:** YOLOv8 PyTorch TXT
*   **Classes:** [Mention the number of traffic sign classes, e.g., 56] classes.
*   **Contents:** The `datasets2/` folder contains the training, validation, and test splits along with the `data.yaml` configuration file.

## Model

*   **Architecture:** YOLOv8 (specifically `yolov8s.pt` was used as the base pre-trained model).
*   **Training:** The model was trained for 50 epochs on the provided dataset.
*   **Weights:** The best-performing trained weights are located at `runs/detect/train2/weights/best.pt`.

## Technology Stack

*   [Python](https://www.python.org/) (e.g., 3.10+)
*   [PyTorch](https://pytorch.org/)
*   [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) (version `8.0.196` or similar used for training, potentially newer for inference/validation shown in logs)
*   [Roboflow](https://roboflow.com/) (for dataset sourcing/management)
*   [Google Colab](https://colab.research.google.com/) (used for training and experimentation)
*   NumPy, OpenCV-Python (as dependencies)

## Setup

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/[ns-0437]/[Traffic-Signal-Detection].git
    cd [Traffic-Signal-Detection]
    ```

2.  **Set up environment (Recommended: Virtual Environment):**
    ```bash
    python -m venv venv
    # Activate the environment (Linux/macOS)
    source venv/bin/activate
    # Activate the environment (Windows)
    # venv\Scripts\activate
    ```

3.  **Install dependencies:**
    *   Install PyTorch matching your system/CUDA version from the [official PyTorch website](https://pytorch.org/get-started/locally/).
    *   Install Ultralytics and other requirements:
        ```bash
        pip install ultralytics==8.0.196 # Or the version you need/used
        pip install -r requirements.txt # Optional: Create a requirements.txt if needed
        ```
    *   *(Alternatively, if running in Google Colab, necessary installations are likely handled within the notebook itself).*

## Usage

*(Code examples assume you are running Python scripts or cells within a notebook where the `ultralytics` library is imported and the model is potentially loaded.)*

### Training

The model was trained using a command similar to this (likely within a Python script or notebook):

```python
from ultralytics import YOLO

# Load a pre-trained model (e.g., yolov8s.pt)
model = YOLO('yolov8s.pt')

# Train the model on the custom dataset
results = model.train(data='datasets2/Indian-Traffic-Signboards--5/data.yaml',
                      epochs=50,
                      imgsz=640,
                      # Add other relevant parameters used, e.g., batch, device etc.
                     )
