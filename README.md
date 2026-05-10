# Pneumonia Detection System

A Machine Learning based web application that detects Pneumonia from Chest X-Ray images using image processing and classification techniques.
The application is built using Streamlit with a modern medical-style user interface.

---

# Project Overview

The system allows users to upload a chest X-ray image and predicts whether the patient is:

* Normal
* Pneumonia Detected

The project uses:

* OpenCV for image processing
* HOG (Histogram of Oriented Gradients) for feature extraction
* Machine Learning classification model
* Streamlit for deployment and frontend UI

---

# Features

* Chest X-Ray image upload
* Real-time pneumonia prediction
* Clean medical dashboard UI
* Fast image processing
* Machine Learning based prediction
* Dark themed responsive interface
* Streamlit deployment ready

---

# Technologies Used

| Technology   | Purpose              |
| ------------ | -------------------- |
| Python       | Core Programming     |
| Streamlit    | Web Application      |
| OpenCV       | Image Processing     |
| Scikit-learn | Machine Learning     |
| NumPy        | Numerical Operations |
| PIL          | Image Handling       |
| Joblib       | Model Serialization  |
| HOG Features | Feature Extraction   |

---

# Dataset

Dataset used:
Chest X-Ray Pneumonia Dataset from Kaggle

Dataset contains:

* Normal Chest X-Rays
* Pneumonia Chest X-Rays

Dataset Structure:

```text
dataset/
│
├── NORMAL/
├── PNEUMONIA/
```

---

# Machine Learning Workflow

1. Dataset Collection
2. Image Preprocessing
3. Feature Extraction using HOG
4. Model Training
5. Model Evaluation
6. Saving Best Model (.pkl)
7. Streamlit Deployment

---

# Image Processing Steps

* Resize image
* Convert image to grayscale
* Extract HOG features
* Pass features to ML model
* Generate prediction

---

# Model Used

The project uses a Classification model because the problem is a binary classification problem.

Possible models tested:

* Logistic Regression
* Random Forest
* SVM
* KNN

Best performing model saved as:

```text
pneumonia_model.pkl
```

---

# Project Structure

```text
Pneumonia-Detection/
│
├── app.py
├── pneumonia_model.pkl
├── requirements.txt
├── README.md
│
├── dataset/
│   ├── NORMAL/
│   └── PNEUMONIA/
│
└── notebooks/
    └── training.ipynb
```

---

# Installation

## Clone Repository

```bash
git clone <repository-url>
```

---

# Install Dependencies

```bash
pip install -r requirements.txt
```

---

# Run Application

```bash
streamlit run app.py
```

---

# Inputs

Input:

* Chest X-Ray image
* Image formats:

  * JPG
  * JPEG
  * PNG

---

# Outputs

Output:

* Normal
* Pneumonia Detected

---

# Model Evaluation Metrics

The model was evaluated using:

* Accuracy
* Confusion Matrix
* Precision
* Recall
* F1 Score

Confidence score is intentionally not displayed in the UI.

---

# User Interface

The application includes:

* Dark themed medical dashboard
* Responsive layout
* Upload section
* Prediction display
* Modern styling

---

# Deployment

The project can be deployed using:

* Streamlit Cloud

---

# Requirements

Example requirements:

```text
streamlit
numpy
opencv-python
scikit-learn
joblib
Pillow
scikit-image
```
