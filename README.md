# 👁️ Empathic-Vision: FER-Mental-Health-Analytics

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Deep%20Learning-orange)
![License](https://img.shields.io/badge/License-MIT-green)

## 🌟 Project Overview
*"The face is the mirror of the mind, and eyes without speaking confess the secrets of the heart."*

This project leverages **Computer Vision** and **Deep Learning** to build a real-time Facial Expression Recognition (FER) system. Unlike standard emotion detectors, this system is conceptualized to assist in **mental health monitoring**. By tracking emotional states over time (e.g., persistent sadness, lack of expression, or high stress indicators), it aims to provide non-invasive insights into a user's well-being.

> **Note:** This tool is a proof-of-concept for educational purposes and is not a substitute for professional clinical diagnosis.

## 🧠 Key Features
* **Real-Time Detection:** Instantly analyzes facial expressions via webcam feed using OpenCV.
* **7 Emotion Classes:** Detects *Angry, Disgust, Fear, Happy, Neutral, Sad, and Surprise*.
* **Mental Health Mapping:** Interprets clusters of negative emotions (like chronic sadness or fear) as potential indicators of depressive or anxiety disorders.
* **Privacy-First:** All processing happens locally on your machine; no video data is sent to the cloud.

## 🛠️ Tech Stack
* **Language:** Python
* **Computer Vision:** OpenCV (`cv2`) for face detection (Haar Cascades).
* **Deep Learning:** TensorFlow/Keras (CNN architecture) trained on the **FER-2013** dataset.
* **Data Processing:** NumPy & Pandas.

## 📂 Project Structure
* `Models/`: Contains the pre-trained deep learning weights (`.h5` files).
* `utilities/`: Helper scripts and Haar Cascade XML files for face detection.
* `FER2013_processed/`: (Ignored in repo) The pre-processed image dataset.
* `run_webcam.bat`: A one-click script to launch the application.

## 🚀 How to Run

1.  **Clone the Repository**
    ```bash
    git clone https://github.com/Sanjeeb58/FER-Mental-Health-Analytics.git
    cd FER-Mental-Health-Analytics
    ```

2.  **Install Dependencies**
    ```bash
    pip install -r requirements.txt
    ```

3.  **Launch the System**
    You can simply double-click the **`run_webcam.bat`** file, or run:
    ```bash
    python main.py
    ```
    *(Note: Ensure your webcam is connected).*

## 📊 Model Performance
The model was trained on the FER-2013 dataset (35,000+ images).
* **Architecture:** Custom Convolutional Neural Network (CNN) with Dropout and Batch Normalization to prevent overfitting.
* **Accuracy:** Achieved ~65-70% accuracy on the test set (comparable to human-level performance on this difficult dataset).

## 🤝 Future Scope
* **Micro-expression Analysis:** Detecting fleeting expressions that people try to hide.
* **Multi-modal Fusion:** Combining this visual data with voice tone analysis for higher accuracy.
* **IoT Integration:** Deploying on edge devices like Raspberry Pi for smart mirrors.

## 📧 Contact
* **GitHub:** [Sanjeeb58](https://github.com/Sanjeeb58)

---
*Built with ❤️ and code to foster mental wellness.*
