# Real_Time_Doodle_Classifier

This project uses computer vision, machine learning, and text-to-speech to classify objects and gestures in real-time. It employs a pre-trained deep learning model and incorporates webcam video processing via OpenCV, along with hand-tracking functionality from MediaPipe.

## Features
- **Real-Time  Doodle Classification**: Detects doodle and gestures from live webcam footage.
- **Text-to-Speech Feedback**: Announces classification results audibly.
- **Hand/Gesture Tracking**: Enhances classification with gesture-based inputs using MediaPipe.

## Dataset
The classification model has been trained on the **Google Quick, Draw! dataset**, which includes millions of doodles from various categories. [Google's Quick, Draw!](https://quickdraw.withgoogle.com/data) project provides a large-scale, hand-drawn dataset for machine learning applications.

## Requirements
Install the following libraries before running the script:
```bash
pip install opencv-python mediapipe tensorflow pyttsx3 numpy
```

## Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/abilbiju/Real_Time_Doodle_Classifier.git
   ```
2. Navigate into the project directory:
   ```bash
   cd Real_Time_Doodle_Classifier
   ```
3. Ensure the `keras_Model.h5` file is in the project directory, as it is required for the model to function.

## Usage
To start the program, run:
```bash
python mascript.py
```

On execution, the script will:
1. Activate your webcam to capture live video.
2. Track gestures and detect objects using the pre-trained model.
3. Provide audio feedback for detected objects or gestures.

## Class Categories
The model can identify a wide range of objects and shapes, including:
- Apple
- Butterfly
- Bus
- Circle
- Cloud
- Ice Cream
- ...and more.

## Acknowledgments
This project utilizes:
- **Google Quick, Draw! Dataset** for model training data.
- **MediaPipe** for efficient hand tracking.
- **TensorFlow/Keras** for model loading and inference.
- **OpenCV** for real-time video handling.
- **pyttsx3** for spoken feedback.

