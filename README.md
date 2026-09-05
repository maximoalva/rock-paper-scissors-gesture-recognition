# Rock-Paper-Scissors Gesture Recognition

> Aprendizaje Automático II  
> Tecnicatura Universitaria en Inteligencia Artificial (Universidad Nacional de Rosario)  
> Máximo Alva, María Sol Aranda  
> 2025

## Overview

Real-time rock-paper-scissors gesture recognition using computer vision and deep learning.

The system uses a webcam to detect a hand with **MediaPipe**, extract its 21 hand landmarks, and classify the gesture as **Rock, Paper, or Scissors** using a fully connected neural network built with TensorFlow/Keras.

Instead of processing raw images, the model uses the normalized `(x, y)` coordinates of the 21 hand landmarks, resulting in a compact 42-feature representation.

## System Pipeline

```text
Webcam
   ↓
MediaPipe Hand Detection
   ↓
21 Hand Landmarks
   ↓
42 Normalized Features
   ↓
Neural Network Classifier
   ↓
Rock / Paper / Scissors
```

## Dataset Collection

The dataset is collected using a webcam and MediaPipe hand landmarks.

- **Classes:** Rock, Paper, Scissors
- **Landmarks:** 21 per hand
- **Features:** 42 normalized `(x, y)` coordinates
- **Augmentation:** horizontally flipped landmark samples
- **Output:** `rps_dataset.npy` and `rps_labels.npy`

Run:

```bash
python record-dataset.py
```

Keyboard controls:

- `P` — Rock
- `A` — Paper
- `T` — Scissors
- `Q` — Exit

Each recorded sample is also horizontally flipped to increase the diversity of the dataset.

## Model

The gesture classifier is a fully connected neural network implemented with Keras.

```text
Input (42 features)
        ↓
Dense (128, ReLU)
        ↓
Dropout (0.2)
        ↓
Dense (64, ReLU)
        ↓
Dropout (0.2)
        ↓
Dense (3, Softmax)
```

### Training Configuration

- Optimizer: Adam
- Learning rate: `0.001`
- Loss: Categorical Crossentropy
- Epochs: `80`
- Batch size: `16`
- Validation split: `20%`
- Dropout: `0.2`

The trained model is saved as:

```text
rps_model.h5
```

## Evaluation

The training script generates:

- Training and validation accuracy
- Training and validation loss
- Confusion matrix
- Precision
- Recall
- F1-score
- Overall accuracy on the collected dataset

The evaluation is performed on the full collected dataset after training; no separate held-out test set is used.

## Real-Time Recognition

Once the model has been trained, the system can perform real-time gesture recognition through the webcam.

For each frame:

1. MediaPipe detects the hand.
2. The 21 hand landmarks are extracted.
3. The landmarks are converted into a 42-feature vector.
4. The neural network predicts the three gesture probabilities.
5. The gesture with the highest probability is displayed on screen.

Run:

```bash
python rock-paper-scissors.py
```

Press `Q` to exit.

## Technologies

- Python 3.10
- TensorFlow / Keras
- MediaPipe
- OpenCV
- NumPy
- Matplotlib
- Neural Networks
- Computer Vision

## Project Structure

```text
.
├── record-dataset.py
├── train-gesture-classifier.py
├── rock-paper-scissors.py
├── utils.py
├── rps_dataset.npy
├── rps_labels.npy
├── rps_model.h5
├── requirements.txt
└── README.md
```

## Installation

Python **3.10** is required, as the versions of TensorFlow and scikit-learn specified in `requirements.txt` are only compatible with Python 3.10.

Clone the repository:

```bash
git clone https://github.com/maximoalva/rock-paper-scissors-gesture-recognition.git
cd rock-paper-scissors-gesture-recognition
```

Create and activate a virtual environment:

### Linux / macOS

```bash
python3.10 -m venv venv
source venv/bin/activate
```

### Windows

```bash
py -3.10 -m venv venv
venv\Scripts\activate
```

Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

### 1. Collect the Dataset

```bash
python record-dataset.py
```

Record examples for each gesture using the keyboard controls described above.

### 2. Train the Classifier

```bash
python train-gesture-classifier.py
```

This loads the collected dataset, trains the neural network, generates the evaluation metrics, and saves the trained model as `rps_model.h5`.

### 3. Run Real-Time Recognition

```bash
python rock-paper-scissors.py
```

Use the webcam to show a Rock, Paper, or Scissors gesture and the predicted class will be displayed in real time.

## Key Takeaways

- Hand landmark detection with MediaPipe
- Custom dataset collection using webcam input
- Landmark-based data augmentation
- Multiclass gesture classification with a neural network
- Evaluation using confusion matrices and classification metrics
- Real-time computer vision inference

Using hand landmarks instead of raw images provides a compact and structured representation of the hand while allowing MediaPipe to handle the hand detection and landmark extraction.
