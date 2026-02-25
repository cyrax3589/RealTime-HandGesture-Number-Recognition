# Real-Time Hand Gesture and Number Recognition

This project implements a real-time hand gesture and number recognition system using deep learning and computer vision. The system detects hands using MediaPipe, crops the hand region, and classifies gestures using a transfer learning model based on MobileNetV2. It supports live webcam inference with confidence scores and smoothing for stable predictions.

The model is trained on a custom hand landmark image dataset containing numerical gestures (1–9) and can be extended to alphabet or custom sign language datasets.

---

## Features

- Automatic dataset train/test splitting
- Transfer learning with MobileNetV2 for high accuracy on small datasets
- Real-time hand tracking using MediaPipe
- Live webcam gesture recognition
- Confidence score visualization
- Prediction smoothing for stable outputs
- Confusion matrix and performance metrics
- Training accuracy and loss visualization

---

## Project Structure
<img width="370" height="521" alt="image" src="https://github.com/user-attachments/assets/683fd010-d930-412d-8fbc-c27c50ef832a" />

---

## Technologies Used

- Python
- TensorFlow / Keras
- OpenCV
- MediaPipe
- MobileNetV2
- NumPy
- Matplotlib
- Scikit-learn

---

## Dataset Format

Each gesture class is stored in a separate folder
Each folder contains approximately 25 hand gesture images of size 300x300 pixels.

The training script automatically splits the dataset into training and testing sets.

---

## How to Train/Test the Model

```bash
python train_model.py
python test_model.py
```

---

## Challenges and Solutions

One of the main challenges was working with a small dataset, where each class had only around 25 images. This increased the risk of overfitting. To address this, I implemented data augmentation techniques such as rotation, zoom, and shift transformations. I also used transfer learning with MobileNetV2, which significantly improved performance compared to training a CNN from scratch.

Another challenge was unstable predictions during real-time webcam testing. Since live video frames can vary due to lighting and slight hand movements, predictions were flickering between classes. To solve this, I implemented prediction smoothing by averaging outputs across multiple frames before displaying the result.

Background noise and full-frame predictions also reduced accuracy. Integrating MediaPipe for hand detection allowed me to crop only the hand region before classification, which improved model focus and prediction stability.

Version compatibility issues between TensorFlow, MediaPipe, and NumPy were another technical challenge. Managing correct library versions and ensuring compatibility was necessary for stable execution.

Finally, evaluating the model using only accuracy was insufficient. By adding confusion matrices and classification reports, I was able to better understand class-wise performance and identify weak predictions.

---

## Future Improvements

There are several directions to enhance this project further.

The model can be fine-tuned by unfreezing deeper layers of MobileNetV2 to improve accuracy once more training data becomes available.

The system can be extended to support alphabet recognition and full sign language word formation instead of just numerical gestures.

A text-to-speech module can be integrated to convert recognized gestures into spoken output, making it more useful for assistive communication.

The model can be optimized and converted to TensorFlow Lite for deployment on mobile devices or embedded systems.

A graphical user interface (GUI) can be developed for better usability and deployment as a desktop application.

Model monitoring and logging can be added to track prediction confidence trends over time, which would be useful in a real production environment.

Finally, the project can be extended to multi-hand recognition and gesture-based control systems for real-world human-computer interaction applications.

---

## Output


