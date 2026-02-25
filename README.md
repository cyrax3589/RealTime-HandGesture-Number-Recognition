# Real-Time Hand Sign Recognition using Deep Learning and Computer Vision

This project implements a high-accuracy real-time hand sign recognition system using deep learning and computer vision techniques. The system detects hands using MediaPipe, crops the hand region, and classifies gestures using a transfer learning model based on MobileNetV2.

It supports live webcam inference with confidence percentages and is trained on a custom dataset containing hand signs for numbers (0–9) and alphabets (A–Z).

---

## Key Features

- Automatic dataset splitting for training and validation  
- Transfer learning using MobileNetV2 for high performance on limited data  
- Real-time hand detection and tracking using MediaPipe  
- Hand region cropping for noise-free classification  
- Live webcam prediction with confidence percentage  
- Training accuracy and loss visualization  
- Confusion matrix and classification report for performance analysis  
- Modern Keras `.keras` model format for saving and loading  

---

## Project Structure
<img width="215" height="266" alt="image" src="https://github.com/user-attachments/assets/02498d99-553a-438d-b192-56d83d5b726d" />

Each folder represents a gesture class and contains corresponding hand images.

---

## Technologies Used

- Python  
- TensorFlow and Keras  
- OpenCV  
- MediaPipe  
- MobileNetV2 (Transfer Learning)  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  

---

## Dataset Details

- Custom self-collected hand gesture image dataset (https://www.kaggle.com/datasets/debabratakuiry/hand-sign-gesture-dataset-az-and-09-25k-images)  
- Captured using webcam under varying lighting conditions and angles  
- Image resolution: 224 × 224 pixels  
- Separate folder per class (A–Z, 0–9)  
- Automatically split into training and validation sets during training   

---


## Model Train/Test:

```bash
python train_model.py
python test_model.py
```

---

Model Performance

- Achieved approximately 99% validation accuracy across 36+ gesture classes.
- High precision, recall, and F1-score for most classes.
- Stable real-time predictions using hand cropping and confidence visualization.

---

## Output

<img width="1096" height="673" alt="Collage" src="https://github.com/user-attachments/assets/8ba0c9fc-552d-45d5-98fe-1224973aa36a" />

---

## What I Learned

Working on this project helped me understand how real-world computer vision and deep learning systems are built beyond simply training a model on a dataset.

I learned the importance of data preprocessing when working with image-based machine learning tasks. Normalizing pixel values, performing real-time data augmentation, and automatically splitting the dataset into training and validation sets significantly improved model generalization and reduced overfitting. Creating and organizing a custom self-collected dataset also helped me understand how data quality directly impacts model performance.

I gained hands-on experience with convolutional neural networks and how they extract meaningful spatial features such as edges, textures, and shapes from images. Instead of training a deep network from scratch, I implemented transfer learning using MobileNetV2, which demonstrated how pretrained models can achieve high accuracy even with limited data while reducing training time and computational cost.

Model evaluation became much clearer beyond simple accuracy metrics. By using confusion matrices along with precision, recall, and F1-score, I was able to analyze class-wise performance and identify any weak predictions. Visualizing training and validation accuracy and loss curves also helped in understanding learning behavior, convergence, and overfitting patterns.

I learned how overfitting can occur quickly when working with small or custom datasets and how techniques such as dropout, data augmentation, and transfer learning help mitigate this issue. Freezing pretrained layers and training only the classifier head proved highly effective for stable learning.

Building a real-time inference system introduced practical challenges such as unstable predictions, lighting variations, and background noise. Integrating MediaPipe for hand detection and cropping only the hand region before classification greatly improved model focus and accuracy. Implementing prediction smoothing across frames made the live webcam output more stable and usable.

This project also taught me how to structure a machine learning pipeline in a production-like manner — including dataset handling, model training, evaluation, visualization, saving models in modern Keras format, and loading them for real-time inference.

Overall, I gained a strong understanding of how computer vision models move from raw data collection to high-accuracy real-time applications, and how preprocessing, evaluation, and system optimization are just as important as the deep learning model itself.
