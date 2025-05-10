
# Hand Gesture Recognition using LSTM

## Project Overview

This project is focused on building a real-time hand gesture recognition system using computer vision and deep learning techniques. The system processes video input from a webcam to recognize hand gestures and interpret them as commands, enhancing Human-Computer Interaction (HCI). It has applications in sign language translation, assistive technology, smart home control, and gesture-based gaming interfaces.

## Objectives

- Recognize a wide range of hand gestures accurately.
- Ensure real-time performance with low latency.
- Maintain recognition quality under different lighting conditions and backgrounds.
- Optimize the system for computational efficiency and portability.
- Enable broad integration with various software applications and devices.

## Technologies Used

- Python
- OpenCV
- Mediapipe
- TensorFlow / Keras
- NumPy
- Scikit-learn
- Matplotlib

## System Requirements

### Hardware
- Laptop with an integrated webcam (720p or higher recommended)
- 16 GB RAM or higher
- Multi-core CPU (Intel i5/i7 or AMD Ryzen)
- Optional GPU for faster training and inference

### Software
- Operating System: Windows / macOS / Ubuntu
- Python 3.x
- IDE: VS Code, PyCharm, or Jupyter Notebook

## Dataset Preparation

- Hand gesture videos are recorded using a webcam.
- Mediapipe is used to extract hand keypoints.
- Preprocessing includes normalization, augmentation, and formatting for LSTM input.

## Methodology

1. Data Collection using webcam
2. Keypoint Extraction using Mediapipe
3. Preprocessing and augmentation
4. LSTM model design and training
5. Model evaluation using accuracy and loss metrics
6. Real-time gesture detection using webcam input

## Model Summary

- Architecture: LSTM layers followed by Dense layers
- Parameters: Approximately 596,675
- Accuracy: 91.2%
- Loss: 0.3186

## Features

- Real-time video stream processing
- Keypoint detection overlay
- Real-time gesture classification and display
- Confusion matrix and evaluation metrics for performance testing

## Applications

- Sign language recognition and translation
- Assistive input for users with speech or mobility impairments
- Touchless control in smart homes and public interfaces
- Gesture-controlled user interfaces in gaming and VR/AR environments

## Limitations

- Performance is sensitive to lighting and background conditions
- Gesture vocabulary is currently limited
- Requires relatively high hardware specifications for smooth real-time processing

## Future Enhancements

- Expand gesture dataset and vocabulary
- Improve robustness to environmental variations
- Develop mobile and cross-platform compatibility
- Explore more advanced deep learning architectures
- Improve UI and feedback mechanisms

## References

- Deepsign: Sign Language Detection Using Deep Learning  
- Recognition of Indian Sign Language Using Deep Learning  
- Real-Time Sign Language Recognition with CNNs  
- Systematic Review on Sign Language Recognition Systems

## Acknowledgements

Thanks to all mentors, contributors, and support staff who provided guidance and technical assistance throughout the development of this project.

## Installation Instructions

```bash
# Clone the repository
git clone https://github.com/your-username/hand-gesture-recognition.git
cd hand-gesture-recognition

# Install dependencies
pip install -r requirements.txt

# Run the application
python run_app.py
```
