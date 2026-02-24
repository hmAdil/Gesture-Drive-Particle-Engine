# Gesture-Driven 3D Space Control System

## Overview

This project integrates real-time hand tracking with a 3D rendering environment built using Pygame.

The system captures live video input, detects hand landmarks using MediaPipe, and processes landmark coordinates in real time. A 3D scene is rendered using Pygame, establishing the foundation for future gesture-based spatial control.

The current implementation focuses on establishing a stable computer vision pipeline and integrating it with a real-time rendering engine.

---

## Motivation

The project explores how computer vision can be connected to a 3D rendering system to enable gesture-based interaction. Rather than relying on traditional input devices, the long-term goal is to control spatial transformations through hand movements.

At this stage, the emphasis is on establishing reliable hand tracking and rendering integration.

---

## System Architecture

1. Video capture using OpenCV
2. Hand landmark detection using MediaPipe
3. Extraction of landmark coordinate data
4. Processing of landmark positions
5. 3D scene rendering using Pygame

The architecture separates vision processing and rendering logic, allowing interaction logic to be added incrementally.

---

## Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Pygame

---

## Current Capabilities

- Real-time webcam capture
- Hand landmark detection and tracking
- Real-time visual feedback of detected hand landmarks
- Extraction of 2D landmark coordinates
- Rendering of a 3D environment using Pygame
- Modular structure for future gesture integration

---

## Limitations

- No gesture-to-transformation mapping implemented yet
- No depth-aware control
- Interaction logic still under development

---

## Future Improvements

The system is designed to be extended with:

- Rule-based gesture-to-transformation mapping
- Continuous object manipulation based on hand position
- Depth-aware control for true 3D interaction
- Machine learning–based gesture classification
- Personalized gesture training
- Integration with more advanced rendering frameworks

---

## How to Run

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/Gesture-Drive-Particle-Engine.git
cd Gesture-Drive-Particle-Engine
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Application

```bash
python main.py
```
