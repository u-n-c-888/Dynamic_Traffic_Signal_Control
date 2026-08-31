# Dynamic Traffic Signal System 🚦

A computer vision based traffic signal system that dynamically allocates green-signal time according to the number of vehicles detected on four roads.

## Project Overview

Traditional traffic signals often use fixed timings. This project uses **YOLOv8** and **OpenCV** to detect vehicles in road images and calculate a dynamic green-signal time.

An emergency priority feature is also included. For the current demo, ambulance presence is provided manually so the project can be demonstrated without requiring a custom ambulance detection model.

## Features

- Detects vehicles using YOLOv8.
- Counts cars, motorcycles, buses and trucks.
- Calculates dynamic green-signal time.
- Roads with more traffic receive more green time.
- Roads with zero vehicles receive the minimum green time.
- Ambulance road receives first priority.
- Supports multiple ambulance roads.
- Displays YOLO detections using OpenCV.

## Technologies

- Python
- OpenCV
- YOLOv8
- Ultralytics
- Computer Vision

## Project Structure

```text
Dynamic-Traffic-Signal/
│
├── main.py
├── requirements.txt
├── README.md
├── .gitignore
└── images/
    ├── road1.png
    ├── road2.png
    ├── road3.png
    └── road4.png
```

## Installation

Install Python 3.10 or 3.11.

Open a terminal inside the project folder and run:

```bash
pip install -r requirements.txt
```

## Input Images

Place the four road images inside the `images` folder:

```text
images/road1.png
images/road2.png
images/road3.png
images/road4.png
```

The YOLO model `yolov8n.pt` is downloaded automatically the first time the program runs.

## Ambulance Priority

Currently, ambulance presence is supplied manually in `main.py`.

For example:

```python
AMBULANCE_ROADS = ["Road 4"]
```

For two ambulance roads:

```python
AMBULANCE_ROADS = ["Road 2", "Road 4"]
```

If there is no ambulance:

```python
AMBULANCE_ROADS = []
```

When multiple ambulance roads are present, the ambulance roads are placed before normal traffic roads. Among ambulance roads, the road with more vehicles is placed first.

## Dynamic Signal Timing

The project uses a simple proportional timing model:

```text
Minimum green time = 5 seconds
Maximum green time = 30 seconds
```

If a road has no vehicles, it receives 5 seconds.

For normal roads, the green time increases according to traffic density.

An ambulance road receives 30 seconds and first priority.

## Example

Suppose:

```text
Road 1 → 40 vehicles
Road 2 → 67 vehicles
Road 3 → 20 vehicles
Road 4 → 80 vehicles
```

The priority becomes:

```text
1. Road 4
2. Road 2
3. Road 1
4. Road 3
```

If Road 4 is the ambulance road:

```text
AMBULANCE → Road 4 → FIRST PRIORITY
```

## Run the Project

From the project directory:

```bash
python main.py
```

## Future Enhancements

- Automatic ambulance detection using a custom YOLO model.
- OCR-based detection of the word "AMBULANCE".
- Live CCTV/video input instead of static images.
- Real traffic-light hardware using Arduino/Raspberry Pi.
- Web dashboard for live traffic monitoring.
- Database storage for traffic statistics.

## Interview Explanation

> "My project is a dynamic traffic signal system using computer vision. I use YOLOv8 to detect and count vehicles on four roads. Based on traffic density, the system dynamically calculates green-signal time instead of using a fixed timer. I also added an emergency priority mechanism for ambulances. In the current prototype, ambulance presence is manually configured, and automatic ambulance recognition is a planned enhancement."

## Note

This is a prototype for demonstrating the traffic-priority algorithm using road images. It is not intended to directly control real-world traffic signals.
