# Real-Time Face Detection & Tracking

A Python-based computer vision application for real-time face, eye, and smile detection using OpenCV and Haar Cascades. This project features interactive keyboard controls, privacy mode (face blurring), and on-screen metrics like FPS and face count.

## Features

- **Real-Time Face Detection:** Accurate detection of multiple faces via webcam.
- **Eye & Smile Detection:** Toggleable sub-features to detect eyes and smiles within detected faces.
- **Privacy Mode:** Easily toggle a blur filter over detected faces instead of drawing bounding boxes.
- **On-Screen Metrics:** Displays the live Frames Per Second (FPS) and the current count of detected faces.
- **Snapshot Capture:** Save the current webcam frame as an image file directly to your project folder with a single keystroke.

## Prerequisites

- Python 3.x
- OpenCV (`cv2`)

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Sanjoy-ui/face_detection.git
   cd face_detection
   ```

2. **Create a virtual environment (Recommended):**
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows use: venv\Scripts\activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Ensure your virtual environment is activated, navigate to the source directory, and run the script:

```bash
cd Face-Detection
python3 face-reco.py
```

### Keyboard Controls

Once the webcam window is active, you can use the following keyboard shortcuts to interact with the application:

| Key | Action |
| --- | --- |
| `s` | **Take a Snapshot:** Saves the current frame as a `.jpg` file in the directory. |
| `b` | **Toggle Blur:** Turns privacy mode on/off (blurs detected faces). |
| `e` | **Toggle Eye Detection:** Turns eye detection within faces on/off. |
| `m` | **Toggle Smile Detection:** Turns smile detection within faces on/off. |
| `ESC` | **Quit:** Exits the application and closes the webcam stream. |

## How it Works

The core of this application relies on **Haar Feature-based Cascade Classifiers**. It utilizes OpenCV's pre-trained `.xml` classifiers to quickly identify facial features in a live video stream in a highly efficient manner.

---
*Built by [Sanjoy-ui](https://github.com/Sanjoy-ui)*
