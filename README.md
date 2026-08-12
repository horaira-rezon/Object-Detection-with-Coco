# Object Detection with COCO

A browser-based real-time object detection application powered by **YOLO26N** and the **COCO dataset**. The application uses the user's webcam to perform object detection directly inside the browser, with no Python backend, server-side inference, or local model installation required.

## Overview

**Object Detection with COCO** provides a simple interface for real-time webcam object detection using the 80 classes of the COCO dataset. Users can:

- Select or deselect individual COCO classes
- Start webcam-based object detection
- Select an object tracking algorithm
- Run the entire pipeline directly in the browser

## How It Works

The YOLO26N ONNX model is loaded from a remote model repository when detection starts. The application does not send frames to a Python server. Instead, the pipeline runs locally inside the user's browser:

```text
Webcam
   ↓
Browser
   ↓
Image preprocessing
   ↓
YOLO26N ONNX model
   ↓
Object detections
   ↓
COCO class filtering
   ↓
OBbject tracking
   ↓
Canvas overlay
   ↓
Bounding boxes
```

## Technologies

- HTML5
- CSS3
- JavaScript
- YOLO26N
- ONNX
- ONNX Runtime Web
- WebGPU
- WebAssembly
- COCO dataset
- ByteTrack
- BoT-SORT
- OC-SORT
- GitHub Pages

## Project Structure

```text
object-detection-with-coco
├── frontend/
│   ├── index.html
│   ├── detection.html
│   ├── app.js
│   ├── classes.js
│   ├── desktop.css
│   └── mobile.css
└── README.md
```

## Run Locally

Clone the repository:

Using HTTPS:
```bash
git clone https://github.com/horaira-rezon/Object-Detection-with-Coco.git
cd local-folder
```

Using SSH:
```bash
git clone git@github.com:horaira-rezon/Object-Detection-with-Coco.git
cd local-folder
```

Start a local server:
```bash
cd frontend
python3 -m http.server 5500
```

Then, open:
```text
http://localhost:5500
```

## Privacy

Webcam frames are processed locally in the user's browser. The application does not require a server to receive webcam frames.
The YOLO26N model is downloaded from its remote model host when required.

## Performance

Performance depends on the user's device, browser, GPU, WebGPU support and available memory.
YOLO26N is used because it is better suited to browser-based real-time inference than larger YOLO26 variants.

## Model

YOLO26N is loaded remotely from: https://huggingface.co/flotek/yolo26n-onnx
No model file needs to be committed to the repository.

## License

This project uses third-party software and model components. Review the applicable licenses before commercial redistribution.
Relevant projects:
- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [YOLO26](https://docs.ultralytics.com/models/yolo26/)
- [ONNX Runtime](https://onnxruntime.ai/)
- [Hugging Face](https://huggingface.co/)
- [COCO](https://cocodataset.org/)