# Object Detection with COCO

A browser-based real-time object detection application powered by **YOLO26N** and the **COCO dataset**. The application uses the user's webcam to perform object detection directly inside the browser, with no Python backend, server-side inference, or local model installation required.

The project is designed to be deployed as a static website, making it suitable for **GitHub Pages** and other static hosting platforms.

## Overview

**Object Detection with COCO** provides a simple interface for real-time webcam object detection using the 80 classes of the COCO dataset.

Users can:

- Select or deselect individual COCO classes
- Search through the COCO classes
- Start webcam-based object detection
- Enable or disable horizontal camera mirroring
- Switch between front and rear cameras on mobile devices
- View real-time camera FPS
- Select an object tracking method
- Run the entire detection pipeline directly in the browser
- Use the application on desktop and mobile devices

The project has two main pages:

1. `index.html` — object/class selection and setup
2. `detection.html` — real-time detection interface

## How It Works

The application does not send webcam frames to a Python server.

Instead, the complete inference pipeline runs locally inside the user's browser:

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
Optional object tracking
   ↓
Canvas overlay
   ↓
Bounding boxes + labels + confidence + tracker IDs
```

The YOLO26N ONNX model is loaded from a remote model repository when detection starts. This means the model does not need to be committed to the GitHub repository.

### Browser-side inference

The application uses **ONNX Runtime Web** to execute the ONNX model inside the browser.

When WebGPU is available, the application attempts to use WebGPU acceleration. If WebGPU is unavailable or cannot initialize the model, the application falls back to the WebAssembly execution provider.

This keeps the project completely client-side.

### Detection

The webcam frame is resized and prepared for the YOLO26N input.

The model returns detections containing:

```text
x1
y1
x2
y2
confidence
class ID
```

The application then:

1. Filters detections using the selected COCO classes.
2. Removes detections below the confidence threshold.
3. Converts model coordinates back to the webcam display coordinates.
4. Applies the horizontal mirror transformation when enabled.
5. Draws the bounding boxes and labels on the overlay canvas.

Each detection label contains:

```text
Tracker ID
Class name
Confidence score
```

When tracking is disabled, the tracker ID is omitted.

## Object Tracking

The detection interface includes three tracking options:

- **ByteTrack**
- **BoT-SORT**
- **OC-SORT**

The application performs tracking client-side after object detection.

Tracking is optional. Selecting **No Tracking** displays detections without assigning persistent object IDs.

> The browser implementations are lightweight client-side tracking implementations designed for this static web application. They are not intended to reproduce every feature of the full Python implementations of these tracking algorithms.

## COCO Dataset

The application uses the standard **80 object classes from the COCO dataset**.

Examples include:

- person
- bicycle
- car
- motorcycle
- airplane
- bus
- train
- truck
- boat
- bird
- cat
- dog
- horse
- backpack
- umbrella
- bottle
- chair
- couch
- laptop
- cell phone
- book
- clock
- scissors
- teddy bear
- suitcase

The complete list is included in `classes.js`.

## Main Technologies

### HTML5

HTML provides the structure of the application:

- Setup interface
- Detection interface
- Webcam video element
- Canvas overlay
- Mobile navigation menu
- Class selection controls

### CSS3

The interface uses separate stylesheets for desktop and mobile layouts:

```text
desktop.css
mobile.css
```

The desktop and mobile interfaces are intentionally handled separately so the mobile detection experience can behave differently without changing the desktop layout.

### JavaScript

JavaScript controls the complete application logic, including:

- Webcam access
- Camera switching
- Class selection
- Model loading
- Image preprocessing
- Browser inference
- Detection parsing
- Bounding-box rendering
- Mirroring
- FPS measurement
- Object tracking
- Mobile navigation
- Local preference storage

### ONNX Runtime Web

[ONNX Runtime Web](https://onnxruntime.ai/docs/get-started/with-javascript.html) executes the YOLO26N ONNX model directly in the browser.

The application can use:

```text
WebGPU
   ↓
GPU-accelerated inference
```

or:

```text
WebAssembly
   ↓
CPU/browser fallback
```

### YOLO26N

The detection model is **YOLO26N**, a lightweight YOLO26 variant intended to provide a practical balance between detection accuracy and inference speed.

The model is loaded remotely rather than stored inside the GitHub repository.

Model source:

[YOLO26N ONNX](https://huggingface.co/flotek/yolo26n-onnx)

### COCO

The model is trained for object detection using the **COCO dataset** and supports 80 object categories.

## Project Structure

```text
object-detection-with-coco/
│
├── frontend/
│   │
│   ├── index.html
│   ├── detection.html
│   │
│   ├── app.js
│   ├── classes.js
│   │
│   ├── desktop.css
│   └── mobile.css
│
└── README.md
```

### File Description

| File | Purpose |
|---|---|
| `index.html` | Main setup page |
| `detection.html` | Webcam detection interface |
| `app.js` | Main application and inference logic |
| `classes.js` | COCO 80-class definitions |
| `desktop.css` | Desktop-specific styling |
| `mobile.css` | Mobile-specific styling |
| `README.md` | Project documentation |

## Getting Started

### Requirements

You only need:

- A modern web browser
- A webcam/camera
- Internet access
- HTTPS when accessing the camera from a deployed website

No Python environment or Node.js installation is required to use the deployed application.

## Run Locally

Because browsers restrict webcam access from ordinary local files, run the frontend through a local web server.

For example, with Python's built-in static server:

```bash
cd object-detection-with-coco/frontend
python3 -m http.server 5500
```

Then open:

```text
http://localhost:5500
```

No Python packages are installed by this command. Python is only being used as a simple local HTTP server.

## Clone the Repository

Clone the GitHub repository:

```bash
git clone https://github.com/YOUR-USERNAME/YOUR-REPOSITORY.git
```

Enter the project directory:

```bash
cd YOUR-REPOSITORY
```

Open the `frontend` directory:

```bash
cd frontend
```

For local testing:

```bash
python3 -m http.server 5500
```

Then open:

```text
http://localhost:5500
```

Replace `YOUR-USERNAME/YOUR-REPOSITORY` with the actual GitHub repository path.

## GitHub Pages Deployment

This project is designed to work as a static GitHub Pages website.

There is no backend to deploy.

The basic deployment structure is:

```text
GitHub Repository
        ↓
GitHub Pages
        ↓
index.html
        ↓
detection.html
        ↓
Browser-side YOLO26N inference
```

### Deploying

1. Create a GitHub repository.
2. Upload the contents of the project.
3. Make sure `index.html` is available in the published directory.
4. Open the repository's **Settings**.
5. Open **Pages**.
6. Select the branch containing the website.
7. Select the appropriate folder, usually `/root`.
8. Save the GitHub Pages configuration.
9. Open the generated GitHub Pages URL.

The website does not require:

- Python hosting
- FastAPI
- Flask
- Node.js server
- GPU server
- Docker
- Database
- API server

## Camera Permissions

The browser must be allowed to access the camera.

On a deployed website, camera access requires a secure context:

```text
https://
```

GitHub Pages automatically provides HTTPS.

For local development, `localhost` is also treated as a secure context by modern browsers.

If camera access is denied, allow camera permissions for the website and reload the page.

## Mobile Support

The mobile interface is designed specifically for phone screens.

The mobile detection page:

- Uses the entire available screen for the camera
- Removes the desktop side panels
- Provides a hamburger control menu
- Supports front/rear camera switching
- Supports horizontal mirroring
- Provides tracking controls
- Provides FPS controls
- Provides all COCO class selections
- Makes the entire control menu scrollable

The desktop layout remains independent from the mobile layout.

## Performance

Browser inference performance depends on:

- GPU capability
- WebGPU support
- Browser implementation
- CPU performance
- Camera resolution
- Device memory
- Mobile thermal limitations

YOLO26N is used instead of larger YOLO26 variants because the goal is real-time browser inference rather than maximum model capacity.

The application also separates the camera display loop from the model inference loop so that slow inference does not unnecessarily freeze the entire user interface.

## Privacy

Webcam frames are processed directly in the browser.

The application does not require a Python backend to receive or process camera frames.

The YOLO26N model is downloaded from its remote model host when required, but webcam inference itself occurs locally in the browser.

## Future Development

Possible future improvements include:

- More efficient browser preprocessing
- WebGPU-specific optimization
- Improved mobile inference performance
- More advanced tracking implementations
- Detection statistics
- Object counting
- Detection history
- Screenshot capture
- Video recording
- Performance monitoring
- Additional YOLO26 model variants
- Custom model support
- Additional computer vision tasks

## License

This project contains and uses third-party software and model components. Check the individual licenses of the dependencies and model before redistributing or using the project commercially.

In particular, review the current licensing terms of the YOLO26/Ultralytics model and the ONNX model being used by the application before commercial distribution.

## Acknowledgements

This project builds upon:

- [Ultralytics](https://github.com/ultralytics/ultralytics)
- [YOLO26](https://docs.ultralytics.com/models/yolo26/)
- [ONNX Runtime Web](https://onnxruntime.ai/)
- [Hugging Face](https://huggingface.co/)
- [COCO Dataset](https://cocodataset.org/)

---

## Author

**Md. Abu Horaira Al Rezon**

Built as a browser-based computer vision project focused on bringing real-time YOLO object detection to static web hosting without requiring a dedicated inference backend.
