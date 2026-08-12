# Object Detection with COCO

A fully browser-based COCO object detector using YOLO26N exported to ONNX and ONNX Runtime Web. There is no Python backend.

## Architecture

```text
GitHub Pages
    ↓
Browser
    ├── Webcam
    ├── ONNX Runtime Web
    └── YOLO26N ONNX
```

The current model is loaded from a public Hugging Face ONNX file:

```text
https://huggingface.co/zwh20081/yolo26-onnx/resolve/main/yolo26n.onnx
```

The referenced file is a 9.94 MB YOLO26N ONNX export. Verify the model source and license before commercial redistribution.

## Run locally

Do not open `index.html` with `file://`.

Run:

```bash
cd frontend
python3 -m http.server 5500
```

Open:

```text
http://localhost:5500/
```

The camera works on localhost.

## GitHub Pages

Upload everything inside `frontend/` to the GitHub Pages repository. No backend is required.

The site downloads the ONNX model when the detection page initializes.

## Model

YOLO26 supports official detection checkpoints from nano through x and supports ONNX export. YOLO26N is used here because it is much more suitable for browser-side inference than YOLO26X. The official YOLO26 documentation reports 40.9 mAP on COCO for YOLO26N and significantly lower CPU ONNX latency than the larger variants.

For maximum control, export your own official `yolo26n.pt` with:

```bash
yolo export model=yolo26n.pt format=onnx
```

Then host the resulting ONNX file from a public HTTPS location and replace `MODEL_URL` in `app.js`.

## Browser acceleration

The app uses ONNX Runtime Web and attempts WebGPU first when the browser exposes WebGPU, with WASM as a fallback.

## Tracking

The browser version includes selectable lightweight browser-side tracking logic for ByteTrack, BotSORT and OC-SORT controls. These are intentionally implemented without a server dependency. They are not a drop-in replacement for the full Ultralytics Python tracker implementations, especially BoT-SORT's ReID/camera-motion components.

## License

YOLO26 is distributed by Ultralytics under AGPL-3.0. The public ONNX file used by the default configuration should be reviewed against its source repository's licensing terms. If you intend to distribute or commercialize a public service/product using YOLO26, review Ultralytics' current licensing terms before deployment.
