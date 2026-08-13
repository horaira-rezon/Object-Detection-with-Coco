const MODEL_FP16_URL = "https://huggingface.co/flotek/yolo26n-onnx/resolve/main/model.onnx?download=true";
const MODEL_FP32_URL = "https://huggingface.co/onnx-community/yolo26n-ONNX/resolve/main/onnx/model.onnx?download=true";
const INPUT_SIZE = 640;
const CONFIDENCE_THRESHOLD = .18;
const state = {
    selected: new Set(COCO_CLASSES.map((_, i) => i)),
    tracker: "none",
    fps: true,
    mirror: false,
    cameraFacing: "environment",
    session: null,
    videoStream: null,
    running: false,
    processing: false,
    lastFrameTime: 0,
    fpsFrames: 0,
    fpsValue: 0,
    tracks: [],
    nextTrackId: 1,
    preprocessCanvas: null,
    preprocessContext: null,
    inferenceTimer: null,
    inferenceBusy: false,
    inferenceInterval: 120,
    session: null,
    inputName: null,
    outputName: null,
    preprocessCanvas: null,
    preprocessContext: null,
    inferenceTimer: null,
    inferenceBusy: false,
    inferenceInterval: 100,
    modelFormat: "ultralytics",
    outputNames: []
};

function loadPreferences() {
    try {
        const saved = JSON.parse(localStorage.getItem("cocoSelectedClasses"));
        if (Array.isArray(saved)) state.selected = new Set(saved.filter(v => Number.isInteger(v) && v >= 0 && v < 80))
    } catch {}
    const tracker = localStorage.getItem("cocoTracker");
    if (["none", "bytetrack", "botsort", "ocsort"].includes(tracker)) state.tracker = tracker;
    state.mirror = localStorage.getItem("cocoMirror") === "1";
    const facing = localStorage.getItem("cocoCameraFacing");
    if (facing === "user" || facing === "environment") state.cameraFacing = facing
}

function savePreferences() {
    localStorage.setItem("cocoSelectedClasses", JSON.stringify(Array.from(state.selected)));
    localStorage.setItem("cocoTracker", state.tracker);
    localStorage.setItem("cocoMirror", state.mirror ? "1" : "0");
    localStorage.setItem("cocoCameraFacing", state.cameraFacing)
}

function createSetupItem(id, container) {
    const label = document.createElement("label");
    label.className = "class-item";
    label.innerHTML = `<input type="checkbox" data-class-id="${id}" ${state.selected.has(id) ? "checked" : ""}><span>${COCO_CLASSES[id]}</span>`;
    label.querySelector("input").addEventListener("change", e => {
        if (e.target.checked) state.selected.add(id);
        else state.selected.delete(id);
        savePreferences();
        updateSelectionCount()
    });
    container.appendChild(label)
}

function buildSetupClasses(filter = "") {
    const grid = document.getElementById("class-grid");
    if (!grid) return;
    grid.innerHTML = "";
    const query = filter.toLowerCase();
    COCO_CLASSES.forEach((name, id) => {
        if (name.includes(query)) createSetupItem(id, grid)
    });
    updateSelectionCount()
}

function updateSelectionCount() {
    const el = document.getElementById("selection-count");
    if (el) el.textContent = `${state.selected.size} of 80 classes selected`
}

function createDetectItem(id, container) {
    const row = document.createElement("label");
    row.className = "detect-class-item";
    row.innerHTML = `<input type="checkbox" data-class-id="${id}" ${state.selected.has(id) ? "checked" : ""}><span>${COCO_CLASSES[id]}</span>`;
    row.querySelector("input").addEventListener("change", e => {
        if (e.target.checked) state.selected.add(id);
        else state.selected.delete(id);
        savePreferences()
    });
    container.appendChild(row)
}

function buildDetectClasses(container, filter = "") {
    if (!container) return;
    container.innerHTML = "";
    const query = filter.toLowerCase();
    COCO_CLASSES.forEach((name, id) => {
        if (name.includes(query)) createDetectItem(id, container)
    })
}

function syncDetectLists() {
    buildDetectClasses(document.getElementById("detect-class-list"), document.getElementById("detect-search-input")?.value || "");
    buildDetectClasses(document.getElementById("mobile-class-list"), document.getElementById("mobile-search-input")?.value || "")
}

function setAll(value) {
    state.selected = value ? new Set(COCO_CLASSES.map((_, i) => i)) : new Set();
    savePreferences();
    if (document.getElementById("class-grid")) buildSetupClasses(document.getElementById("class-search").value);
    syncDetectLists()
}

function setTracker(value) {
    if (!["none", "bytetrack", "botsort", "ocsort"].includes(value)) return;
    state.tracker = value;
    state.tracks = [];
    savePreferences();
    document.querySelectorAll(".tracker-button").forEach(b => b.classList.toggle("active", (b.dataset.tracker || b.dataset.mobileTracker) === value))
}

function setFps(value) {
    state.fps = value;
    const display = document.getElementById("fps-display");
    if (display) display.style.display = value ? "block" : "none";
    const a = document.getElementById("fps-toggle"),
        b = document.getElementById("mobile-fps-toggle");
    if (a) a.checked = value;
    if (b) b.checked = value
}

function setMirror(value) {
    state.mirror = Boolean(value);
    savePreferences();
    const frame = document.querySelector(".camera-frame");
    if (frame) frame.classList.toggle("mirrored", state.mirror);
    const a = document.getElementById("mirror-toggle"),
        b = document.getElementById("mobile-mirror-toggle");
    if (a) a.checked = state.mirror;
    if (b) b.checked = state.mirror
}

function updateCameraButtons() {
    document.querySelectorAll("[data-camera-facing]").forEach(button => button.classList.toggle("active", button.dataset.cameraFacing === state.cameraFacing))
}

function setCameraFacing(facing) {
    if (facing !== "user" && facing !== "environment") return;
    if (facing === state.cameraFacing && state.videoStream) {
        updateCameraButtons();
        return
    }
    startCamera(facing)
}

function setModelStatus(text, ready = false) {
    const dot = document.getElementById("model-dot"),
        label = document.getElementById("model-status"),
        stateText = document.getElementById("stream-state"),
        message = document.getElementById("camera-message");
    if (dot) dot.classList.toggle("connected", ready);
    if (label) label.textContent = text;
    if (stateText) stateText.textContent = ready ? "DETECTION READY" : text.toUpperCase();
    if (message && ready) message.textContent = ""
}

async function createSession() {
    if (!window.ort) throw new Error("ONNX Runtime Web did not load.");
    ort.env.wasm.numThreads = Math.min(4, Math.max(1, navigator.hardwareConcurrency || 2));
    ort.env.wasm.simd = true;
    const isMobile = /Android|iPhone|iPad|iPod/i.test(navigator.userAgent) || window.matchMedia("(max-width:700px)").matches;
    const primaryModel = isMobile ? MODEL_FP32_URL : MODEL_FP16_URL;
    state.modelFormat = isMobile ? "yolos" : "ultralytics";
    let lastError = null;
    if (navigator.gpu) {
        try {
            state.session = await ort.InferenceSession.create(primaryModel, {
                executionProviders: ["webgpu"],
                graphOptimizationLevel: "all",
                enableMemPattern: true,
                enableCpuMemArena: true
            });
            state.inputName = state.session.inputNames[0];
            state.outputName = state.session.outputNames[0];
            state.outputNames = [...state.session.outputNames];
            return state.session
        } catch (error) {
            lastError = error;
            console.warn("WebGPU load failed, trying WASM", error)
        }
    }
    try {
        state.session = await ort.InferenceSession.create(primaryModel, {
            executionProviders: ["wasm"],
            graphOptimizationLevel: "all",
            enableMemPattern: true,
            enableCpuMemArena: true
        });
        state.inputName = state.session.inputNames[0];
        state.outputName = state.session.outputNames[0];
        state.outputNames = [...state.session.outputNames];
        return state.session
    } catch (error) {
        throw new Error(`YOLO26N could not initialize. ${lastError?.message || error.message}`)
    }
}

function prepareInputs(video) {
    if (!state.preprocessCanvas) {
        state.preprocessCanvas = document.createElement("canvas");
        state.preprocessCanvas.width = INPUT_SIZE;
        state.preprocessCanvas.height = INPUT_SIZE;
        state.preprocessContext = state.preprocessCanvas.getContext("2d", {
            willReadFrequently: true,
            alpha: false
        })
    }
    const c = state.preprocessContext;
    c.fillStyle = "#808080";
    c.fillRect(0, 0, INPUT_SIZE, INPUT_SIZE);
    const scale = Math.min(INPUT_SIZE / video.videoWidth, INPUT_SIZE / video.videoHeight),
        w = Math.round(video.videoWidth * scale),
        h = Math.round(video.videoHeight * scale),
        x = Math.floor((INPUT_SIZE - w) / 2),
        y = Math.floor((INPUT_SIZE - h) / 2);
    c.drawImage(video, 0, 0, video.videoWidth, video.videoHeight, x, y, w, h);
    const image = c.getImageData(0, 0, INPUT_SIZE, INPUT_SIZE).data,
        plane = INPUT_SIZE * INPUT_SIZE,
        data = new Float32Array(plane * 3);
    for (let i = 0, p = 0; i < image.length; i += 4, p++) {
        data[p] = image[i] / 255;
        data[plane + p] = image[i + 1] / 255;
        data[plane * 2 + p] = image[i + 2] / 255
    }
    return {
        tensor: new ort.Tensor("float32", data, [1, 3, INPUT_SIZE, INPUT_SIZE]),
        scale,
        x,
        y
    }
}

function sigmoid(value) {
    return 1 / (1 + Math.exp(-value))
}

function parseOutput(output, meta, videoWidth, videoHeight) {
    if (state.modelFormat === "yolos") {
        const logits = output.logits || output[state.outputNames.find(name => /logits/i.test(name))];
        const predBoxes = output.pred_boxes || output[state.outputNames.find(name => /pred_boxes/i.test(name))];
        if (!logits?.data || !predBoxes?.data) return [];
        const ld = logits.dims || [],
            bd = predBoxes.dims || [];
        const count = ld.length === 3 ? ld[1] : 300,
            classes = ld.length === 3 ? ld[2] : 80;
        const detections = [];
        for (let i = 0; i < count; i++) {
            let maxScore = 0,
                maxClass = 0;
            for (let j = 0; j < Math.min(classes, 80); j++) {
                const score = sigmoid(Number(logits.data[i * classes + j]));
                if (score > maxScore) {
                    maxScore = score;
                    maxClass = j
                }
            }
            if (!Number.isFinite(maxScore) || maxScore < CONFIDENCE_THRESHOLD || maxClass < 0 || maxClass >= COCO_CLASSES.length || !state.selected.has(maxClass)) continue;
            const cx = Number(predBoxes.data[i * 4]),
                cy = Number(predBoxes.data[i * 4 + 1]),
                w = Number(predBoxes.data[i * 4 + 2]),
                h = Number(predBoxes.data[i * 4 + 3]);
            const x1 = (cx - w / 2) * INPUT_SIZE,
                y1 = (cy - h / 2) * INPUT_SIZE,
                x2 = (cx + w / 2) * INPUT_SIZE,
                y2 = (cy + h / 2) * INPUT_SIZE;
            const vx1 = Math.max(0, Math.min(videoWidth, (x1 - meta.x) / meta.scale)),
                vy1 = Math.max(0, Math.min(videoHeight, (y1 - meta.y) / meta.scale)),
                vx2 = Math.max(0, Math.min(videoWidth, (x2 - meta.x) / meta.scale)),
                vy2 = Math.max(0, Math.min(videoHeight, (y2 - meta.y) / meta.scale));
            if (vx2 <= vx1 || vy2 <= vy1) continue;
            detections.push({
                box: [vx1, vy1, vx2, vy2],
                conf: maxScore,
                classId: maxClass,
                label: COCO_CLASSES[maxClass]
            })
        }
        return detections
    }
    const tensor = output instanceof ort.Tensor ? output : output[state.outputName] || output[Object.keys(output)[0]];
    if (!tensor?.data) return [];
    const data = tensor.data,
        dims = tensor.dims || [],
        count = dims.length === 3 ? dims[1] : Math.floor(data.length / 6),
        detections = [];
    for (let i = 0; i < count; i++) {
        const o = i * 6,
            x1 = Number(data[o]),
            y1 = Number(data[o + 1]),
            x2 = Number(data[o + 2]),
            y2 = Number(data[o + 3]),
            conf = Number(data[o + 4]),
            classId = Math.round(Number(data[o + 5]));
        if (!Number.isFinite(conf) || conf < CONFIDENCE_THRESHOLD || classId < 0 || classId >= COCO_CLASSES.length || !state.selected.has(classId)) continue;
        const bx1 = Math.max(0, Math.min(INPUT_SIZE, x1)),
            by1 = Math.max(0, Math.min(INPUT_SIZE, y1)),
            bx2 = Math.max(0, Math.min(INPUT_SIZE, x2)),
            by2 = Math.max(0, Math.min(INPUT_SIZE, y2));
        const vx1 = Math.max(0, Math.min(videoWidth, (bx1 - meta.x) / meta.scale)),
            vy1 = Math.max(0, Math.min(videoHeight, (by1 - meta.y) / meta.scale)),
            vx2 = Math.max(0, Math.min(videoWidth, (bx2 - meta.x) / meta.scale)),
            vy2 = Math.max(0, Math.min(videoHeight, (by2 - meta.y) / meta.scale));
        if (vx2 <= vx1 || vy2 <= vy1) continue;
        detections.push({
            box: [vx1, vy1, vx2, vy2],
            conf,
            classId,
            label: COCO_CLASSES[classId]
        })
    }
    return detections
}

function iou(a, b) {
    const x1 = Math.max(a[0], b[0]),
        y1 = Math.max(a[1], b[1]),
        x2 = Math.min(a[2], b[2]),
        y2 = Math.min(a[3], b[3]),
        w = Math.max(0, x2 - x1),
        h = Math.max(0, y2 - y1),
        inter = w * h,
        areaA = Math.max(0, a[2] - a[0]) * Math.max(0, a[3] - a[1]),
        areaB = Math.max(0, b[2] - b[0]) * Math.max(0, b[3] - b[1]);
    return inter / (areaA + areaB - inter + 1e-6)
}

function center(box) {
    return [(box[0] + box[2]) / 2, (box[1] + box[3]) / 2]
}

function distance(a, b) {
    return Math.hypot(a[0] - b[0], a[1] - b[1])
}

function trackDetections(detections) {
    if (state.tracker === "none") {
        state.tracks = [];
        return detections.map(d => ({
            ...d,
            trackId: null
        }))
    }
    const config = {
            bytetrack: {
                min: .35,
                iou: .28,
                maxAge: 900
            },
            botsort: {
                min: .25,
                iou: .25,
                maxAge: 1000
            },
            ocsort: {
                min: .25,
                iou: .2,
                maxAge: 900
            }
        } [state.tracker],
        now = performance.now(),
        candidates = detections.filter(d => d.conf >= config.min),
        used = new Set(),
        output = [];
    for (const track of state.tracks) {
        let best = -1,
            bestScore = 0;
        for (let i = 0; i < candidates.length; i++) {
            const d = candidates[i];
            if (used.has(i) || d.classId !== track.classId) continue;
            const overlap = iou(track.box, d.box),
                diag = Math.hypot(track.box[2] - track.box[0], track.box[3] - track.box[1]),
                motion = state.tracker === "ocsort" ? Math.min(1, distance(center(track.box), center(d.box)) / (diag + 1e-6)) : 0,
                score = overlap * (1 - motion * .35);
            if (score > bestScore) {
                bestScore = score;
                best = i
            }
        }
        if (best >= 0 && bestScore >= config.iou) {
            const d = candidates[best];
            used.add(best);
            track.box = d.box;
            track.conf = d.conf;
            track.lastSeen = now;
            output.push({
                ...d,
                trackId: track.id
            })
        }
    }
    for (let i = 0; i < candidates.length; i++) {
        if (used.has(i)) continue;
        const d = candidates[i],
            track = {
                id: state.nextTrackId++,
                box: d.box,
                classId: d.classId,
                lastSeen: now
            };
        state.tracks.push(track);
        output.push({
            ...d,
            trackId: track.id
        })
    }
    state.tracks = state.tracks.filter(t => now - t.lastSeen <= config.maxAge);
    return output
}

function resizeCanvas() {
    const canvas = document.getElementById("overlay");
    if (!canvas) return;
    const rect = canvas.getBoundingClientRect();
    canvas.width = Math.max(1, Math.round(rect.width * devicePixelRatio));
    canvas.height = Math.max(1, Math.round(rect.height * devicePixelRatio))
}

function drawDetections(detections, videoWidth, videoHeight) {
    const canvas = document.getElementById("overlay"),
        ctx = canvas.getContext("2d"),
        rect = canvas.getBoundingClientRect(),
        sx = rect.width / videoWidth,
        sy = rect.height / videoHeight;
    ctx.setTransform(devicePixelRatio, 0, 0, devicePixelRatio, 0, 0);
    ctx.clearRect(0, 0, rect.width, rect.height);
    for (const d of detections) {
        const [x1, y1, x2, y2] = d.box, rx1 = state.mirror ? videoWidth - x2 : x1, rx2 = state.mirror ? videoWidth - x1 : x2, x = rx1 * sx, y = y1 * sy, w = (rx2 - rx1) * sx, h = (y2 - y1) * sy, color = `hsl(${(d.classId * 47) % 360} 85% 65%)`;
        ctx.strokeStyle = color;
        ctx.lineWidth = 2;
        ctx.strokeRect(x, y, w, h);
        const text = `${d.trackId !== null ? d.trackId + " " : ""}${d.label} ${(d.conf * 100).toFixed(1)}%`;
        ctx.font = `${Math.max(10, Math.min(13, rect.width / 95))}px monospace`;
        const m = ctx.measureText(text),
            th = 17,
            ly = Math.max(0, y - th - 4);
        ctx.fillStyle = color;
        ctx.fillRect(x, ly, m.width + 10, th + 4);
        ctx.fillStyle = "#141414";
        ctx.fillText(text, x + 5, ly + th)
    }
}
async function startCamera(facing = state.cameraFacing) {
    const message = document.getElementById("camera-message"),
        stateText = document.getElementById("stream-state");
    if (state.videoStream) state.videoStream.getTracks().forEach(t => t.stop());
    if (!navigator.mediaDevices?.getUserMedia) {
        message.textContent = "This browser does not provide camera access.";
        return
    }
    if (!window.isSecureContext && location.hostname !== "localhost" && location.hostname !== "127.0.0.1") {
        message.textContent = "Camera access requires HTTPS.";
        stateText.textContent = "HTTPS REQUIRED";
        return
    }
    message.textContent = "Requesting camera access...";
    stateText.textContent = "REQUESTING CAMERA";
    try {
        let stream;
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: {
                        exact: facing
                    },
                    width: {
                        ideal: 1280
                    },
                    height: {
                        ideal: 720
                    }
                },
                audio: false
            })
        } catch (error) {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: facing,
                    width: {
                        ideal: 1280
                    },
                    height: {
                        ideal: 720
                    }
                },
                audio: false
            })
        }
        state.videoStream = stream;
        state.cameraFacing = facing;
        savePreferences();
        const video = document.getElementById("camera");
        video.srcObject = stream;
        await video.play();
        message.textContent = "";
        stateText.textContent = facing === "user" ? "FRONT CAMERA" : "REAR CAMERA";
        updateCameraButtons();
        resizeCanvas()
    } catch (error) {
        message.textContent = error.name === "NotAllowedError" ? "Camera permission was denied." : error.name === "NotFoundError" ? "The selected camera was not found." : error.name === "NotReadableError" ? "The camera is already being used by another application." : "Camera access failed.";
        stateText.textContent = "CAMERA ERROR"
    }
}

function startFPSLoop() {
    state.lastFrameTime = performance.now();
    state.fpsFrames = 0;
    const tick = now => {
        const video = document.getElementById("camera");
        if (video && video.readyState >= 2) {
            state.fpsFrames++;
            if (now - state.lastFrameTime >= 1000) {
                state.fpsValue = state.fpsFrames * 1000 / (now - state.lastFrameTime);
                state.fpsFrames = 0;
                state.lastFrameTime = now;
                const fps = document.getElementById("fps-display");
                if (fps) fps.textContent = `FPS: ${state.fpsValue.toFixed(1)}`
            }
        }
        requestAnimationFrame(tick)
    };
    requestAnimationFrame(tick)
}
async function runInferenceLoop() {
    if (!state.running || !state.session || state.inferenceTimer !== null) return;
    const run = async () => {
        state.inferenceTimer = null;
        if (!state.running || !state.session) return;
        const video = document.getElementById("camera");
        if (video.readyState >= 2 && !state.inferenceBusy) {
            state.inferenceBusy = true;
            try {
                const prepared = prepareInputs(video),
                    output = await state.session.run({
                        [state.inputName]: prepared.tensor
                    }),
                    detections = parseOutput(output, prepared, video.videoWidth, video.videoHeight),
                    tracked = trackDetections(detections);
                drawDetections(tracked, video.videoWidth, video.videoHeight);
                const streamState = document.getElementById("stream-state");
                if (streamState) streamState.textContent = tracked.length ? `DETECTION READY · ${tracked.length} OBJECT${tracked.length === 1 ? "" : "S"}` : "DETECTION READY"
            } catch (error) {
                console.error("YOLO26 inference error:", error);
                const message = document.getElementById("camera-message");
                if (message) message.textContent = `Inference error: ${error.message}`
            } finally {
                state.inferenceBusy = false
            }
        }
        if (state.running) state.inferenceTimer = setTimeout(run, state.inferenceInterval)
    };
    state.inferenceTimer = setTimeout(run, 0)
}

function stopDetection() {
    state.running = false;
    if (state.inferenceTimer !== null) {
        clearTimeout(state.inferenceTimer);
        state.inferenceTimer = null
    }
    if (state.videoStream) state.videoStream.getTracks().forEach(t => t.stop());
    state.videoStream = null;
    state.session = null;
    state.inputName = null;
    state.outputName = null;
    state.outputNames = [];
    state.inferenceBusy = false
}
async function initializeDetection() {
    loadPreferences();
    syncDetectLists();
    setTracker(state.tracker);
    setMirror(state.mirror);
    setFps(true);
    updateCameraButtons();
    document.getElementById("back-button").onclick = () => {
        stopDetection();
        location.href = "index.html"
    };
    document.getElementById("detect-select-all").onclick = () => setAll(true);
    document.getElementById("detect-clear-all").onclick = () => setAll(false);
    document.getElementById("mobile-select-all").onclick = () => setAll(true);
    document.getElementById("mobile-clear-all").onclick = () => setAll(false);
    document.getElementById("detect-search-input").addEventListener("input", e => buildDetectClasses(document.getElementById("detect-class-list"), e.target.value));
    document.getElementById("mobile-search-input").addEventListener("input", e => buildDetectClasses(document.getElementById("mobile-class-list"), e.target.value));
    document.querySelectorAll("[data-tracker]").forEach(b => b.addEventListener("click", () => setTracker(b.dataset.tracker)));
    document.querySelectorAll("[data-mobile-tracker]").forEach(b => b.addEventListener("click", () => setTracker(b.dataset.mobileTracker)));
    document.getElementById("fps-toggle").addEventListener("change", e => setFps(e.target.checked));
    document.getElementById("mobile-fps-toggle").addEventListener("change", e => setFps(e.target.checked));
    document.querySelectorAll("[data-camera-facing]").forEach(b => b.addEventListener("click", () => setCameraFacing(b.dataset.cameraFacing)));
    document.getElementById("mirror-toggle").addEventListener("change", e => setMirror(e.target.checked));
    document.getElementById("mobile-mirror-toggle").addEventListener("change", e => setMirror(e.target.checked));
    document.getElementById("menu-button").addEventListener("click", () => document.getElementById("mobile-menu").classList.toggle("open"));
    window.addEventListener("resize", resizeCanvas);
    startFPSLoop();
    startCamera();
    try {
        await createSession();
        setModelStatus(navigator.gpu ? "YOLO26N WebGPU ready" : "YOLO26N ready", true);
        state.running = true;
        runInferenceLoop()
    } catch (error) {
        setModelStatus("YOLO26N failed to load");
        document.getElementById("camera-message").textContent = `YOLO26N could not be loaded: ${error.message}`
    }
}

function initSetupPage() {
    loadPreferences();
    buildSetupClasses();
    document.getElementById("class-search").addEventListener("input", e => buildSetupClasses(e.target.value));
    document.getElementById("select-all").onclick = () => setAll(true);
    document.getElementById("clear-all").onclick = () => setAll(false);
    document.getElementById("start-button").onclick = () => {
        savePreferences();
        location.href = "detection.html"
    }
}

function initDetectionPage() {
    initializeDetection()
}