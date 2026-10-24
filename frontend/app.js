// ============================================================
// SIGNBRIDGE
// CAMERA + MEDIAPIPE + TRAINED ISL MODEL + GEMINI + SPEECH
// ============================================================
 
console.log("======================================");
console.log("SIGNBRIDGE AI STARTING");
console.log("======================================");
 
 
// ============================================================
// DOM ELEMENTS
// ============================================================
 
const video = document.querySelector("video");
const canvas = document.querySelector("canvas");
 
const cameraStatus =
    document.getElementById("cameraStatus");
 
const detectedSign =
    document.getElementById("detectedSign") ||
    document.getElementById("prediction");
 
const confidenceElement =
    document.getElementById("confidence");
 
const recognizedWords =
    document.getElementById("recognizedWords") ||
    document.getElementById("words");
 
const sentenceElement =
    document.getElementById("sentence") ||
    document.getElementById("sentenceOutput");
 
const startButton =
    document.getElementById("startInterpreter") ||
    document.getElementById("startButton");
 
const stopButton =
    document.getElementById("stopCamera") ||
    document.getElementById("stopButton");
 
const clearButton =
    document.getElementById("clearWords") ||
    document.getElementById("clearButton");
 
const speakButton =
    document.getElementById("speakButton") ||
    document.getElementById("speak");
 
const improveButton =
    document.getElementById("improveButton") ||
    document.getElementById("improveSentence") ||
    document.getElementById("geminiButton");
 
 
// ============================================================
// GLOBAL VARIABLES
// ============================================================
 
let handLandmarker = null;
let islModel = null;
let labels = [];
 
let sequence = [];
 
const SEQUENCE_LENGTH = 20;
const FEATURE_SIZE = 126;
 
let lastVideoTime = -1;
 
let processing = false;
let interpreterRunning = false;
 
 
// Prediction stabilization
 
let candidateLabel = "";
let candidateCount = 0;
 
const STABLE_FRAMES = 2; // resampled-gesture predictions are already "one shot", so 2 agreeing predictions is enough
const CONFIDENCE_THRESHOLD = 0.65;
 
let lastAddedTime = 0;
 
const PREDICTION_COOLDOWN = 800;
 
 
// ============================================================
// GESTURE SEGMENTATION (matches train.py's resample_to_length)
// ============================================================
// train.py resamples each *whole* recorded sign to 20 evenly-spaced
// frames. The old sliding-window approach fed the model 20 raw,
// un-resampled webcam frames on every tick, which is a different
// distribution than what the model was trained on. Instead, we now
// buffer frames for one full gesture (start of motion -> pause) and
// resample that buffer to 20 frames the same way train.py does.
 
let gestureBuffer = [];      // raw 126-dim feature frames for the in-progress gesture
let restStreak = 0;          // consecutive low-motion frames seen
 
const REST_THRESHOLD = 0.035;     // avg per-feature movement below this = "hand is still"
const REST_FRAMES_NEEDED = 5;     // consecutive still frames to treat the gesture as finished
const MIN_GESTURE_FRAMES = 8;     // ignore accidental blips shorter than this
const MAX_GESTURE_FRAMES = 90;    // ~3s safety cap so a sign that never pauses still gets predicted
 
 
function motionMagnitude(a, b) {
    let sum = 0;
    for (let i = 0; i < a.length; i++) {
        sum += Math.abs(a[i] - b[i]);
    }
    return sum / a.length;
}
 
 
// Linear-interpolation resample - JS port of train.py's resample_to_length()
function resampleFrames(frames, targetLen) {
 
    const n = frames.length;
 
    if (n === 0) return null;
 
    if (n === targetLen) return frames;
 
    const featureDim = frames[0].length;
    const out = [];
 
    for (let t = 0; t < targetLen; t++) {
 
        const u = targetLen === 1 ? 0 : t / (targetLen - 1);
        const pos = u * (n - 1);
        const i0 = Math.floor(pos);
        const i1 = Math.min(i0 + 1, n - 1);
        const frac = pos - i0;
 
        const frame = new Array(featureDim);
 
        for (let d = 0; d < featureDim; d++) {
            frame[d] = frames[i0][d] + (frames[i1][d] - frames[i0][d]) * frac;
        }
 
        out.push(frame);
    }
 
    return out;
}
 
 
// Recognized words
 
let words = [];
 
 
// ============================================================
// MEDIAPIPE
// ============================================================
 
const MEDIAPIPE_VERSION =
    "0.10.22-rc.20250304";
 
const MEDIAPIPE_PACKAGE =
    `https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@${MEDIAPIPE_VERSION}`;
 
const MEDIAPIPE_WASM =
    `${MEDIAPIPE_PACKAGE}/wasm`;
 
const HAND_MODEL =
    "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task";
 
 
// ============================================================
// LOAD MEDIAPIPE
// ============================================================
 
async function loadMediaPipe() {
 
    console.log("Loading MediaPipe...");
 
    try {
 
        const mediaPipe =
            await import(MEDIAPIPE_PACKAGE);
 
        const HandLandmarker =
            mediaPipe.HandLandmarker;
 
        const FilesetResolver =
            mediaPipe.FilesetResolver;
 
 
        console.log("MediaPipe package loaded");
 
 
        const vision =
            await FilesetResolver.forVisionTasks(
                MEDIAPIPE_WASM
            );
 
 
        console.log("MediaPipe WASM loaded");
 
 
        handLandmarker =
            await HandLandmarker.createFromOptions(
                vision,
                {
 
                    baseOptions: {
 
                        modelAssetPath:
                            HAND_MODEL,
 
                        delegate: "GPU"
 
                    },
 
                    runningMode: "VIDEO",
 
                    numHands: 2,
 
                    minHandDetectionConfidence: 0.5,
 
                    minHandPresenceConfidence: 0.5,
 
                    minTrackingConfidence: 0.5
 
                }
            );
 
 
        console.log("MEDIAPIPE READY");
 
        return true;
 
    }
 
    catch (error) {
 
        console.error(
            "MediaPipe error:",
            error
        );
 
        alert(
            "MediaPipe failed to load.\n\n" +
            error.message
        );
 
        return false;
    }
}
 
 
// ============================================================
// LOAD TRAINED TENSORFLOW MODEL
// ============================================================
 
async function loadAIModel() {
 
    console.log("Loading trained ISL model...");
 
    try {
 
        // TensorFlow.js must be loaded
 
        if (typeof tf === "undefined") {
 
            throw new Error(
                "TensorFlow.js is not loaded."
            );
 
        }
 
 
        await tf.ready();
 
 
        console.log(
            "TensorFlow.js ready"
        );
 
 
        // Load your converted model
 
 
islModel =
    await tf.loadLayersModel(
        "/model/model.json"
    );
 
        console.log(
            "======================================"
        );
 
        console.log(
            "TRAINED ISL MODEL LOADED"
        );
 
        console.log(
            "======================================"
        );
 
 
        console.log(
            "Input shape:",
            islModel.inputs[0].shape
        );
 
        console.log(
            "Output shape:",
            islModel.outputs[0].shape
        );
 
 
        return true;
 
    }
 
    catch (error) {
 
        console.error(
            "AI model loading error:",
            error
        );
 
        alert(
            "AI model failed to load.\n\n" +
            error.message
        );
 
        return false;
    }
}
 
 
// ============================================================
// LOAD LABELS
// ============================================================
 
async function loadLabels() {
 
    console.log(
        "Loading labels..."
    );
 
    try {
 
        const response =
            await fetch(
                "/training/labels.json"
            );
 
 
        if (!response.ok) {
 
            throw new Error(
                "Could not load labels.json"
            );
 
        }
 
 
        labels =
            await response.json();
 
 
        console.log(
            "Labels loaded:",
            labels.length
        );
 
 
        console.log(
            labels
        );
 
 
        return true;
 
    }
 
    catch (error) {
 
        console.error(
            "Labels error:",
            error
        );
 
        alert(
            "Could not load labels.json"
        );
 
        return false;
    }
}
 
 
// ============================================================
// START CAMERA
// ============================================================
 
async function startCamera() {
 
    if (!video) {
 
        throw new Error(
            "Video element not found."
        );
 
    }
 
 
    console.log(
        "Requesting camera..."
    );
 
 
    const stream =
        await navigator.mediaDevices.getUserMedia({
 
            video: {
 
                facingMode: "user",
 
                width: {
                    ideal: 1280
                },
 
                height: {
                    ideal: 720
                }
 
            },
 
            audio: false
 
        });
 
 
    video.srcObject =
        stream;
 
    video.autoplay =
        true;
 
    video.muted =
        true;
 
    video.playsInline =
        true;
 
 
    await video.play();
 
 
    console.log(
        "CAMERA STARTED"
    );
 
 
    if (cameraStatus) {
 
        cameraStatus.textContent =
            "Camera On";
 
    }
}
 
 
// ============================================================
// NORMALIZE ONE HAND
// ============================================================
 
function normalizeHand(landmarks) {
 
    const output = [];
 
 
    if (
        !landmarks ||
        landmarks.length !== 21
    ) {
 
        return new Array(63).fill(0);
 
    }
 
 
    // Wrist
 
    const wrist =
        landmarks[0];
 
 
    // Landmark 9 = middle finger MCP
 
    const middleMCP =
        landmarks[9];
 
 
    const dx =
        middleMCP.x - wrist.x;
 
    const dy =
        middleMCP.y - wrist.y;
 
    const dz =
        middleMCP.z - wrist.z;
 
 
    let scale =
        Math.sqrt(
            dx * dx +
            dy * dy +
            dz * dz
        );
 
 
    // Avoid division by zero
 
    if (
        !Number.isFinite(scale) ||
        scale < 0.000001
    ) {
 
        scale = 1;
 
    }
 
 
    for (
        const point
        of landmarks
    ) {
 
        const x =
            (point.x - wrist.x) /
            scale;
 
        const y =
            (point.y - wrist.y) /
            scale;
 
        const z =
            (point.z - wrist.z) /
            scale;
 
 
        output.push(x);
        output.push(y);
        output.push(z);
 
    }
 
 
    return output;
}
 
 
// ============================================================
// CREATE 126 FEATURES
// ============================================================
 
function createFeatures(results) {
 
    let leftHand =
        new Array(63).fill(0);
 
    let rightHand =
        new Array(63).fill(0);
 
 
    if (
        results.landmarks &&
        results.handednesses
    ) {
 
        for (
            let i = 0;
            i < results.landmarks.length;
            i++
        ) {
 
            const landmarks =
                results.landmarks[i];
 
 
            const handInfo =
                results.handednesses[i];
 
 
            const handType =
                handInfo &&
                handInfo[0] &&
                handInfo[0].categoryName;
 
 
            const normalized =
                normalizeHand(
                    landmarks
                );
 
 
            if (
                handType === "Left"
            ) {
 
                leftHand =
                    normalized;
 
            }
 
            else if (
                handType === "Right"
            ) {
 
                rightHand =
                    normalized;
 
            }
 
        }
 
    }
 
 
    // IMPORTANT:
    // Training format = LEFT 63 + RIGHT 63
 
    return [
        ...leftHand,
        ...rightHand
    ];
}
 
 
// ============================================================
// DRAW HAND LANDMARKS
// ============================================================
 
function drawHands(results) {
 
    if (!canvas || !video) {
        return;
    }
 
 
    const ctx =
        canvas.getContext("2d");
 
 
    canvas.width =
        video.videoWidth;
 
    canvas.height =
        video.videoHeight;
 
 
    ctx.clearRect(
        0,
        0,
        canvas.width,
        canvas.height
    );
 
 
    if (!results.landmarks) {
        return;
    }
 
 
    const connections = [
 
        [0,1],
        [1,2],
        [2,3],
        [3,4],
 
        [0,5],
        [5,6],
        [6,7],
        [7,8],
 
        [5,9],
        [9,10],
        [10,11],
        [11,12],
 
        [9,13],
        [13,14],
        [14,15],
        [15,16],
 
        [13,17],
        [17,18],
        [18,19],
        [19,20],
 
        [0,17]
 
    ];
 
 
    for (
        const landmarks
        of results.landmarks
    ) {
 
        // Points
 
        for (
            const point
            of landmarks
        ) {
 
            // (1 - x) mirrors the point horizontally so it lines up
            // with the CSS-mirrored #camera video (see style.css).
 
            const x =
                (1 - point.x) *
                canvas.width;
 
            const y =
                point.y *
                canvas.height;
 
 
            ctx.beginPath();
 
            ctx.arc(
                x,
                y,
                5,
                0,
                Math.PI * 2
            );
 
            ctx.fill();
 
        }
 
 
        // Connections
 
        for (
            const [a,b]
            of connections
        ) {
 
            const p1 =
                landmarks[a];
 
            const p2 =
                landmarks[b];
 
 
            ctx.beginPath();
 
            ctx.moveTo(
                (1 - p1.x) *
                canvas.width,
 
                p1.y *
                canvas.height
            );
 
 
            ctx.lineTo(
                (1 - p2.x) *
                canvas.width,
 
                p2.y *
                canvas.height
            );
 
 
            ctx.stroke();
 
        }
 
    }
}
 
 
// ============================================================
// PREDICT SIGN
// ============================================================
 
async function predictSign(inputSequence) {
 
    if (
        !islModel ||
        !inputSequence ||
        inputSequence.length !==
        SEQUENCE_LENGTH
    ) {
 
        return;
 
    }
 
 
    if (processing) {
        return;
    }
 
 
    processing = true;
 
 
    try {
 
 
        // Shape:
        // [1, 20, 126]
 
        const input =
            tf.tensor3d(
                [inputSequence],
                [
                    1,
                    SEQUENCE_LENGTH,
                    FEATURE_SIZE
                ]
            );
 
 
       const prediction =
    islModel.predict(
        input
    );
 
 
        const probabilities =
            await prediction.data();
            // DEBUG: show top 5 predictions
let debugResults = [];
 
for (let i = 0; i < probabilities.length; i++) {
    debugResults.push({
        label: labels[i],
        confidence: probabilities[i]
    });
}
 
debugResults.sort((a, b) => b.confidence - a.confidence);
 
console.log("========== MODEL PREDICTION ==========");
console.log(JSON.stringify(debugResults.slice(0, 5), null, 2));
console.log("======================================");
 
 
        // Find highest probability
 
        let bestIndex = 0;
        let bestProbability = 0;
 
 
        for (
            let i = 0;
            i < probabilities.length;
            i++
        ) {
 
            if (
                probabilities[i] >
                bestProbability
            ) {
 
                bestProbability =
                    probabilities[i];
 
                bestIndex =
                    i;
 
            }
 
        }
 
 
        const label =
            labels[bestIndex] ||
            "Unknown";
 
 
        const confidence =
            bestProbability;
 
 
        console.log(
            "Prediction:",
            label,
            (confidence * 100).toFixed(1) + "%"
        );
 
 
        // Update UI
 
        if (detectedSign) {
 
            detectedSign.textContent =
                label;
 
        }
 
 
        if (confidenceElement) {
 
            confidenceElement.textContent =
                "Confidence: " +
                (confidence * 100).toFixed(1) +
                "%";
 
        }
 
 
        // Tensor cleanup
 
        input.dispose();
 
        if (
            prediction &&
            prediction.dispose
        ) {
 
            prediction.dispose();
 
        }
 
 
        // Confidence filter
        //
        // Each call to predictSign() now represents one full, resampled
        // gesture (see gesture segmentation in processFrame), not a raw
        // per-frame sample - so we no longer need to wait for several
        // consecutive frames to agree before accepting a word. A single
        // confident prediction per completed gesture is enough.
 
        if (
            confidence <
            CONFIDENCE_THRESHOLD
        ) {
 
            return;
 
        }
 
 
        const now =
            Date.now();
 
 
        if (
            now -
            lastAddedTime >=
            PREDICTION_COOLDOWN
        ) {
 
            addWord(
                label
            );
 
 
            lastAddedTime =
                now;
 
        }
 
    }
 
    catch (error) {
 
        console.error(
            "Prediction error:",
            error
        );
 
    }
 
    finally {
 
        processing = false;
 
    }
}
 
 
// ============================================================
// ADD WORD
// ============================================================
 
function addWord(word) {
 
    words.push(word);
 
    console.log(
        "WORD ADDED:",
        word
    );
 
 
    updateWordsUI();
 
 
    // Automatically create simple sentence
 
    if (sentenceElement) {
 
        sentenceElement.textContent =
            words.join(" ");
 
    }
 
}
 
 
// ============================================================
// UPDATE WORD UI
// ============================================================
 
function updateWordsUI() {
 
    if (!recognizedWords) {
        return;
    }
 
 
    recognizedWords.textContent =
        words.join(" ");
}
 
 
// ============================================================
// PROCESS CAMERA FRAME
// ============================================================
 
async function processFrame() {
 
    if (
        !interpreterRunning ||
        !handLandmarker ||
        !video
    ) {
 
        requestAnimationFrame(
            processFrame
        );
 
        return;
 
    }
 
 
    if (
        video.readyState >=
        HTMLMediaElement.HAVE_CURRENT_DATA
    ) {
 
        const currentTime =
            video.currentTime;
 
 
        if (
            currentTime !==
            lastVideoTime
        ) {
 
            lastVideoTime =
                currentTime;
 
 
            try {
 
                // MediaPipe detection
 
                const results =
                    handLandmarker.detectForVideo(
                        video,
                        performance.now()
                    );
 
 
                // Draw landmarks
 
                drawHands(
                    results
                );
 
 
                // Did we find a hand?
 
                if (
                    results.landmarks &&
                    results.landmarks.length > 0
                ) {
 
                    // Create 126 features
 
                    const features =
                        createFeatures(
                            results
                        );
 
 
                    // Track motion vs. the previous frame so we can
                    // detect the natural pause between one sign and
                    // the next (gesture boundary).
 
                    if (gestureBuffer.length > 0) {
 
                        const lastFrame =
                            gestureBuffer[
                                gestureBuffer.length - 1
                            ];
 
                        const motion =
                            motionMagnitude(
                                features,
                                lastFrame
                            );
 
                        if (motion < REST_THRESHOLD) {
                            restStreak++;
                        } else {
                            restStreak = 0;
                        }
 
                    }
 
                    gestureBuffer.push(features);
 
 
                    // Safety cap: if the user never pauses, still
                    // force a prediction so the app doesn't go silent.
 
                    if (
                        gestureBuffer.length >
                        MAX_GESTURE_FRAMES
                    ) {
 
                        const resampled =
                            resampleFrames(
                                gestureBuffer,
                                SEQUENCE_LENGTH
                            );
 
                        if (resampled) {
                            predictSign(resampled);
                        }
 
                        gestureBuffer = [];
                        restStreak = 0;
 
                    }
 
                    // Gesture finished: hand has paused after moving.
 
                    else if (
                        restStreak >=
                        REST_FRAMES_NEEDED &&
                        gestureBuffer.length >=
                        MIN_GESTURE_FRAMES
                    ) {
 
                        // Drop the trailing "still" frames - we want
                        // the motion of the sign itself, not the pause
                        // after it - then resample to 20 frames the
                        // same way train.py resampled each recording.
 
                        const trimmed =
                            gestureBuffer.slice(
                                0,
                                gestureBuffer.length -
                                REST_FRAMES_NEEDED +
                                1
                            );
 
                        const resampled =
                            resampleFrames(
                                trimmed,
                                SEQUENCE_LENGTH
                            );
 
                        if (resampled) {
                            predictSign(resampled);
                        }
 
                        gestureBuffer = [];
                        restStreak = 0;
 
                    }
 
                }
 
                else {
 
                    // No hand - flush any in-progress gesture so a
                    // sign performed right up to the edge of frame
                    // isn't lost, then reset for the next one.
 
                    if (
                        gestureBuffer.length >=
                        MIN_GESTURE_FRAMES
                    ) {
 
                        const resampled =
                            resampleFrames(
                                gestureBuffer,
                                SEQUENCE_LENGTH
                            );
 
                        if (resampled) {
                            predictSign(resampled);
                        }
 
                    }
 
                    gestureBuffer = [];
                    restStreak = 0;
 
                    sequence = [];
 
                    candidateCount = 0;
                    candidateLabel = "";
 
 
                    if (detectedSign) {
 
                        detectedSign.textContent =
                            "Show your hand";
 
                    }
 
 
                    if (confidenceElement) {
 
                        confidenceElement.textContent =
                            "Confidence: 0%";
 
                    }
 
                }
 
            }
 
            catch (error) {
 
                console.error(
                    "Frame processing error:",
                    error
                );
 
            }
 
        }
 
    }
 
 
    requestAnimationFrame(
        processFrame
    );
}
 
 
// ============================================================
// START INTERPRETER
// ============================================================
 
async function startInterpreter() {
 
    if (interpreterRunning) {
        return;
    }
 
 
    try {
 
        console.log(
            "======================================"
        );
 
        console.log(
            "STARTING SIGNBRIDGE INTERPRETER"
        );
 
        console.log(
            "======================================"
        );
 
 
        // 1. Load TensorFlow model
 
        const modelReady =
            await loadAIModel();
 
 
        if (!modelReady) {
            return;
        }
 
 
        // 2. Load labels
 
        const labelsReady =
            await loadLabels();
 
 
        if (!labelsReady) {
            return;
        }
 
 
        // 3. Load MediaPipe
 
        const mediaPipeReady =
            await loadMediaPipe();
 
 
        if (!mediaPipeReady) {
            return;
        }
 
 
        // 4. Camera
 
        await startCamera();
 
 
        // 5. Start interpreter
 
        interpreterRunning =
            true;
 
 
        if (startButton) {
 
            startButton.textContent =
                "Interpreter Running";
 
            startButton.disabled =
                true;
 
        }
 
 
        console.log(
            "======================================"
        );
 
        console.log(
            "SIGNBRIDGE IS READY"
        );
 
        console.log(
            "======================================"
        );
 
 
        // Start processing
 
        processFrame();
 
    }
 
    catch (error) {
 
        console.error(
            "STARTUP ERROR:",
            error
        );
 
 
        alert(
            "SignBridge could not start.\n\n" +
            error.name +
            "\n" +
            error.message
        );
 
    }
 
}
 
 
// ============================================================
// STOP INTERPRETER
// ============================================================
 
function stopInterpreter() {
 
    if (!interpreterRunning) {
        return;
    }
 
    interpreterRunning = false;
 
 
    if (video && video.srcObject) {
 
        const tracks =
            video.srcObject.getTracks();
 
        for (const track of tracks) {
            track.stop();
        }
 
        video.srcObject = null;
 
    }
 
 
    sequence = [];
    gestureBuffer = [];
    restStreak = 0;
    candidateLabel = "";
    candidateCount = 0;
 
 
    if (cameraStatus) {
 
        cameraStatus.textContent =
            "Camera Off";
 
    }
 
    if (detectedSign) {
 
        detectedSign.textContent =
            "—";
 
    }
 
    if (confidenceElement) {
 
        confidenceElement.textContent =
            "Confidence: 0%";
 
    }
 
    if (startButton) {
 
        startButton.textContent =
            "Start Interpreter";
 
        startButton.disabled =
            false;
 
    }
 
    console.log(
        "Interpreter stopped."
    );
 
}
 
 
// ============================================================
// CLEAR WORDS
// ============================================================
 
function clearWords() {
 
    words = [];
 
    sequence = [];
 
    gestureBuffer = [];
 
    restStreak = 0;
 
    candidateLabel = "";
 
    candidateCount = 0;
 
 
    if (recognizedWords) {
 
        recognizedWords.textContent =
            "";
 
    }
 
 
    if (sentenceElement) {
 
        sentenceElement.textContent =
            "Start signing to translate...";
 
    }
 
 
    if (detectedSign) {
 
        detectedSign.textContent =
            "—";
 
    }
 
 
    if (confidenceElement) {
 
        confidenceElement.textContent =
            "Confidence: 0%";
 
    }
 
 
    console.log(
        "Words cleared"
    );
 
}
 
 
// ============================================================
// TEXT TO SPEECH
// ============================================================
 
function speakSentence() {
 
    if (!sentenceElement) {
        return;
    }
 
 
    const text =
        sentenceElement.textContent.trim();
 
 
    if (
        !text ||
        text ===
        "Start signing to translate..."
    ) {
 
        return;
 
    }
 
 
    if (
        !window.speechSynthesis
    ) {
 
        alert(
            "Speech synthesis is not supported."
        );
 
        return;
 
    }
 
 
    window.speechSynthesis.cancel();
 
 
    const utterance =
        new SpeechSynthesisUtterance(
            text
        );
 
 
    utterance.lang =
        "en-IN";
 
    utterance.rate =
        0.9;
 
 
    window.speechSynthesis.speak(
        utterance
    );
 
}
 
 
// ============================================================
// GEMINI SENTENCE IMPROVEMENT
// ============================================================
 
async function improveSentence() {
 
    console.log("Improve Sentence clicked");
    console.log("Words being sent to Gemini:", words);
 
    if (!words || words.length === 0) {
        if (sentenceElement) {
            sentenceElement.textContent =
                "No signs detected yet.";
        }
        return;
    }
 
    if (sentenceElement) {
        sentenceElement.textContent =
            "Improving sentence...";
    }
 
    try {
 
        const response = await fetch("/api/improve", {
            method: "POST",
 
            headers: {
                "Content-Type": "application/json"
            },
 
            body: JSON.stringify({
                words: words
            })
        });
 
        console.log(
            "Gemini response status:",
            response.status
        );
 
        const data = await response.json();
 
        console.log(
            "Gemini response:",
            data
        );
 
        if (data.sentence) {
 
            sentenceElement.textContent =
                data.sentence;
 
        } else {
 
            sentenceElement.textContent =
                words.join(" ");
 
        }
 
    } catch (error) {
 
        console.error(
            "Gemini connection error:",
            error
        );
 
        sentenceElement.textContent =
            words.join(" ");
 
    }
}
 
 
// ============================================================
// BUTTON EVENTS
// ============================================================
 
if (startButton) {
 
    startButton.addEventListener(
        "click",
        startInterpreter
    );
 
}
 
 
if (stopButton) {
 
    stopButton.addEventListener(
        "click",
        stopInterpreter
    );
 
}
 
 
if (clearButton) {
 
    clearButton.addEventListener(
        "click",
        clearWords
    );
 
}
 
 
if (speakButton) {
 
    speakButton.addEventListener(
        "click",
        speakSentence
    );
 
}
 
 
if (improveButton) {
 
    improveButton.addEventListener(
        "click",
        improveSentence
    );
 
}
 
 
// ============================================================
// PAGE LOAD
// ============================================================
 
window.addEventListener(
    "load",
    () => {
 
        console.log(
            "SignBridge page loaded."
        );
 
        console.log(
            "Click Start Interpreter."
        );
 
    }
);
 