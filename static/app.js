const video = document.getElementById("video");
const canvas = document.getElementById("canvas");

const startBtn = document.getElementById("startBtn");
const stopBtn = document.getElementById("stopBtn");
const resetBtn = document.getElementById("resetBtn");

const cameraStatus = document.getElementById("cameraStatus");
const currentSign = document.getElementById("currentSign");
const rawPrediction = document.getElementById("rawPrediction");
const confidence = document.getElementById("confidence");
const confirmation = document.getElementById("confirmation");
const inference = document.getElementById("inference");
const progressBar = document.getElementById("progressBar");
const sentence = document.getElementById("sentence");
const historyElement = document.getElementById("history");
const featureStatus = document.getElementById("featureStatus");

const SAMPLE_COUNT = 20;
const WINDOW_MS = 3000;
const SAMPLE_INTERVAL_MS = WINDOW_MS / SAMPLE_COUNT;

let stream = null;
let timer = null;
let running = false;
let busy = false;

function setStatus(text, online) {
    cameraStatus.textContent = text;
    cameraStatus.classList.toggle("online", online);
    cameraStatus.classList.toggle("offline", !online);
}

function updateUI(data) {
    rawPrediction.textContent = data.prediction || "Waiting...";
    confidence.textContent = `${((data.confidence || 0) * 100).toFixed(1)}%`;
    currentSign.textContent = data.current_sign || "—";

    const count = data.consecutive_count || 0;
    const required = data.required_consecutive || 5;

    confirmation.textContent = `${count} / ${required}`;
    progressBar.style.width =
        `${Math.min(100, (count / required) * 100)}%`;

    inference.textContent =
        data.inference_ms != null
            ? `${data.inference_ms.toFixed(1)} ms`
            : "—";

    featureStatus.textContent =
        data.feature_status === "ok"
            ? "Landmarks detected"
            : "Waiting for landmarks";

    sentence.textContent =
        data.sentence || "Start the camera and perform a sign...";

    historyElement.innerHTML = "";
    (data.history || []).forEach(item => {
        const chip = document.createElement("span");
        chip.className = "word-chip";
        chip.textContent = item.word;
        historyElement.appendChild(chip);
    });
}

async function sendFrame() {
    if (!running || busy || video.readyState < 2) return;

    busy = true;

    try {
        const width = 640;
        const height = Math.round(
            width * video.videoHeight / video.videoWidth
        );

        canvas.width = width;
        canvas.height = height;

        const ctx = canvas.getContext("2d");
        ctx.drawImage(video, 0, 0, width, height);

        const frame = canvas.toDataURL("image/jpeg", 0.72);

        const response = await fetch("/api/predict", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ frame })
        });

        const data = await response.json();

        if (!response.ok) {
            throw new Error(data.error || "Prediction failed");
        }

        updateUI(data);
    } catch (error) {
        console.error(error);
        featureStatus.textContent = error.message;
    } finally {
        busy = false;
    }
}

async function startCamera() {
    if (running) return;

    try {
        stream = await navigator.mediaDevices.getUserMedia({
            video: {
                width: { ideal: 1280 },
                height: { ideal: 720 },
                facingMode: "user"
            },
            audio: false
        });

        video.srcObject = stream;
        await video.play();

        running = true;
        startBtn.disabled = true;
        stopBtn.disabled = false;
        setStatus("LIVE", true);

        timer = setInterval(sendFrame, SAMPLE_INTERVAL_MS);
        sendFrame();
    } catch (error) {
        console.error(error);
        alert(
            "Could not access the camera. " +
            "Please allow camera permission."
        );
    }
}

function stopCamera() {
    running = false;

    if (timer) {
        clearInterval(timer);
        timer = null;
    }

    if (stream) {
        stream.getTracks().forEach(track => track.stop());
        stream = null;
    }

    video.srcObject = null;
    startBtn.disabled = false;
    stopBtn.disabled = true;
    setStatus("OFFLINE", false);
}

async function resetSentence() {
    try {
        await fetch("/api/reset", { method: "POST" });

        currentSign.textContent = "—";
        rawPrediction.textContent = "Waiting...";
        confidence.textContent = "0%";
        confirmation.textContent = "0 / 5";
        progressBar.style.width = "0%";
        sentence.textContent =
            "Start the camera and perform a sign...";
        historyElement.innerHTML = "";
    } catch (error) {
        console.error(error);
    }
}

startBtn.addEventListener("click", startCamera);
stopBtn.addEventListener("click", stopCamera);
resetBtn.addEventListener("click", resetSentence);
window.addEventListener("beforeunload", stopCamera);

setStatus("OFFLINE", false);
