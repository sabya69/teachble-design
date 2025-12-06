// app.js
// Two-class image classifier using MobileNet embeddings + small dense classifier in-browser

// ----- HTML Elements -----
let videoEl = document.getElementById('webcam');
let previewCanvas = document.getElementById('previewCanvas');
const ctxPreview = previewCanvas.getContext('2d');

const captureA = document.getElementById('captureA');
const captureB = document.getElementById('captureB');
const galleryA = document.getElementById('galleryA');
const galleryB = document.getElementById('galleryB');
const thumbA = document.getElementById('thumbA');
const thumbB = document.getElementById('thumbB');
const trainBtn = document.getElementById('trainBtn');
const startPredictBtn = document.getElementById('startPredictBtn');
const stopPredictBtn = document.getElementById('stopPredictBtn');
const statusEl = document.getElementById('status');

const barA = document.getElementById('barA');
const barB = document.getElementById('barB');
const pctA = document.getElementById('pctA');
const pctB = document.getElementById('pctB');

// ----- Variables -----
let mobilenetModel;
let classifierModel;
let embeddingsA = [];
let embeddingsB = [];
let predictInterval = null;

// ----- Setup Webcam -----
async function setupWebcam() {
  const stream = await navigator.mediaDevices.getUserMedia({ video: true });
  videoEl.srcObject = stream;
  await videoEl.play();
}

// ----- Capture frame as 224x224 canvas -----
function captureFrameToCanvas() {
  const c = document.createElement('canvas');
  c.width = 224;
  c.height = 224;
  c.getContext('2d').drawImage(videoEl, 0, 0, 224, 224);
  return c;
}

// ----- Add thumbnail to gallery -----
function addThumbnail(imgUrl, gallery, thumb) {
  const img = document.createElement('img');
  img.src = imgUrl;
  img.className = "thumb-img";
  gallery.appendChild(img);
  thumb.src = imgUrl;
}

// ----- Get embedding from canvas using MobileNet -----
async function embedFromCanvas(canvas) {
  return tf.tidy(() => {
    const img = tf.browser.fromPixels(canvas).toFloat().div(255).expandDims(0);
    const embedding = mobilenetModel.infer(img, 'global_average'); // shape [1,1024]
    return embedding.squeeze().clone(); // shape [1024]
  });
}

// ----- Capture Class A -----
captureA.addEventListener('click', async () => {
  const canvas = captureFrameToCanvas();
  addThumbnail(canvas.toDataURL(), galleryA, thumbA);
  embeddingsA.push(await embedFromCanvas(canvas));
  statusEl.innerText = `Class A samples: ${embeddingsA.length}`;
});

// ----- Capture Class B -----
captureB.addEventListener('click', async () => {
  const canvas = captureFrameToCanvas();
  addThumbnail(canvas.toDataURL(), galleryB, thumbB);
  embeddingsB.push(await embedFromCanvas(canvas));
  statusEl.innerText = `Class B samples: ${embeddingsB.length}`;
});

// ----- Build classifier model -----
function buildModel(inputSize) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ units: 128, activation: 'relu', inputShape: [inputSize] }));
  model.add(tf.layers.dropout({ rate: 0.25 }));
  model.add(tf.layers.dense({ units: 2, activation: 'softmax' }));
  model.compile({
    optimizer: tf.train.adam(0.0005),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });
  return model;
}

// ----- TRAIN -----
trainBtn.addEventListener('click', async () => {
  if ((embeddingsA.length + embeddingsB.length) < 10) {
    alert("At least 10 total images needed (example: 5A + 5B)");
    return;
  }

  trainBtn.disabled = true;
  statusEl.innerText = "Preparing training data...";

  const xs = tf.stack([...embeddingsA, ...embeddingsB]);
  const labels = [...Array(embeddingsA.length).fill(0), ...Array(embeddingsB.length).fill(1)];
  const ys = tf.oneHot(tf.tensor1d(labels, 'int32'), 2);

  classifierModel = buildModel(xs.shape[1]);

  statusEl.innerText = "Training...";
  await classifierModel.fit(xs, ys, {
    epochs: 25,
    shuffle: true,
    callbacks: {
      onEpochEnd: (epoch, logs) => {
        statusEl.innerText = `Epoch ${epoch + 1} — Loss: ${logs.loss.toFixed(3)} Acc: ${logs.acc.toFixed(3)}`;
      }
    }
  });

  xs.dispose();
  ys.dispose();

  statusEl.innerText = "Training Done! Start Prediction.";
  startPredictBtn.disabled = false;
  trainBtn.disabled = false;
});

// ----- LIVE PREDICTION -----
startPredictBtn.addEventListener('click', () => {
  startPredictBtn.disabled = true;
  stopPredictBtn.disabled = false;
  statusEl.innerText = "Predicting Live...";

  predictInterval = setInterval(async () => {
    const canvas = captureFrameToCanvas();
    ctxPreview.drawImage(canvas, 0, 0, previewCanvas.width, previewCanvas.height);

    // Get embedding
    const emb = await embedFromCanvas(canvas);
    const input = emb.expandDims(0);

    // Predict using classifier
    const prediction = tf.tidy(() => classifierModel.predict(input));
    const data = await prediction.data();
    const [pA, pB] = data;

    // Update UI
    updateBars(pA, pB);

    // Cleanup
    emb.dispose();
    input.dispose();
    prediction.dispose();
  }, 300);
});

// ----- STOP PREDICTION -----
stopPredictBtn.addEventListener('click', () => {
  if (predictInterval) clearInterval(predictInterval);
  predictInterval = null;
  startPredictBtn.disabled = false;
  stopPredictBtn.disabled = true;
  statusEl.innerText = "Stopped.";
});

// ----- Update Result Bars -----
function updateBars(pA, pB) {
  const aPct = (pA * 100).toFixed(2);
  const bPct = (pB * 100).toFixed(2);

  barA.style.width = `${aPct}%`;
  barB.style.width = `${bPct}%`;

  pctA.innerText = `${aPct}%`;
  pctB.innerText = `${bPct}%`;
}

// ----- Load MobileNet -----
async function loadMobileNet() {
  statusEl.innerText = "Loading MobileNet...";
  mobilenetModel = await mobilenet.load({ version: 2, alpha: 1.0 });
  statusEl.innerText = "MobileNet Loaded ✓";
}

// ----- INIT -----
async function init() {
  await setupWebcam();
  await loadMobileNet();
  statusEl.innerText = "Ready! Capture images now.";
}

init();
