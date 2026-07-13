# 🧠 NeuroScan AI — Model Integration Guide

## Current State (Demo Mode)
The app currently runs in **simulation mode**. The AI predicts tumor types
randomly to demonstrate the full workflow. To use a real trained model,
follow one of the integration paths below.

---

## Integration Option A: TensorFlow.js (Run model in Browser — No Backend Needed)

### Step 1 — Convert your trained model
```bash
# Install converter
pip install tensorflowjs

# Convert Keras .h5 model to TF.js format
tensorflowjs_converter \
  --input_format=keras \
  my_brain_tumor_model.h5 \
  ./model/tfjs_model/
```

### Step 2 — Add TF.js to index.html
```html
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs@latest"></script>
```

### Step 3 — Replace simulateAIAnalysis() in app.js
```javascript
// Set in CONFIG:
CONFIG.USE_REAL_MODEL = true;

// Load model once at startup:
let tfModel = null;
async function loadModel() {
  tfModel = await tf.loadLayersModel('./model/tfjs_model/model.json');
  console.log('Model loaded!');
}
loadModel();

// Replace callRealModelAPI() with this:
async function callRealModelAPI(imageFile) {
  // Preprocess image to match your model's input shape
  // Most models use 224x224 or 256x256 input
  const img = await loadImageAsTensor(imageFile, 224, 224);
  const prediction = tfModel.predict(img);
  const probabilities = await prediction.data();

  // Map probabilities to tumor classes
  // Adjust class order to match YOUR model's training labels
  const classes = ['glioma', 'meningioma', 'no_tumor', 'pituitary'];
  const maxIdx = probabilities.indexOf(Math.max(...probabilities));

  return {
    tumorType: classes[maxIdx],
    levelIndex: 0,  // You can add grade prediction separately
    confidence: probabilities[maxIdx],
  };
}

async function loadImageAsTensor(file, width, height) {
  return new Promise((resolve) => {
    const img = new Image();
    img.onload = () => {
      const tensor = tf.browser.fromPixels(img)
        .resizeBilinear([height, width])
        .toFloat()
        .div(255.0)
        .expandDims(0);
      resolve(tensor);
    };
    img.src = URL.createObjectURL(file);
  });
}
```

---

## Integration Option B: Python Flask Backend API

### Step 1 — Create Flask server (server.py)
```python
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io

app = Flask(__name__)
CORS(app)  # Allow requests from browser

# Load your trained model
model = tf.keras.models.load_model('brain_tumor_model.h5')

# Class labels (adjust to match your training)
CLASSES = ['glioma', 'meningioma', 'no_tumor', 'pituitary']

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(io.BytesIO(file.read())).convert('RGB')
    img = img.resize((224, 224))
    arr = np.array(img) / 255.0
    arr = np.expand_dims(arr, axis=0)

    preds = model.predict(arr)[0]
    tumor_type = CLASSES[np.argmax(preds)]
    confidence = float(np.max(preds))

    return jsonify({
        'tumor_type': tumor_type,
        'grade_index': 1,       # Add grade model if available
        'confidence': confidence
    })

if __name__ == '__main__':
    app.run(port=5000, debug=True)
```

### Step 2 — Install dependencies
```bash
pip install flask flask-cors tensorflow pillow numpy
```

### Step 3 — Run backend
```bash
python server.py
```

### Step 4 — Update app.js
```javascript
CONFIG.USE_REAL_MODEL = true;
CONFIG.MODEL_API_URL = 'http://localhost:5000/predict';
```
The `callRealModelAPI()` function in app.js is already written to call this API!

---

## Integration Option C: Cloud API (Google Vertex AI / AWS / Azure)

### Step 1 — Deploy model to cloud
- **Google Vertex AI**: Upload SavedModel to GCS, deploy endpoint
- **AWS SageMaker**: Deploy model as endpoint
- **Azure ML**: Deploy to Azure ML endpoint

### Step 2 — Update callRealModelAPI() in app.js
```javascript
async function callRealModelAPI(imageFile) {
  // Convert image to base64
  const base64 = await fileToBase64(imageFile);

  const response = await fetch('YOUR_CLOUD_ENDPOINT_URL', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json',
      'Authorization': 'Bearer YOUR_API_KEY'
    },
    body: JSON.stringify({
      instances: [{ image: base64 }]
    })
  });

  const data = await response.json();
  return {
    tumorType: data.predictions[0].tumor_class,
    levelIndex: data.predictions[0].grade,
    confidence: data.predictions[0].confidence,
  };
}

function fileToBase64(file) {
  return new Promise((resolve) => {
    const reader = new FileReader();
    reader.onload = () => resolve(reader.result.split(',')[1]);
    reader.readAsDataURL(file);
  });
}
```

---

## Model Training Tips

If you're training a new model for this app:

| Parameter | Recommended Value |
|-----------|------------------|
| Architecture | ResNet50 / VGG16 / EfficientNetB0 |
| Input size | 224 × 224 × 3 |
| Output classes | 4 (glioma, meningioma, no_tumor, pituitary) |
| Dataset | Brain Tumor MRI Dataset (Kaggle) |
| Epochs | 25–50 with early stopping |
| Augmentation | Rotation, flip, zoom, brightness |

**Dataset link**: https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset

---

## File Structure
```
brain-tumor-detection/
├── index.html          ← Main app UI
├── styles.css          ← Styling
├── app.js              ← Application logic + model integration
├── model_integration.md ← This file
└── model/              ← (Create this folder for TF.js model)
    └── tfjs_model/
        ├── model.json
        └── *.bin
```
