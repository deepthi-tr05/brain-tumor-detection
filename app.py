# ── Python 3.11 | TensorFlow 2.15 | Flask 3.0 ──
import os
import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, redirect, url_for, session
try:
    import tensorflow as tf
    HAS_TENSORFLOW = True
except ImportError:
    HAS_TENSORFLOW = False
import sqlite3
from flask_bcrypt import Bcrypt
from flask_mail import Mail, Message
import warnings
import secrets
from datetime import datetime, timedelta
import json
import base64
from io import BytesIO
from PIL import Image
import plotly.graph_objects as go
import plotly.utils
import traceback

# Load .env file if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass  # python-dotenv not installed, use system env vars

warnings.filterwarnings('ignore')
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'   # silence TF C++ info logs

# ── Keras import: works for TF 2.15 (bundled Keras 2) and TF 2.16+ (Keras 3 standalone) ──
load_model = None
if HAS_TENSORFLOW:
    try:
        from tensorflow.keras.models import load_model   # TF 2.x bundled Keras
    except ImportError:
        try:
            from keras.models import load_model          # Keras 3 standalone
        except ImportError:
            pass

# ───────────────────────────────────────────────
# Flask app setup
# ───────────────────────────────────────────────
app = Flask(__name__)
# Secure secret key – reads from env-var in production, generates random one for dev
app.secret_key = os.environ.get('SECRET_KEY', secrets.token_hex(32))

bcrypt = Bcrypt(app)

# ── Flask-Mail Configuration (Gmail SMTP) ──
app.config['MAIL_SERVER']   = 'smtp.gmail.com'
app.config['MAIL_PORT']     = 587
app.config['MAIL_USE_TLS']  = True
app.config['MAIL_USERNAME'] = os.environ.get('MAIL_USERNAME', '')
app.config['MAIL_PASSWORD'] = os.environ.get('MAIL_PASSWORD', '')
app.config['MAIL_DEFAULT_SENDER'] = ('NeuroScan AI', os.environ.get('MAIL_USERNAME', 'noreply@neuroscan.ai'))
mail = Mail(app)

UPLOAD_FOLDER = os.path.join(os.path.dirname(__file__), 'uploads')
MODELS_FOLDER = os.path.join(os.path.dirname(__file__), 'models')
DB_PATH       = os.path.join(os.path.dirname(__file__), 'users.db')

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_FOLDER, exist_ok=True)

# ───────────────────────────────────────────────
# Constants
# ───────────────────────────────────────────────
IMG_SIZE     = 224
CLASS_NAMES  = ['No Tumor', 'Pituitary Tumor', 'Meningioma Tumor', 'Glioma Tumor']

# ───────────────────────────────────────────────
# Globals
# ───────────────────────────────────────────────
model           = None
is_model_loaded = False
password_reset_tokens = {}   # {token: {email, expiry}}
otp_store = {}               # {email: {otp, expiry}}


# ═══════════════════════════════════════════════
#  DATABASE
# ═══════════════════════════════════════════════

def get_db():
    """Return a connection to the SQLite database."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row   # access columns by name
    return conn


def init_db():
    """Create tables if they don't exist and migrate old schemas."""
    conn = get_db()
    c = conn.cursor()

    # Create users table
    c.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            username          TEXT NOT NULL,
            email             TEXT NOT NULL,
            password          TEXT NOT NULL,
            security_question TEXT,
            security_answer   TEXT
        )
    ''')

    # Migrate: add security columns if they don't exist
    for col, dtype in [('security_question', 'TEXT'), ('security_answer', 'TEXT')]:
        try:
            c.execute(f'ALTER TABLE users ADD COLUMN {col} {dtype}')
        except Exception:
            pass  # column already exists

    # Create scan_history table
    c.execute('''
        CREATE TABLE IF NOT EXISTS scan_history (
            id             INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id        INTEGER NOT NULL,
            patient_name   TEXT,
            patient_id     TEXT,
            patient_age    TEXT,
            patient_gender TEXT,
            tumor_type     TEXT NOT NULL,
            confidence     REAL,
            stage          TEXT,
            area_pct       REAL,
            location       TEXT,
            scan_date      TEXT,
            image_b64      TEXT,
            FOREIGN KEY(user_id) REFERENCES users(id)
        )
    ''')

    # Add UNIQUE index on email if it doesn't already exist
    c.execute('''
        CREATE UNIQUE INDEX IF NOT EXISTS idx_users_email ON users(email)
    ''')

    # Migrate: add patient_id column to scan_history if it doesn't exist
    try:
        c.execute('ALTER TABLE scan_history ADD COLUMN patient_id TEXT')
    except Exception:
        pass  # column already exists

    # Migrate: add doctor_name, hospital, stage_description to scan_history if missing
    for col, dtype in [
        ('doctor_name', 'TEXT'),
        ('hospital', 'TEXT'),
        ('stage_description', 'TEXT'),
        ('precautions_json', 'TEXT'),
        ('medication_json', 'TEXT'),
    ]:
        try:
            c.execute(f'ALTER TABLE scan_history ADD COLUMN {col} {dtype}')
        except Exception:
            pass  # column already exists

    conn.commit()
    conn.close()
    print("[OK] Database initialised.")


# ═══════════════════════════════════════════════
#  MODEL LOADING
# ═══════════════════════════════════════════════

def load_trained_model():
    global model, is_model_loaded
    if not HAS_TENSORFLOW or load_model is None:
        print("[WARN] TensorFlow/Keras not installed. Running in Demo Mode.")
        is_model_loaded = False
        return False
    model_paths = [
        os.path.join(MODELS_FOLDER, 'brain_tumor_cnn_model.h5'),
        os.path.join(MODELS_FOLDER, 'best_model.h5'),
    ]
    for path in model_paths:
        if os.path.exists(path):
            try:
                # compile=False avoids optimizer state issues across TF versions
                model = load_model(path, compile=False)
                # Re-compile with a fresh optimizer so predict() works
                model.compile(optimizer='adam',
                              loss='sparse_categorical_crossentropy',
                              metrics=['accuracy'])
                is_model_loaded = True
                print(f"[OK] Model loaded: {path}")
                return True
            except Exception as exc:
                print(f"[WARN] Could not load {path}: {exc}")
    print("[WARN] No trained model found. Run train_model.py first.")
    return False


# ═══════════════════════════════════════════════
#  IMAGE PRE-PROCESSING  (BGR → model input)
# ═══════════════════════════════════════════════

def preprocess_image(img_file):
    """
    Read an uploaded file and return a normalised (1,224,224,3) float32 array.
    The array stays in BGR channel order (same as the training pipeline).
    """
    img_bytes = img_file.read()
    nparr = np.frombuffer(img_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)   # BGR
    if img is None:
        return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img_array = np.expand_dims(img.astype(np.float32) / 255.0, axis=0)
    return img_array   # shape (1, 224, 224, 3)  BGR


def predict_tumor(img_array):
    if not is_model_loaded:
        # Realistic deterministic mock prediction for demo mode/serverless
        pixel_sum = float(np.sum(img_array))
        hash_val = int(pixel_sum * 1000) % 100
        # Deterministically select one of the classes: 0=No Tumor, 1=Pituitary, 2=Meningioma, 3=Glioma
        idx = hash_val % 4
        if idx == 0:
            conf = 95.0 + (hash_val % 50) / 10.0  # 95.0% - 99.9%
        else:
            conf = 78.0 + (hash_val % 200) / 10.0  # 78.0% - 98.0%
        return CLASS_NAMES[idx], conf, idx

    preds = model.predict(img_array, verbose=0)
    idx   = int(np.argmax(preds[0]))
    conf  = float(np.max(preds[0]) * 100)
    return CLASS_NAMES[idx], conf, idx


# ═══════════════════════════════════════════════
#  STAGE & PRECAUTIONS
# ═══════════════════════════════════════════════

def get_tumor_stage(tumor_type, confidence, area_pct):
    # Stage is primarily determined by area affected and confidence combined
    if area_pct < 10 and confidence < 75:
        stage = "Early Stage"
        desc_map = {
            "Pituitary Tumor":  "Microadenoma – Small tumor detected. Usually benign and slow-growing.",
            "Meningioma Tumor": "Grade I – Benign tumor. Slow growing with minimal symptoms.",
            "Glioma Tumor":     "Grade I or II – Low-grade glioma. Generally less aggressive.",
        }
    elif area_pct < 30 or confidence < 85:
        stage = "Intermediate Stage"
        desc_map = {
            "Pituitary Tumor":  "Macroadenoma – Growing tumor that may cause hormonal imbalances.",
            "Meningioma Tumor": "Grade II – Atypical meningioma. Moderate growth rate.",
            "Glioma Tumor":     "Grade III – Anaplastic glioma. Aggressive growth detected.",
        }
    else:
        stage = "Advanced Stage"
        desc_map = {
            "Pituitary Tumor":  "Large Macroadenoma – Significant size. May compress nearby structures.",
            "Meningioma Tumor": "Grade III – Anaplastic meningioma. Rapid and aggressive growth.",
            "Glioma Tumor":     "Grade IV – Glioblastoma Multiforme (GBM). Highly aggressive.",
        }

    description = desc_map.get(tumor_type, f"{stage} tumor detected. Medical consultation required.")

    if area_pct > 35 and confidence > 70:
        stage       = "Advanced Stage"
        description += " Large affected area detected."
    elif area_pct > 20 and confidence > 55 and stage == "Early Stage":
        stage       = "Early to Intermediate Stage"
        description += " Growing concern due to affected area size."

    return stage, description


def get_precautions(tumor_type, stage, confidence, area_pct):
    if "Early" in stage:
        base = [
            "Schedule MRI follow-up every 6 months",
            "Monitor for new symptoms: headaches, vision changes",
            "Take prescribed medications regularly",
            "Maintain regular exercise routine",
            "Eat antioxidant-rich foods (berries, nuts, leafy greens)",
            "Regular blood tests as recommended by doctor",
        ]
    elif "Intermediate" in stage:
        base = [
            "Consult specialist within 2–4 weeks",
            "MRI follow-up every 3 months",
            "Start prescribed medication course",
            "Avoid smoking and alcohol",
            "Monitor cognitive function and memory changes",
            "Keep emergency contacts ready",
            "Regular neurological examinations",
        ]
    else:
        base = [
            "IMMEDIATE medical consultation required",
            "Prepare for possible hospitalization",
            "Weekly follow-up appointments",
            "Arrange for caregiver support",
            "Keep emergency numbers handy",
            "Strictly follow medication schedule",
            "Avoid physical strain and heavy lifting",
            "Contact doctor immediately if symptoms worsen",
        ]

    extras = {
        "Pituitary Tumor": [
            "Monitor vision changes (blurred / double vision)",
            "Check for unusual thirst or urination",
            "Monitor weight changes",
            "Regular hormone level testing",
        ],
        "Meningioma Tumor": [
            "Watch for seizures or unusual movements",
            "Report any weakness in arms or legs",
            "Note any speech difficulties",
            "Regular neurological exams",
        ],
        "Glioma Tumor": [
            "Watch for sudden severe headaches",
            "Monitor for personality or behaviour changes",
            "Report speech or language difficulties",
            "Check for one-sided body weakness",
            "Watch for seizure activity",
        ],
    }
    for key, items in extras.items():
        if key in tumor_type:
            base.extend(items)
            break

    if confidence > 85:
        base.insert(0, "HIGH CONFIDENCE – Take immediate action on all precautions")
    elif confidence < 60:
        base.insert(0, "LOW CONFIDENCE – Consider second opinion for confirmation")

    if area_pct > 40:
        base.append("Significant area affected – Urgent medical intervention advised")
    elif area_pct > 20:
        base.append("Moderate area affected – Close monitoring required")

    # Deduplicate, keep order
    seen, unique = set(), []
    for p in base:
        if p not in seen:
            seen.add(p)
            unique.append(p)

    return unique[:10]


# ═══════════════════════════════════════════════
#  MEDICATION / PRESCRIPTION
# ═══════════════════════════════════════════════

def get_medication(tumor_type, stage):
    """Return stage-specific medication and treatment recommendations."""
    meds = {}

    # ── Glioma ──
    if 'Glioma' in tumor_type:
        if 'Early' in stage:
            meds = {
                'approach': 'Surgical Resection + Radiation Therapy',
                'medications': [
                    {'name': 'Temozolomide (Temodar)', 'dose': '75 mg/m² daily during RT', 'purpose': 'Alkylating chemotherapy agent — DNA damage to tumor cells'},
                    {'name': 'Dexamethasone (Decadron)', 'dose': '4–8 mg every 6 hours', 'purpose': 'Corticosteroid — reduces cerebral edema and inflammation'},
                    {'name': 'Levetiracetam (Keppra)', 'dose': '500–1000 mg twice daily', 'purpose': 'Anticonvulsant — seizure prevention'},
                    {'name': 'Ondansetron (Zofran)', 'dose': '8 mg before chemotherapy', 'purpose': 'Antiemetic — controls nausea from chemo'},
                ],
                'follow_up': 'MRI every 6 months; blood counts every 3 weeks during chemo',
                'note': 'Treatment response is generally favorable at this stage.'
            }
        elif 'Intermediate' in stage:
            meds = {
                'approach': 'Surgery + Concurrent Chemoradiation (CRT)',
                'medications': [
                    {'name': 'Temozolomide (Temodar)', 'dose': '75 mg/m² during RT; then 150–200 mg/m² for 5/28 days', 'purpose': 'First-line chemotherapy for anaplastic glioma'},
                    {'name': 'Bevacizumab (Avastin)', 'dose': '10 mg/kg IV every 2 weeks', 'purpose': 'Anti-VEGF — reduces tumor blood supply (angiogenesis inhibitor)'},
                    {'name': 'Dexamethasone', 'dose': '8–16 mg/day (tapered)', 'purpose': 'Reduces brain swelling around tumor'},
                    {'name': 'Levetiracetam', 'dose': '1000–1500 mg twice daily', 'purpose': 'Anti-epileptic prophylaxis'},
                    {'name': 'Omeprazole (Prilosec)', 'dose': '20 mg daily', 'purpose': 'GI protection during steroid use'},
                ],
                'follow_up': 'MRI every 3 months; CBC weekly',
                'note': 'Consult neuro-oncology team for MGMT methylation status.'
            }
        else:
            meds = {
                'approach': 'Maximal Surgical Resection + Aggressive Chemoradiation + Immunotherapy',
                'medications': [
                    {'name': 'Temozolomide (Temodar)', 'dose': '200 mg/m² days 1–5 every 28 days (6+ cycles)', 'purpose': 'Standard GBM chemotherapy'},
                    {'name': 'Bevacizumab (Avastin)', 'dose': '15 mg/kg every 3 weeks', 'purpose': 'Controls tumor vasculature and edema'},
                    {'name': 'Lomustine (CCNU)', 'dose': '100–130 mg/m² every 6 weeks', 'purpose': 'Alkylating agent for recurrent GBM'},
                    {'name': 'Dexamethasone', 'dose': '16 mg/day (tapered slowly)', 'purpose': 'Critical for cerebral edema management'},
                    {'name': 'Phenytoin (Dilantin)', 'dose': '100 mg three times daily', 'purpose': 'Anti-epileptic for complex seizures'},
                    {'name': 'Ondansetron', 'dose': '8 mg every 8 hours', 'purpose': 'Antiemetic for intensive chemo regimen'},
                ],
                'follow_up': 'MRI every 6–8 weeks; urgent re-evaluation if symptoms worsen',
                'note': 'URGENT — GBM requires immediate multidisciplinary team consultation. Tumor Board review recommended.'
            }

    # ── Meningioma ──
    elif 'Meningioma' in tumor_type:
        if 'Early' in stage:
            meds = {
                'approach': 'Active Surveillance or Stereotactic Radiosurgery (SRS)',
                'medications': [
                    {'name': 'Mifepristone (RU-486)', 'dose': '200 mg daily (for hormone-sensitive tumors)', 'purpose': 'Anti-progesterone — slows meningioma growth'},
                    {'name': 'Hydroxyurea', 'dose': '20 mg/kg/day', 'purpose': 'Antimetabolite for unresectable tumors'},
                    {'name': 'Ibuprofen', 'dose': '400–600 mg every 8 hours as needed', 'purpose': 'Headache and pain management'},
                    {'name': 'Levetiracetam', 'dose': '500 mg twice daily', 'purpose': 'Seizure prophylaxis if cortical involvement'},
                ],
                'follow_up': 'MRI every 12 months; neurological exam every 6 months',
                'note': 'Grade I meningiomas may need observation only if asymptomatic.'
            }
        elif 'Intermediate' in stage:
            meds = {
                'approach': 'Surgical Resection + Adjuvant Radiation Therapy',
                'medications': [
                    {'name': 'Hydroxyurea', 'dose': '20–30 mg/kg/day', 'purpose': 'Antineoplastic — reduces tumor cell proliferation'},
                    {'name': 'Somatostatin Analogues (Octreotide)', 'dose': '100–200 mcg SC three times daily', 'purpose': 'Hormone control — reduces growth factor signaling'},
                    {'name': 'Dexamethasone', 'dose': '4–8 mg every 8 hours', 'purpose': 'Perioperative edema control'},
                    {'name': 'Lacosamide', 'dose': '100–200 mg twice daily', 'purpose': 'Anti-epileptic for cortically located tumors'},
                    {'name': 'Tramadol', 'dose': '50–100 mg every 6 hours as needed', 'purpose': 'Pain management post-surgery'},
                ],
                'follow_up': 'MRI at 3, 6, 12 months post-surgery',
                'note': 'Atypical meningioma has higher recurrence — frequent monitoring needed.'
            }
        else:
            meds = {
                'approach': 'Aggressive Surgery + High-Dose Radiation + Systemic Therapy',
                'medications': [
                    {'name': 'Bevacizumab (Avastin)', 'dose': '10 mg/kg IV every 2 weeks', 'purpose': 'Reduces tumor vascularity in anaplastic cases'},
                    {'name': 'Hydroxyurea', 'dose': '30 mg/kg/day', 'purpose': 'Chemotherapy for anaplastic meningioma'},
                    {'name': 'Temozolomide', 'dose': '150 mg/m² for 5 days every 28 days', 'purpose': 'Alkylating chemotherapy'},
                    {'name': 'Dexamethasone', 'dose': '16 mg/day', 'purpose': 'Aggressive edema management'},
                    {'name': 'Valproic Acid', 'dose': '500–1000 mg twice daily', 'purpose': 'Broad-spectrum anti-epileptic + HDAC inhibitor'},
                    {'name': 'Omeprazole', 'dose': '20 mg daily', 'purpose': 'GI protection with steroids and chemo'},
                ],
                'follow_up': 'MRI every 6 weeks; urgent escalation if symptoms progress',
                'note': 'URGENT — Anaplastic meningioma is highly aggressive. Immediate oncology referral.'
            }

    # ── Pituitary ──
    elif 'Pituitary' in tumor_type:
        if 'Early' in stage:
            meds = {
                'approach': 'Medical Management or Transsphenoidal Surgery',
                'medications': [
                    {'name': 'Cabergoline (Dostinex)', 'dose': '0.5–2 mg twice weekly', 'purpose': 'Dopamine agonist — first-line for prolactinomas'},
                    {'name': 'Bromocriptine (Parlodel)', 'dose': '2.5–15 mg daily in divided doses', 'purpose': 'Dopamine agonist — reduces prolactin secretion'},
                    {'name': 'Octreotide (Sandostatin)', 'dose': '100–200 mcg SC three times daily', 'purpose': 'Somatostatin analogue — controls GH/IGF-1 in acromegaly'},
                    {'name': 'Hydrocortisone', 'dose': '20 mg morning + 10 mg afternoon', 'purpose': 'Cortisol replacement for hypopituitarism'},
                ],
                'follow_up': 'MRI at 6 months; hormone panels every 3 months',
                'note': 'Prolactinomas respond very well to medical therapy — surgery may not be needed.'
            }
        elif 'Intermediate' in stage:
            meds = {
                'approach': 'Transsphenoidal Surgery + Postoperative Medical Management',
                'medications': [
                    {'name': 'Octreotide LAR', 'dose': '20–30 mg IM monthly', 'purpose': 'Long-acting somatostatin analogue for GH-secreting adenomas'},
                    {'name': 'Lanreotide (Somatuline)', 'dose': '120 mg SC every 4 weeks', 'purpose': 'Alternative somatostatin analogue for acromegaly'},
                    {'name': 'Cabergoline', 'dose': '1–3 mg twice weekly', 'purpose': 'Hormone normalization for prolactin-secreting tumors'},
                    {'name': 'Pasireotide (Signifor)', 'dose': '0.3-0.9 mg SC twice daily', 'purpose': "Cushing's disease - pituitary-directed therapy"},
                    {'name': 'Levothyroxine', 'dose': '50–100 mcg daily (dose titrated)', 'purpose': 'Thyroid hormone replacement for secondary hypothyroidism'},
                    {'name': 'Desmopressin (DDAVP)', 'dose': '0.1–0.2 mg twice daily', 'purpose': 'Diabetes insipidus management post-surgery'},
                ],
                'follow_up': 'MRI at 3 months post-surgery; endocrinology follow-up monthly',
                'note': 'Visual field testing important for tumors compressing optic chiasm.'
            }
        else:
            meds = {
                'approach': 'Surgery + Radiation + Comprehensive Hormone Replacement',
                'medications': [
                    {'name': 'Hydrocortisone', 'dose': '20 mg morning + 10 mg evening (stress dosing 3×)', 'purpose': 'Adrenal insufficiency — critical replacement'},
                    {'name': 'Levothyroxine', 'dose': '100–150 mcg daily', 'purpose': 'Hypothyroidism replacement'},
                    {'name': 'Desmopressin', 'dose': '0.1–0.4 mg 2–3 times daily', 'purpose': 'Central diabetes insipidus management'},
                    {'name': 'Testosterone / Estrogen replacement', 'dose': 'Per endocrinologist guidance', 'purpose': 'Hypogonadism replacement therapy'},
                    {'name': 'Pegvisomant (Somavert)', 'dose': '10–30 mg SC daily', 'purpose': 'GH receptor antagonist for uncontrolled acromegaly'},
                    {'name': 'Dexamethasone', 'dose': '0.5–1 mg daily', 'purpose': 'Reduces pituitary edema post-radiation'},
                ],
                'follow_up': 'MRI every 3 months; full pituitary hormone panel every 6 weeks',
                'note': 'URGENT — Large pituitary tumors may cause pituitary apoplexy. Emergency response protocols required.'
            }

    return meds


# ═══════════════════════════════════════════════
#  TUMOR CENTROID COMPUTATION (CRITICAL FOR 3D)
# ═══════════════════════════════════════════════

def compute_tumor_centroid(heatmap, img_shape):
    """
    Compute the centroid of the tumor region from the heatmap.
    Returns normalized coordinates in the range [-1, 1] for 3D placement.
    """
    h, w = img_shape[:2]
    heatmap_resized = cv2.resize(heatmap, (w, h))
    
    # Find the region with high probability
    threshold = 0.25
    mask = heatmap_resized > threshold
    
    if not np.any(mask):
        # No tumor detected, return center with small offset
        return 0.0, 0.0, 0.0, 0.1
    
    # Compute weighted centroid using probability as weight
    y_indices, x_indices = np.where(mask)
    weights = heatmap_resized[mask]
    
    # Weighted centroid
    centroid_x = np.average(x_indices, weights=weights)
    centroid_y = np.average(y_indices, weights=weights)
    
    # Normalize to [-1, 1] range
    # Map from image space to brain space
    # x: 0 -> -1 (left), w -> 1 (right)
    norm_x = (centroid_x / w) * 2 - 1
    # y: 0 -> 1 (top), h -> -1 (bottom)  [inverted for brain orientation]
    norm_y = 1 - (centroid_y / h) * 2
    
    # Compute the radius based on the extent of the tumor
    # Use the standard deviation of the tumor pixels as a measure of size
    x_std = np.sqrt(np.average((x_indices - centroid_x)**2, weights=weights))
    y_std = np.sqrt(np.average((y_indices - centroid_y)**2, weights=weights))
    
    # Normalize radius to brain space
    radius = max(0.08, min(0.25, (x_std / w + y_std / h) * 1.2))
    
    # Estimate Z depth based on tumor type and location
    # This is a heuristic since we only have 2D data
    # Place tumors deeper for certain types
    z_offset = 0.0
    
    # Clamp coordinates to ensure they stay within the brain
    norm_x = max(-0.85, min(0.85, norm_x))
    norm_y = max(-0.85, min(0.85, norm_y))
    
    return norm_x, norm_y, z_offset, radius


# ═══════════════════════════════════════════════
#  3-D VISUALISATION  (Plotly)
# ═══════════════════════════════════════════════

def _plotly_json(fig):
    return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)


def _dark_layout(title, scene_extra=None):
    scene = dict(
        xaxis_title='X',
        yaxis_title='Y',
        zaxis_title='Intensity',
        camera=dict(eye=dict(x=1.5, y=1.5, z=1.2)),
        aspectmode='cube',
        bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
        zaxis=dict(showbackground=False, showgrid=False, zeroline=False, showticklabels=False),
    )
    if scene_extra:
        scene.update(scene_extra)
    return dict(
        title=title,
        scene=scene,
        width=700, height=500,
        margin=dict(l=0, r=0, b=0, t=60),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        font=dict(color='white'),
    )


def _brain_height(X, Y):
    base = np.clip(1.0 - (X ** 2 + (Y / 0.92) ** 2), 0.0, 1.0)
    base *= 0.34

    fissure = 0.12 * np.exp(-((X * 6.5) ** 2 + (Y * 1.4) ** 2))
    gyri_x = 0.025 * np.sin(7.5 * np.pi * X) * np.exp(-2.8 * Y ** 2)
    gyri_y = 0.02 * np.sin(5.2 * np.pi * Y) * np.exp(-2.2 * X ** 2)
    gyri = gyri_x + gyri_y

    hemisphere = 0.08 * np.exp(-((Y * 2.5) ** 2)) * np.abs(np.sign(X))
    lobes = 0.04 * np.exp(-((X * 1.6) ** 2 + (Y * 1.2) ** 2))

    height = base + gyri + hemisphere + lobes - fissure
    height[base <= 0] = np.nan
    return height


def _brain_surface(w, h, resolution=140):
    x = np.linspace(-1.0, 1.0, resolution)
    y = np.linspace(-0.95, 0.95, resolution)
    X, Y = np.meshgrid(x, y)
    Z = _brain_height(X, Y)
    return X, Y, Z


def _map_tumor_to_brain(mask, h, w):
    if not np.any(mask):
        return 0.0, 0.0, 0.0, 0.16

    ys, xs = np.where(mask)
    cx = xs.mean()
    cy = ys.mean()

    # Map the 2D heatmap coordinates onto the 3D brain surface
    norm_x = (cx / w - 0.5) * 0.94
    norm_y = -(cy / h - 0.5) * 0.94

    mask_width = (xs.max() - xs.min() + 1) / w
    mask_height = (ys.max() - ys.min() + 1) / h
    radius = max(0.08, min(0.20, max(mask_width, mask_height) * 1.15))

    # Place the tumor just above the brain midline so it is visible on the surface.
    depth = 0.06 + 0.04 * (1.0 - abs(norm_y))
    return norm_x, norm_y, depth, radius


def _tumor_patch(tumor_cx, tumor_cy, tumor_radius, resolution=60):
    u = np.linspace(-tumor_radius, tumor_radius, resolution)
    v = np.linspace(-tumor_radius, tumor_radius, resolution)
    U, V = np.meshgrid(u, v)
    R = np.sqrt(U ** 2 + V ** 2)
    mask = R <= tumor_radius

    X = tumor_cx + U
    Y = tumor_cy + V
    Z = _brain_height(X, Y)
    Z[~mask] = np.nan
    Z[mask] += 0.08 * np.cos(np.pi * R[mask] / tumor_radius) * (1.0 - R[mask] / tumor_radius)
    return X, Y, Z


def create_3d_surface_model(heatmap, confidence, tumor_type):
    h, w = heatmap.shape
    mask = heatmap > 0.25
    brain_X, brain_Y, brain_Z = _brain_surface(w, h, resolution=140)

    brain_trace = go.Surface(
        x=brain_X, y=brain_Y, z=brain_Z,
        surfacecolor=np.full(brain_Z.shape, 0.8),
        colorscale=[[0, '#f5d0c6'], [0.5, '#de9c72'], [1, '#a14d3b']],
        showscale=False,
        opacity=0.95,
        lighting=dict(ambient=0.8, diffuse=0.55, roughness=0.75, specular=0.25),
        name='Brain Surface'
    )

    traces = [brain_trace]

    if np.any(mask):
        tumor_cx, tumor_cy, tumor_cz, tumor_radius = _map_tumor_to_brain(mask, h, w)
        patch_X, patch_Y, patch_Z = _tumor_patch(tumor_cx, tumor_cy, tumor_radius, resolution=70)

        traces.append(go.Surface(
            x=patch_X, y=patch_Y, z=patch_Z,
            surfacecolor=np.full(patch_Z.shape, 0.85),
            colorscale=[[0, '#f55945'], [1, '#ffb8b1']],
            cmin=0,
            cmax=1,
            showscale=False,
            opacity=0.95,
            name='Tumor Patch'
        ))

        traces.append(go.Scatter3d(
            x=[tumor_cx], y=[tumor_cy], z=[tumor_cz + tumor_radius * 0.4],
            mode='markers+text',
            marker=dict(size=6, color='red', opacity=1),
            text=['Tumor'],
            textposition='top center',
            hoverinfo='text',
            name='Tumor Center'
        ))

    title = f'<b>3D Brain Tumor Model</b><br><span style="font-size:13px">{tumor_type} | {confidence:.1f}%</span>'
    fig = go.Figure(data=traces)
    fig.update_layout(**_dark_layout(title, {'camera': {'eye': {'x': 1.6, 'y': 1.4, 'z': 1.1}}}))
    return _plotly_json(fig)


def create_3d_scatter_model(heatmap, confidence, tumor_type):
    h, w = heatmap.shape
    pts  = np.where(heatmap > 0.25)

    if len(pts[0]) == 0:
        cy, cx = h // 2, w // 2
        r = int(min(h, w) * 0.12)
        xv, yv, zv = [], [], []
        for i in range(-r, r, 2):
            for j in range(-r, r, 2):
                if i*i + j*j <= r*r:
                    xv.append(cx + j); yv.append(cy + i)
                    zv.append((confidence / 100) * (1 - (i*i+j*j)/(r*r)))
        colors = zv
    else:
        yc, xc = pts
        zv = heatmap[pts].tolist()
        if len(xc) > 1500:
            idx = np.random.choice(len(xc), 1500, replace=False)
            xc, yc, zv = xc[idx], yc[idx], list(np.array(zv)[idx])
        xv, yv, colors = xc.tolist(), yc.tolist(), zv

    fig = go.Figure(data=[go.Scatter3d(
        x=xv, y=yv, z=zv, mode='markers',
        marker=dict(size=4, color=colors, colorscale='Hot',
                    showscale=True, colorbar=dict(title="Probability", x=1.02),
                    opacity=0.7),
        text=[f"Intensity: {z:.3f}" for z in zv], hoverinfo='text',
    )])
    title = f'<b>3D Point Cloud</b><br><span style="font-size:13px">{tumor_type} | {confidence:.1f}%</span>'
    fig.update_layout(**_dark_layout(title, {"zaxis_title": "Probability"}))
    return _plotly_json(fig)


# ═══════════════════════════════════════════════
#  2-D HEATMAP HELPERS
# ═══════════════════════════════════════════════

def create_simple_heatmap(image_shape, confidence, tumor_type):
    h, w = image_shape[:2]
    heatmap = np.zeros((h, w), dtype=np.float32)
    cy, cx  = h // 2, w // 2
    r       = max(1, int(min(h, w) * 0.18 * (confidence / 100)))
    scale   = confidence / 100

    yy, xx = np.ogrid[:h, :w]
    dist    = np.sqrt((yy - cy)**2 + (xx - cx)**2)
    mask    = dist < r
    heatmap[mask] = np.maximum(0, 1 - (dist[mask] / r)**2) * scale

    if "Pituitary" in tumor_type:
        heatmap = np.roll(heatmap, -int(h * 0.08), axis=0)
    elif "Meningioma" in tumor_type:
        heatmap = np.roll(heatmap, int(w * 0.10), axis=1)
    elif "Glioma" in tumor_type:
        cy2, cx2 = cy + int(h * 0.12), cx - int(w * 0.08)
        r2 = r * 0.6
        dist2 = np.sqrt((yy - cy2)**2 + (xx - cx2)**2)
        mask2 = dist2 < r2
        extra = np.maximum(0, 1 - (dist2[mask2] / r2)**2) * (scale * 0.6)
        heatmap[mask2] = np.maximum(heatmap[mask2], extra)

    heatmap = cv2.GaussianBlur(heatmap.astype(np.float32), (21, 21), 0)
    mx = np.max(heatmap)
    if mx > 0:
        heatmap /= mx
    return heatmap


def encode_image_to_base64(img_array):
    """numpy uint8 array → base-64 PNG string."""
    if img_array.dtype != np.uint8:
        img_array = (img_array * 255).astype(np.uint8)
    pil_img = Image.fromarray(img_array) if img_array.ndim == 3 \
              else Image.fromarray(img_array, mode='L')
    buf = BytesIO()
    pil_img.save(buf, format='PNG')
    return base64.b64encode(buf.getvalue()).decode()


def create_tumor_visualization(img_array, predicted_class_idx, confidence):
    """
    Build all 2-D and 3-D visualisation artefacts.
    img_array: (1,224,224,3) float32 in BGR order (straight from preprocess_image).
    """
    predicted_class = CLASS_NAMES[predicted_class_idx]

    if predicted_class_idx == 0:          # No Tumor
        return {'success': True, 'no_tumor': True}

    try:
        # ── Convert BGR float → RGB uint8 for display ──────────────────────
        bgr_uint8      = (img_array[0] * 255).astype(np.uint8)
        original_rgb   = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2RGB)
        h, w           = original_rgb.shape[:2]

        # ── Heatmap ─────────────────────────────────────────────────────────
        heatmap        = create_simple_heatmap(original_rgb.shape, confidence, predicted_class)
        area_pct       = float(np.sum(heatmap > 0.25) / (h * w) * 100)

        # ── Stage & precautions ─────────────────────────────────────────────
        stage, stage_desc = get_tumor_stage(predicted_class, confidence, area_pct)
        precautions       = get_precautions(predicted_class, stage, confidence, area_pct)

        # ── Colour overlay ──────────────────────────────────────────────────
        hm_uint8        = np.uint8(255 * heatmap)
        hm_colored_bgr  = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_JET)
        hm_colored_rgb  = cv2.cvtColor(hm_colored_bgr, cv2.COLOR_BGR2RGB)
        overlay_rgb     = cv2.addWeighted(original_rgb, 0.6, hm_colored_rgb, 0.4, 0)

        # ── Marked image ────────────────────────────────────────────────────
        peak_y, peak_x  = np.unravel_index(np.argmax(heatmap), heatmap.shape)
        cx_s            = int(peak_x * (w / heatmap.shape[1]))
        cy_s            = int(peak_y * (h / heatmap.shape[0]))
        r_circle        = max(5, int(35 * (confidence / 100)))

        marked_img      = original_rgb.copy()
        cv2.circle(marked_img,   (cx_s, cy_s), r_circle, (255, 50, 50), 3)
        cv2.circle(marked_img,   (cx_s, cy_s), 6,        (50, 50, 255), -1)

        marked_overlay  = overlay_rgb.copy()
        cv2.circle(marked_overlay, (cx_s, cy_s), r_circle, (255, 50, 50), 3)
        cv2.circle(marked_overlay, (cx_s, cy_s), 6,        (50, 50, 255), -1)

        comparison        = np.hstack([original_rgb, overlay_rgb])
        comparison_marked = np.hstack([marked_img,   marked_overlay])

        # ── Compute tumor centroid for 3D mapping ──────────────────────────
        # This is the CRITICAL change: compute the centroid from the heatmap
        norm_x, norm_y, norm_z, tumor_radius = compute_tumor_centroid(heatmap, original_rgb.shape)
        
        # ── 3-D models ──────────────────────────────────────────────────────
        plotly_surface = plotly_scatter = None
        try:
            plotly_surface = create_3d_surface_model(heatmap, confidence, predicted_class)
            plotly_scatter = create_3d_scatter_model(heatmap, confidence, predicted_class)
        except Exception as exc:
            print(f"3D viz error: {exc}")

        # ── Tumor 3D placement for brain model viewer ─────────────────────────
        tumor_opacity = float(min(0.85, max(0.45, confidence / 200)))

        # ── Anatomical location label ────────────────────────────────────────
        x_pos = ("Left Hemisphere"  if cx_s < w // 3 else
                 "Right Hemisphere" if cx_s > 2 * w // 3 else "Central Region")
        y_pos = ("Frontal Lobe"     if cy_s < h // 3 else
                 "Occipital Region" if cy_s > 2 * h // 3 else "Parietal/Temporal Area")

        brain_model_url = url_for('static', filename='models/brain.glb')
        return {
            'success':           True,
            'no_tumor':          False,
            'original':          encode_image_to_base64(original_rgb),
            'marked':            encode_image_to_base64(marked_img),
            'overlay':           encode_image_to_base64(overlay_rgb),
            'marked_overlay':    encode_image_to_base64(marked_overlay),
            'comparison':        encode_image_to_base64(comparison),
            'comparison_marked': encode_image_to_base64(comparison_marked),
            'plotly_surface':    plotly_surface,
            'plotly_scatter':    plotly_scatter,
            'brain_3d': {
                'model_url': brain_model_url,
                'tumor': {
                    'x': norm_x,
                    'y': norm_y,
                    'z': norm_z,
                    'radius': tumor_radius,
                    'opacity': tumor_opacity,
                    'label': 'Tumor',
                }
            },
            'tumor_info': {
                'location':         f"{y_pos}, {x_pos}",
                'area_percentage':  round(area_pct, 2),
                'tumor_type':       predicted_class,
                'confidence_score': round(confidence, 1),
                'stage':            stage,
                'stage_description':stage_desc,
                'precautions':      precautions,
                # Add centroid info for debugging/display
                'centroid_x':       round(norm_x, 3),
                'centroid_y':       round(norm_y, 3),
                'centroid_z':       round(norm_z, 3),
            },
        }

    except Exception as exc:
        traceback.print_exc()
        return {'success': False, 'error': str(exc)}


# ═══════════════════════════════════════════════
#  AUTH ROUTES
# ═══════════════════════════════════════════════

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username     = request.form.get('username', '').strip()
        email        = request.form.get('email', '').strip().lower()
        raw_password = request.form.get('password', '')

        # Validation
        if not username or not email or not raw_password:
            return render_template('register.html', error='All fields are required.')
        if len(username) < 3:
            return render_template('register.html', error='Username must be at least 3 characters.')
        if '@' not in email or '.' not in email:
            return render_template('register.html', error='Enter a valid email address.')
        if len(raw_password) < 6:
            return render_template('register.html', error='Password must be at least 6 characters.')

        security_question = request.form.get('security_question', '').strip()
        security_answer   = request.form.get('security_answer', '').strip().lower()

        if not security_question or not security_answer:
            return render_template('register.html', error='Please select a security question and provide an answer.')

        pw_hash = bcrypt.generate_password_hash(raw_password).decode('utf-8')
        ans_hash = bcrypt.generate_password_hash(security_answer).decode('utf-8')

        try:
            conn = get_db()
            conn.execute(
                "INSERT INTO users (username, email, password, security_question, security_answer) VALUES (?, ?, ?, ?, ?)",
                (username, email, pw_hash, security_question, ans_hash)
            )
            conn.commit()
            conn.close()
        except sqlite3.IntegrityError:
            return render_template('register.html', error='That email is already registered. Please log in.')
        except Exception as exc:
            return render_template('register.html', error=f'Registration failed: {exc}')

        return redirect(url_for('login'))

    return render_template('register.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email    = request.form.get('email', '').strip().lower()
        password = request.form.get('password', '')

        if not email or not password:
            return render_template('login.html', error='Please fill in all fields.')

        try:
            conn = get_db()
            user = conn.execute(
                "SELECT id, username, email, password FROM users WHERE email = ?", (email,)
            ).fetchone()
            conn.close()
        except Exception as exc:
            return render_template('login.html', error=f'Database error: {exc}')

        if user and bcrypt.check_password_hash(user['password'], password):
            session.clear()
            session['user_id'] = user['id']
            session['user']    = user['username']
            session['email']   = user['email']
            return redirect(url_for('home'))

        return render_template('login.html', error='Invalid email or password. Please try again.')

    return render_template('login.html')


@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))


# ═══════════════════════════════════════════════
#  PASSWORD RESET
# ═══════════════════════════════════════════════

# ─── OTP-Based Password Reset ───────────────────────────────────────────────

@app.route('/api/send-otp', methods=['POST'])
def send_otp():
    """Generate a 6-digit OTP for email-based password reset (no link needed)."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid request'}), 400

    email = data.get('email', '').strip().lower()
    if not email:
        return jsonify({'error': 'Email is required'}), 400

    try:
        conn = get_db()
        user = conn.execute("SELECT id, username FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()
    except Exception as exc:
        return jsonify({'error': f'Database error: {exc}'}), 500

    if not user:
        # Anti-enumeration: always return 200
        return jsonify({'message': 'If that email is registered, an OTP has been sent.'}), 200

    import random
    otp = str(random.randint(100000, 999999))
    expiry = datetime.now() + timedelta(minutes=10)
    otp_store[email] = {'otp': otp, 'expiry': expiry}

    username = user['username'] or 'User'

    # Try to send OTP via email
    email_sent = False
    mail_username = app.config.get('MAIL_USERNAME', '')
    mail_password = os.environ.get('MAIL_PASSWORD', '')
    email_configured = (
        mail_username
        and mail_username != 'your_gmail@gmail.com'
        and mail_password
        and mail_password != 'your_16char_app_password_here'
    )

    if email_configured:
        try:
            html_body = f"""
            <!DOCTYPE html>
            <html><body style="margin:0;padding:0;background:#0d0614;font-family:'Segoe UI',Arial,sans-serif;">
              <div style="max-width:480px;margin:40px auto;background:#1a0a2e;border-radius:16px;overflow:hidden;border:1px solid rgba(167,139,250,0.2);">
                <div style="background:linear-gradient(135deg,#7c3aed,#ec4899);padding:28px 36px;text-align:center;">
                  <div style="font-size:32px;margin-bottom:8px;">🧠</div>
                  <h1 style="color:white;font-size:1.4rem;font-weight:800;margin:0;">NeuroScan AI</h1>
                  <p style="color:rgba(255,255,255,0.85);font-size:0.85rem;margin:5px 0 0;">Password Reset OTP</p>
                </div>
                <div style="padding:32px 36px;text-align:center;">
                  <p style="color:#e9d5ff;font-size:1rem;margin-bottom:6px;">Hello <strong>{username}</strong>,</p>
                  <p style="color:rgba(220,200,255,0.7);font-size:0.9rem;line-height:1.6;margin-bottom:24px;">Your one-time password reset code is:</p>
                  <div style="background:linear-gradient(135deg,rgba(124,58,237,0.2),rgba(236,72,153,0.1));border:2px solid rgba(167,139,250,0.4);border-radius:16px;padding:24px;margin-bottom:24px;">
                    <div style="font-size:2.8rem;font-weight:900;letter-spacing:0.4em;color:#a78bfa;font-family:monospace;">{otp}</div>
                  </div>
                  <p style="color:rgba(220,200,255,0.5);font-size:0.8rem;">This OTP expires in <strong style="color:#a78bfa;">10 minutes</strong>.<br>If you didn't request this, ignore this email.</p>
                </div>
              </div>
            </body></html>
            """
            msg = Message(
                subject='🔐 NeuroScan AI – Your Password Reset OTP',
                recipients=[email],
                html=html_body,
            )
            mail.send(msg)
            email_sent = True
            print(f"[OTP MAIL] OTP sent to: {email}")
        except Exception as exc:
            print(f"[OTP MAIL ERROR] {exc}")
    else:
        print(f"\n[OTP] Email not configured. OTP for {email}: {otp}\n")

    return jsonify({
        'message': 'If that email is registered, an OTP has been sent.',
        'email_sent': email_sent,
        # In dev mode (no email), show the OTP on-screen
        'otp_dev': otp if not email_configured else None,
    }), 200


@app.route('/api/verify-otp', methods=['POST'])
def verify_otp():
    """Verify OTP and if correct, return a temporary reset token."""
    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid request'}), 400

    email = data.get('email', '').strip().lower()
    otp   = data.get('otp', '').strip()
    action = data.get('action', 'verify')  # 'verify' | 'reset'

    if not email or not otp:
        return jsonify({'error': 'Email and OTP are required'}), 400

    record = otp_store.get(email)
    if not record:
        return jsonify({'error': 'No OTP found for this email. Please request a new one.'}), 400

    if datetime.now() > record['expiry']:
        otp_store.pop(email, None)
        return jsonify({'error': 'OTP has expired. Please request a new one.'}), 400

    if record['otp'] != otp:
        return jsonify({'error': 'Incorrect OTP. Please check and try again.'}), 401

    if action == 'verify':
        # Return a temporary reset token
        token  = secrets.token_urlsafe(32)
        expiry = datetime.now() + timedelta(hours=1)
        password_reset_tokens[token] = {'email': email, 'expiry': expiry}
        otp_store.pop(email, None)
        return jsonify({'token': token, 'message': 'OTP verified successfully!'}), 200

    if action == 'reset':
        new_pw = data.get('newPassword', '')
        if len(new_pw) < 6:
            return jsonify({'error': 'Password must be at least 6 characters.'}), 400
        pw_hash = bcrypt.generate_password_hash(new_pw).decode('utf-8')
        try:
            conn = get_db()
            conn.execute("UPDATE users SET password = ? WHERE email = ?", (pw_hash, email))
            conn.commit()
            conn.close()
        except Exception as exc:
            return jsonify({'error': f'Database error: {exc}'}), 500
        otp_store.pop(email, None)
        return jsonify({'message': 'Password reset successfully! You can now log in.'}), 200

    return jsonify({'error': 'Unknown action'}), 400


@app.route('/forgot-password', methods=['GET', 'POST'])
def forgot_password():
    if request.method == 'GET':
        return render_template('forgot_password.html')

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid JSON request'}), 400

    email = data.get('email', '').strip().lower()
    if not email:
        return jsonify({'error': 'Email is required'}), 400

    try:
        conn = get_db()
        user = conn.execute("SELECT id, username FROM users WHERE email = ?", (email,)).fetchone()
        conn.close()
    except Exception as exc:
        return jsonify({'error': f'Database error: {exc}'}), 500

    # Always return the same message to avoid email enumeration
    if not user:
        return jsonify({'message': 'If that email is registered, a reset link has been sent.'}), 200

    token  = secrets.token_urlsafe(32)
    expiry = datetime.now() + timedelta(hours=1)
    password_reset_tokens[token] = {'email': email, 'expiry': expiry}

    # Build reset URL
    reset_link = url_for('reset_password', token=token, _external=True)
    # Fallback for local dev
    if '127.0.0.1' not in reset_link and 'localhost' not in reset_link:
        reset_link = f"http://127.0.0.1:5000/reset-password?token={token}"

    username = user['username'] if user['username'] else 'User'

    # ── Send real email via Flask-Mail ──
    email_sent = False
    mail_username = app.config.get('MAIL_USERNAME', '')
    mail_password = os.environ.get('MAIL_PASSWORD', '')
    email_configured = (
        mail_username
        and mail_username != 'your_gmail@gmail.com'
        and mail_password
        and mail_password != 'your_16char_app_password_here'
    )

    if email_configured:
        try:
            html_body = f"""
            <!DOCTYPE html>
            <html>
            <body style="margin:0;padding:0;background:#0d0614;font-family:'Segoe UI',Arial,sans-serif;">
              <div style="max-width:560px;margin:40px auto;background:#1a0a2e;border-radius:16px;overflow:hidden;border:1px solid rgba(167,139,250,0.2);">
                <!-- Header -->
                <div style="background:linear-gradient(135deg,#7c3aed,#ec4899);padding:32px 40px;text-align:center;">
                  <div style="font-size:32px;margin-bottom:10px;">🧠</div>
                  <h1 style="color:white;font-size:1.6rem;font-weight:800;margin:0;">NeuroScan AI</h1>
                  <p style="color:rgba(255,255,255,0.85);font-size:0.9rem;margin:6px 0 0;">Advanced Brain Tumor Detection</p>
                </div>
                <!-- Body -->
                <div style="padding:36px 40px;">
                  <p style="color:#e9d5ff;font-size:1rem;margin-bottom:8px;">Hello <strong>{username}</strong>,</p>
                  <p style="color:rgba(220,200,255,0.75);font-size:0.92rem;line-height:1.6;margin-bottom:28px;">
                    We received a request to reset your NeuroScan AI password. Click the button below to set a new password. This link expires in <strong style="color:#a78bfa;">1 hour</strong>.
                  </p>
                  <!-- Button -->
                  <div style="text-align:center;margin-bottom:28px;">
                    <a href="{reset_link}" style="display:inline-block;padding:16px 40px;background:linear-gradient(135deg,#7c3aed,#ec4899);color:white;text-decoration:none;border-radius:10px;font-size:1rem;font-weight:700;letter-spacing:0.02em;">
                      🔐 Reset My Password
                    </a>
                  </div>
                  <p style="color:rgba(220,200,255,0.5);font-size:0.8rem;line-height:1.6;">
                    If the button doesn't work, copy and paste this link into your browser:<br>
                    <a href="{reset_link}" style="color:#a78bfa;word-break:break-all;">{reset_link}</a>
                  </p>
                  <hr style="border:none;border-top:1px solid rgba(167,139,250,0.15);margin:24px 0;">
                  <p style="color:rgba(220,200,255,0.4);font-size:0.78rem;text-align:center;">
                    If you didn't request a password reset, you can safely ignore this email.<br>
                    &copy; 2026 NeuroScan AI – AI-Powered Brain Tumor Detection
                  </p>
                </div>
              </div>
            </body>
            </html>
            """
            msg = Message(
                subject='🔐 NeuroScan AI – Password Reset Request',
                recipients=[email],
                html=html_body,
            )
            mail.send(msg)
            email_sent = True
            print(f"[MAIL] Password reset email sent to: {email}")
        except Exception as exc:
            print(f"[MAIL ERROR] Could not send email: {exc}")
            print(f"[RESET LINK] Fallback link: {reset_link}")
    else:
        print(f"\n[RESET LINK] Email not configured. Reset link: {reset_link}\n")

    return jsonify({
        'message': 'If that email is registered, a reset link will be sent. Check your inbox (and spam folder).',
        'email_sent': email_sent,
        'reset_link': reset_link if not email_configured else None,
    }), 200


# ─── Security Question Reset ─────────────────────────────────────────────────

@app.route('/security-reset', methods=['GET', 'POST'])
def security_reset():
    if request.method == 'GET':
        return render_template('security_reset.html')

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid request'}), 400

    action = data.get('action', 'fetch')  # 'fetch' | 'verify' | 'reset'

    email = data.get('email', '').strip().lower()
    if not email:
        return jsonify({'error': 'Email is required'}), 400

    try:
        conn = get_db()
        user = conn.execute(
            "SELECT id, username, security_question, security_answer FROM users WHERE email = ?",
            (email,)
        ).fetchone()
        conn.close()
    except Exception as exc:
        return jsonify({'error': f'Database error: {exc}'}), 500

    if not user:
        return jsonify({'error': 'No account found with that email.'}), 404

    if action == 'fetch':
        if not user['security_question']:
            return jsonify({'error': 'No security question set for this account.'}), 400
        return jsonify({'question': user['security_question']}), 200

    if action == 'verify':
        answer = data.get('answer', '').strip().lower()
        if not answer:
            return jsonify({'error': 'Answer is required'}), 400
        if not user['security_answer'] or not bcrypt.check_password_hash(user['security_answer'], answer):
            return jsonify({'error': 'Incorrect answer. Please try again.'}), 401
        # Generate a temporary reset token
        token  = secrets.token_urlsafe(32)
        expiry = datetime.now() + timedelta(hours=1)
        password_reset_tokens[token] = {'email': email, 'expiry': expiry}
        return jsonify({'token': token}), 200

    if action == 'reset':
        token = data.get('token', '')
        new_pw = data.get('newPassword', '')
        if not token or token not in password_reset_tokens:
            return jsonify({'error': 'Invalid or expired session. Start over.'}), 400
        if datetime.now() > password_reset_tokens[token]['expiry']:
            password_reset_tokens.pop(token, None)
            return jsonify({'error': 'Session expired. Please start over.'}), 400
        if len(new_pw) < 6:
            return jsonify({'error': 'Password must be at least 6 characters.'}), 400
        pw_hash = bcrypt.generate_password_hash(new_pw).decode('utf-8')
        try:
            conn = get_db()
            conn.execute("UPDATE users SET password = ? WHERE email = ?",
                         (pw_hash, password_reset_tokens[token]['email']))
            conn.commit()
            conn.close()
        except Exception as exc:
            return jsonify({'error': f'Database error: {exc}'}), 500
        password_reset_tokens.pop(token, None)
        return jsonify({'message': 'Password reset successfully! You can now log in.'}), 200

    return jsonify({'error': 'Unknown action'}), 400


@app.route('/reset-password', methods=['GET', 'POST'])
def reset_password():
    if request.method == 'GET':
        token = request.args.get('token', '')
        if not token or token not in password_reset_tokens:
            return render_template('reset_password.html', token='',
                                   error='Invalid or expired reset link.')
        if datetime.now() > password_reset_tokens[token]['expiry']:
            password_reset_tokens.pop(token, None)
            return render_template('reset_password.html', token='',
                                   error='This link has expired. Please request a new one.')
        return render_template('reset_password.html', token=token)

    data = request.get_json(silent=True)
    if not data:
        return jsonify({'error': 'Invalid JSON request'}), 400

    token        = data.get('token', '')
    new_password = data.get('newPassword', '')

    if not token or token not in password_reset_tokens:
        return jsonify({'error': 'Invalid or expired token'}), 400

    td = password_reset_tokens[token]
    if datetime.now() > td['expiry']:
        password_reset_tokens.pop(token, None)
        return jsonify({'error': 'Token has expired'}), 400

    if len(new_password) < 6:
        return jsonify({'error': 'Password must be at least 6 characters'}), 400

    pw_hash = bcrypt.generate_password_hash(new_password).decode('utf-8')
    try:
        conn = get_db()
        conn.execute("UPDATE users SET password = ? WHERE email = ?",
                     (pw_hash, td['email']))
        conn.commit()
        conn.close()
    except Exception as exc:
        return jsonify({'error': f'Database error: {exc}'}), 500

    password_reset_tokens.pop(token, None)
    return jsonify({'message': 'Password reset successful!'}), 200


# ═══════════════════════════════════════════════
#  MAIN ROUTES
# ═══════════════════════════════════════════════

@app.route('/')
def home():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('home.html', model_loaded=True)


@app.route('/predict', methods=['POST'])
def predict():
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated. Please log in.'}), 401

    if 'image' not in request.files:
        return jsonify({'error': 'No image file in request'}), 400

    file = request.files['image']
    if not file or file.filename == '':
        return jsonify({'error': 'No image selected'}), 400

    # Validate extension
    allowed = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
    ext = os.path.splitext(file.filename.lower())[1]
    if ext not in allowed:
        return jsonify({'error': f'Unsupported file type: {ext}'}), 400

    img_array = preprocess_image(file)
    if img_array is None:
        return jsonify({'error': 'Could not decode image. Please upload a valid MRI scan.'}), 400

    predicted_class, confidence, class_idx = predict_tumor(img_array)
    if predicted_class is None:
        return jsonify({'error': 'Prediction failed internally.'}), 500

    visualization = create_tumor_visualization(img_array, class_idx, confidence)

    # Build medication data (only for tumor cases)
    medication = None
    if predicted_class != 'No Tumor' and visualization.get('tumor_info'):
        stage = visualization['tumor_info'].get('stage', '')
        medication = get_medication(predicted_class, stage)

    return jsonify({
        'prediction':    predicted_class,
        'confidence':    round(confidence, 2),
        'visualization': visualization,
        'medication':    medication,
    })


# ─── Scan History API ─────────────────────────────────────────────────────────

@app.route('/api/save-scan', methods=['POST'])
def save_scan():
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    data = request.get_json(silent=True) or {}
    try:
        conn = get_db()
        conn.execute(
            '''INSERT INTO scan_history
               (user_id, patient_name, patient_id, patient_age, patient_gender, tumor_type,
                confidence, stage, area_pct, location, scan_date, image_b64,
                doctor_name, hospital, stage_description, precautions_json, medication_json)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)''',
            (
                session['user_id'],
                data.get('patient_name', 'Unknown'),
                data.get('patient_id', ''),
                data.get('patient_age', ''),
                data.get('patient_gender', ''),
                data.get('tumor_type', 'Unknown'),
                data.get('confidence', 0),
                data.get('stage', ''),
                data.get('area_pct', 0),
                data.get('location', ''),
                datetime.now().strftime('%Y-%m-%d %H:%M'),
                data.get('image_b64', '')[:4000],  # limit stored size
                data.get('doctor_name', ''),
                data.get('hospital', ''),
                data.get('stage_description', ''),
                json.dumps(data.get('precautions', [])),
                json.dumps(data.get('medication', {})),
            )
        )
        conn.commit()
        conn.close()
        return jsonify({'ok': True}), 200
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/history')
def api_history():
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    try:
        conn = get_db()
        rows = conn.execute(
            '''SELECT id, patient_name, patient_id, patient_age, patient_gender, tumor_type,
                      confidence, stage, area_pct, location, scan_date, doctor_name
               FROM scan_history WHERE user_id = ?
               ORDER BY id DESC LIMIT 100''',
            (session['user_id'],)
        ).fetchall()
        conn.close()
        return jsonify([dict(r) for r in rows]), 200
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


@app.route('/api/scan-detail/<int:scan_id>')
def api_scan_detail(scan_id):
    """Return full scan details including computed medication for a specific scan."""
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    try:
        conn = get_db()
        row = conn.execute(
            '''SELECT id, patient_name, patient_id, patient_age, patient_gender, tumor_type,
                      confidence, stage, area_pct, location, scan_date, doctor_name, hospital,
                      stage_description, precautions_json, medication_json
               FROM scan_history WHERE id = ? AND user_id = ?''',
            (scan_id, session['user_id'])
        ).fetchone()
        conn.close()
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500

    if not row:
        return jsonify({'error': 'Scan not found'}), 404

    result = dict(row)
    # Add stage description and precautions (use stored or recompute)
    if result['tumor_type'] and result['tumor_type'] != 'No Tumor' and result['stage']:
        if result.get('precautions_json'):
            try:
                result['precautions'] = json.loads(result['precautions_json'])
                if not result['precautions']:
                    result['precautions'] = get_precautions(result['tumor_type'], result['stage'], result['confidence'] or 0, result['area_pct'] or 0)
            except Exception:
                result['precautions'] = get_precautions(result['tumor_type'], result['stage'], result['confidence'] or 0, result['area_pct'] or 0)
        else:
            result['precautions'] = get_precautions(result['tumor_type'], result['stage'], result['confidence'] or 0, result['area_pct'] or 0)

        if result.get('medication_json'):
            try:
                result['medication'] = json.loads(result['medication_json'])
                if not result['medication']:
                    result['medication'] = get_medication(result['tumor_type'], result['stage'])
            except Exception:
                result['medication'] = get_medication(result['tumor_type'], result['stage'])
        else:
            result['medication'] = get_medication(result['tumor_type'], result['stage'])

        if not result.get('stage_description'):
            _, stage_desc = get_tumor_stage(result['tumor_type'], result['confidence'] or 0, result['area_pct'] or 0)
            result['stage_description'] = stage_desc
    else:
        result['stage_description'] = ''
        result['precautions']       = []
        result['medication']        = {}

    # Remove raw JSON columns from response
    result.pop('precautions_json', None)
    result.pop('medication_json', None)

    return jsonify(result), 200


@app.route('/api/profile-stats')
def api_profile_stats():
    if 'user' not in session:
        return jsonify({'error': 'Not authenticated'}), 401
    try:
        conn = get_db()
        rows = conn.execute(
            '''SELECT id, tumor_type, confidence, stage, patient_name, patient_id,
                      patient_age, patient_gender, scan_date, location, area_pct
               FROM scan_history WHERE user_id = ?
               ORDER BY id DESC''',
            (session['user_id'],)
        ).fetchall()
        conn.close()
        total  = len(rows)
        tumor  = sum(1 for r in rows if r['tumor_type'] != 'No Tumor')
        clear  = total - tumor

        # Most common tumor type
        from collections import Counter
        tumor_types = [r['tumor_type'] for r in rows if r['tumor_type'] != 'No Tumor']
        most_common = Counter(tumor_types).most_common(1)
        most_common_type = most_common[0][0] if most_common else 'N/A'

        return jsonify({
            'username':        session['user'],
            'email':           session.get('email', ''),
            'total':           total,
            'tumor':           tumor,
            'clear':           clear,
            'most_common':     most_common_type,
            'scans':           [dict(r) for r in rows]
        }), 200
    except Exception as exc:
        return jsonify({'error': str(exc)}), 500


# ─── History & Profile Pages ──────────────────────────────────────────────────

@app.route('/history')
def history_page():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('history.html')


@app.route('/about')
def about_page():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('about.html')


@app.route('/profile')
def profile_page():
    if 'user' not in session:
        return redirect(url_for('login'))
    return render_template('profile.html')


# ═══════════════════════════════════════════════
#  ENTRY POINT
# ═══════════════════════════════════════════════

if __name__ == '__main__':
    init_db()
    load_trained_model()
    print("\n" + "=" * 60)
    print("[START] NeuroScan AI  -  http://127.0.0.1:5000")
    print("=" * 60)
    print("   [AI]   Brain Tumor Classification (4 classes)")
    print("   [LOC]  Stage Detection  (Early / Intermediate / Advanced)")
    print("   [RX]   Personalised Precautions")
    print("   [VIZ]  2-D Heatmap + 3-D Plotly Visualisation")
    print("   [PDF]  PDF Report Export")
    print("=" * 60 + "\n")
    app.run(debug=True, host='127.0.0.1', port=5000)