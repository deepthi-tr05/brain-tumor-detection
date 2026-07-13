#!/usr/bin/env python3
"""
NeuroScan AI - Brain Tumor Detection System
One-click launcher: checks environment and starts the Flask server.
Compatible with Python 3.11
"""

import subprocess
import sys
import os

# ── Find Python 3.11 (checks common install locations) ──────────────
PYTHON_PATHS = [
    # Standard Python 3.11 install locations on Windows
    r"C:\Users\DeepthiGowda\AppData\Local\Programs\Python\Python311\python.exe",
    r"C:\Python311\python.exe",
    r"C:\Program Files\Python311\python.exe",
    # Fallback to whichever interpreter is running this script
    sys.executable,
]

def find_python():
    """Return the path to a working Python interpreter."""
    for path in PYTHON_PATHS:
        if os.path.exists(path):
            return path
    return sys.executable   # last resort

def print_banner():
    print("\n" + "=" * 65)
    print("[BRAIN] NeuroScan AI - Brain Tumor Detection System")
    print("=" * 65)
    print("   Powered by VGG16 Transfer Learning | Flask Backend")
    print(f"   Python: {sys.version.split()[0]}")
    print("=" * 65)

def check_model():
    """Check if a trained model file exists."""
    paths = [
        'models/brain_tumor_cnn_model.h5',
        'models/best_model.h5',
        'models/brain_tumor_cnn_model.keras',
    ]
    for p in paths:
        if os.path.exists(p):
            print(f"\n[OK] Model found: {p}")
            return True
    print("\n[WARN] No trained model found!")
    print("   Run train_model.py first or place your .h5 model in models/")
    print("   The app will still start but predictions will be disabled.")
    return False

def check_packages():
    """Verify that key packages are importable."""
    required = ['flask', 'tensorflow', 'cv2', 'numpy', 'PIL', 'plotly', 'flask_bcrypt']
    missing = []
    for pkg in required:
        try:
            __import__(pkg)
        except ImportError:
            missing.append(pkg)
    if missing:
        print(f"\n[ERROR] Missing packages: {', '.join(missing)}")
        print("   Run:  pip install -r requirements.txt")
        return False
    print("\n[OK] All required packages found.")
    return True

def main():
    print_banner()
    check_model()

    if not check_packages():
        input("\nPress Enter to exit...")
        return

    print("\n[START] Starting Flask server...")
    print("   Open http://127.0.0.1:5000 in your browser")
    print("   Press Ctrl+C to stop the server")
    print("=" * 65 + "\n")

    python_exe = find_python()
    subprocess.run([python_exe, 'app.py'])

if __name__ == '__main__':
    main()