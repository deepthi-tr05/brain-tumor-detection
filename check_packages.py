import sys
print(f"Python: {sys.version}")
packages = {
    "flask": "Flask",
    "tensorflow": "TensorFlow",
    "cv2": "OpenCV",
    "numpy": "NumPy",
    "PIL": "Pillow",
    "plotly": "Plotly",
    "flask_bcrypt": "Flask-Bcrypt",
    "h5py": "h5py",
    "sklearn": "scikit-learn",
    "scipy": "SciPy",
    "matplotlib": "Matplotlib",
}
all_ok = True
for mod, name in packages.items():
    try:
        m = __import__(mod)
        ver = getattr(m, "__version__", "ok")
        print(f"  OK  {name} == {ver}")
    except ImportError as e:
        print(f"  MISSING  {name}: {e}")
        all_ok = False

print()
if all_ok:
    print("ALL PACKAGES INSTALLED SUCCESSFULLY!")
else:
    print("Some packages are missing - run: pip install -r requirements.txt")
