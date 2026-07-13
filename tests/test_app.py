"""
NeuroScan AI – Test Suite (Updated)
=====================================
Run with:  python -m pytest tests/ -v
Requires:  pip install pytest
"""

import io
import os
import sys
import sqlite3
import tempfile

import pytest

# ── Make sure the project root is on the Python path ──────────────────────────
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Import app utilities (avoids loading TensorFlow model at import time) ─────
import app as app_module
from app import (
    app as flask_app, init_db,
    get_tumor_stage, get_precautions, get_medication,
    preprocess_image, otp_store, password_reset_tokens
)


# ══════════════════════════════════════════════════════════════════════════════
#  FIXTURES
# ══════════════════════════════════════════════════════════════════════════════

@pytest.fixture
def client(tmp_path):
    """Flask test client with a fresh temporary SQLite database."""
    db_file = tmp_path / "test_users.db"
    upload_dir = tmp_path / "uploads"
    upload_dir.mkdir()

    flask_app.config.update({
        "TESTING": True,
        "SECRET_KEY": "test-secret-key",
        "WTF_CSRF_ENABLED": False,
        "UPLOAD_FOLDER": str(upload_dir),
    })

    # Patch DB path and uploads folder for test isolation
    original_db = app_module.DB_PATH
    original_upload = app_module.UPLOAD_FOLDER

    app_module.DB_PATH = str(db_file)
    app_module.UPLOAD_FOLDER = str(upload_dir)

    with flask_app.test_client() as client:
        with flask_app.app_context():
            init_db()
        yield client

    # Restore originals
    app_module.DB_PATH = original_db
    app_module.UPLOAD_FOLDER = original_upload


def _register(client, username="testuser", email="test@example.com",
              password="password123",
              security_question="What is your pet's name?",
              security_answer="fluffy"):
    """Helper: register a user via POST (includes security question)."""
    return client.post("/register", data={
        "username": username,
        "email": email,
        "password": password,
        "security_question": security_question,
        "security_answer": security_answer,
    }, follow_redirects=False)


def _login(client, email="test@example.com", password="password123"):
    """Helper: login via POST."""
    return client.post("/login", data={
        "email": email,
        "password": password,
    }, follow_redirects=False)


def _make_png_bytes():
    """Return minimal valid PNG bytes (1×1 white pixel)."""
    import zlib, struct

    def chunk(name: bytes, data: bytes) -> bytes:
        c = name + data
        return struct.pack(">I", len(data)) + c + struct.pack(">I", zlib.crc32(c) & 0xFFFFFFFF)

    png = b"\x89PNG\r\n\x1a\n"
    png += chunk(b"IHDR", struct.pack(">IIBBBBB", 1, 1, 8, 2, 0, 0, 0))
    raw = b"\x00\xff\xff\xff"
    compressed = zlib.compress(raw)
    png += chunk(b"IDAT", compressed)
    png += chunk(b"IEND", b"")
    return png


# ══════════════════════════════════════════════════════════════════════════════
#  1. PAGE LOADS
# ══════════════════════════════════════════════════════════════════════════════

def test_login_page_loads(client):
    """GET /login should return HTTP 200."""
    resp = client.get("/login")
    assert resp.status_code == 200
    assert b"NeuroScan" in resp.data or b"Sign" in resp.data


def test_register_page_loads(client):
    """GET /register should return HTTP 200."""
    resp = client.get("/register")
    assert resp.status_code == 200



def test_forgot_password_page_loads(client):
    """GET /forgot-password should return HTTP 200."""
    resp = client.get("/forgot-password")
    assert resp.status_code == 200


def test_security_reset_page_loads(client):
    """GET /security-reset should return HTTP 200."""
    resp = client.get("/security-reset")
    assert resp.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
#  2. AUTHENTICATION FLOW
# ══════════════════════════════════════════════════════════════════════════════

def test_home_redirect_unauthenticated(client):
    """Unauthenticated GET / should redirect to /login."""
    resp = client.get("/")
    assert resp.status_code == 302
    assert "/login" in resp.headers.get("Location", "")


def test_register_success(client):
    """Registering with valid data should redirect to /login."""
    resp = _register(client)
    assert resp.status_code == 302
    assert "/login" in resp.headers.get("Location", "")


def test_register_duplicate_email(client):
    """Registering with an already-used email should show an error."""
    _register(client)  # first registration
    resp = _register(client)  # duplicate
    assert resp.status_code == 200
    assert b"already registered" in resp.data or b"error" in resp.data.lower()


def test_register_short_username(client):
    """Username shorter than 3 chars should fail."""
    resp = _register(client, username="ab")
    assert resp.status_code == 200
    assert b"3 char" in resp.data or b"username" in resp.data.lower()


def test_register_short_password(client):
    """Password shorter than 6 chars should fail."""
    resp = _register(client, password="abc")
    assert resp.status_code == 200
    assert b"6 char" in resp.data or b"password" in resp.data.lower()


def test_login_valid_credentials(client):
    """After registering, valid login should redirect to home."""
    _register(client)
    resp = _login(client)
    assert resp.status_code == 302
    assert "/" in resp.headers.get("Location", "")


def test_login_invalid_password(client):
    """Wrong password should stay on login page with error."""
    _register(client)
    resp = _login(client, password="wrongpassword")
    assert resp.status_code == 200
    assert b"Invalid" in resp.data or b"error" in resp.data.lower()


def test_login_unknown_email(client):
    """Unknown email should return error on login page."""
    resp = _login(client, email="nobody@nowhere.com")
    assert resp.status_code == 200
    assert b"Invalid" in resp.data or b"error" in resp.data.lower()


def test_logout_clears_session(client):
    """After login, /logout should redirect and clear session."""
    _register(client)
    _login(client)
    resp = client.get("/logout", follow_redirects=False)
    assert resp.status_code == 302
    # Home should now redirect to login (session cleared)
    home_resp = client.get("/", follow_redirects=False)
    assert home_resp.status_code == 302


# ══════════════════════════════════════════════════════════════════════════════
#  3. PREDICT ENDPOINT
# ══════════════════════════════════════════════════════════════════════════════

def test_predict_no_auth(client):
    """POST /predict without session should return 401."""
    resp = client.post("/predict")
    assert resp.status_code == 401


def test_predict_no_file(client):
    """POST /predict (authenticated) with no file should return 400."""
    _register(client)
    _login(client)
    resp = client.post("/predict", content_type="multipart/form-data", data={})
    assert resp.status_code in (400, 503)  # 503 if model not loaded


def test_predict_invalid_extension(client):
    """POST /predict with a .exe file should return 400."""
    _register(client)
    _login(client)
    data = {"image": (io.BytesIO(b"fakecontent"), "malware.exe")}
    resp = client.post("/predict", content_type="multipart/form-data", data=data)
    assert resp.status_code in (400, 503)


# ══════════════════════════════════════════════════════════════════════════════
#  4. FORGOT PASSWORD (email link)
# ══════════════════════════════════════════════════════════════════════════════

def test_forgot_password_missing_email(client):
    """POST /forgot-password with no email body should return 400."""
    resp = client.post(
        "/forgot-password",
        json={},
        content_type="application/json",
    )
    assert resp.status_code == 400


def test_forgot_password_unknown_email(client):
    """POST /forgot-password with unknown email should still return 200 (no enumeration)."""
    resp = client.post(
        "/forgot-password",
        json={"email": "ghost@nowhere.com"},
        content_type="application/json",
    )
    assert resp.status_code == 200


# ══════════════════════════════════════════════════════════════════════════════
#  5. OTP PASSWORD RESET
# ══════════════════════════════════════════════════════════════════════════════

def test_otp_send_unknown_email(client):
    """POST /api/send-otp with unknown email returns 200 (anti-enumeration)."""
    resp = client.post(
        "/api/send-otp",
        json={"email": "nobody@nowhere.com"},
        content_type="application/json",
    )
    assert resp.status_code == 200


def test_otp_send_known_email_stores_otp(client):
    """POST /api/send-otp for a real user stores OTP in otp_store and returns otp_dev."""
    _register(client)
    # Clear any existing OTP
    app_module.otp_store.clear()

    resp = client.post(
        "/api/send-otp",
        json={"email": "test@example.com"},
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    # In dev mode (no email configured), otp_dev is returned
    assert data.get("otp_dev") is not None or "test@example.com" in app_module.otp_store
    # OTP should be in store
    assert "test@example.com" in app_module.otp_store


def test_otp_verify_wrong_code(client):
    """Submitting a wrong OTP should return 401."""
    _register(client)
    app_module.otp_store.clear()

    # Put a known OTP in the store
    from datetime import datetime, timedelta
    app_module.otp_store["test@example.com"] = {
        "otp": "123456",
        "expiry": datetime.now() + timedelta(minutes=10),
    }

    resp = client.post(
        "/api/verify-otp",
        json={"email": "test@example.com", "otp": "999999", "action": "verify"},
        content_type="application/json",
    )
    assert resp.status_code == 401


def test_otp_verify_correct_code_returns_token(client):
    """Correct OTP should return a reset token."""
    _register(client)
    app_module.otp_store.clear()

    from datetime import datetime, timedelta
    app_module.otp_store["test@example.com"] = {
        "otp": "654321",
        "expiry": datetime.now() + timedelta(minutes=10),
    }

    resp = client.post(
        "/api/verify-otp",
        json={"email": "test@example.com", "otp": "654321", "action": "verify"},
        content_type="application/json",
    )
    assert resp.status_code == 200
    data = resp.get_json()
    assert "token" in data


def test_otp_full_reset_flow(client):
    """Full OTP flow: send → verify → reset password."""
    _register(client)
    app_module.otp_store.clear()
    app_module.password_reset_tokens.clear()

    # Inject OTP directly
    from datetime import datetime, timedelta
    app_module.otp_store["test@example.com"] = {
        "otp": "111222",
        "expiry": datetime.now() + timedelta(minutes=10),
    }

    # Verify OTP → get token
    resp = client.post(
        "/api/verify-otp",
        json={"email": "test@example.com", "otp": "111222", "action": "verify"},
        content_type="application/json",
    )
    assert resp.status_code == 200
    token = resp.get_json()["token"]

    # Use token to reset password
    resp2 = client.post(
        "/reset-password",
        json={"token": token, "newPassword": "newpass123"},
        content_type="application/json",
    )
    assert resp2.status_code == 200
    assert b"success" in resp2.data.lower()

    # Login with new password should work
    resp3 = _login(client, password="newpass123")
    assert resp3.status_code == 302


# ══════════════════════════════════════════════════════════════════════════════
#  6. UNIT TESTS — get_tumor_stage()
# ══════════════════════════════════════════════════════════════════════════════

def test_tumor_stage_early():
    """Low area + low confidence → Early Stage."""
    stage, desc = get_tumor_stage("Glioma Tumor", confidence=60, area_pct=5)
    assert "Early" in stage
    assert isinstance(desc, str) and len(desc) > 0


def test_tumor_stage_intermediate():
    """Moderate area + moderate confidence → Intermediate Stage."""
    stage, desc = get_tumor_stage("Pituitary Tumor", confidence=78, area_pct=20)
    assert "Intermediate" in stage or "Stage" in stage


def test_tumor_stage_advanced():
    """High area + high confidence → Advanced Stage."""
    stage, desc = get_tumor_stage("Meningioma Tumor", confidence=95, area_pct=50)
    assert "Advanced" in stage


# ══════════════════════════════════════════════════════════════════════════════
#  7. UNIT TESTS — get_precautions()
# ══════════════════════════════════════════════════════════════════════════════

def test_precautions_early_returns_list():
    """get_precautions() should return a non-empty list for Early Stage."""
    result = get_precautions("Pituitary Tumor", "Early Stage", confidence=65, area_pct=8)
    assert isinstance(result, list)
    assert len(result) > 0


def test_precautions_advanced_urgent():
    """Advanced stage should include urgent action in precautions."""
    result = get_precautions("Glioma Tumor", "Advanced Stage", confidence=92, area_pct=45)
    assert any("IMMEDIATE" in p or "urgent" in p.lower() or "consult" in p.lower() for p in result)


def test_precautions_max_ten():
    """get_precautions() should return at most 10 items."""
    result = get_precautions("Meningioma Tumor", "Intermediate Stage", confidence=80, area_pct=25)
    assert len(result) <= 10


# ══════════════════════════════════════════════════════════════════════════════
#  8. UNIT TESTS — get_medication()
# ══════════════════════════════════════════════════════════════════════════════

def test_medication_glioma_early():
    """get_medication() for Glioma Early Stage returns a dict with 'approach' key."""
    med = get_medication("Glioma Tumor", "Early Stage")
    assert isinstance(med, dict)
    assert "approach" in med
    assert "medications" in med
    assert len(med["medications"]) > 0


def test_medication_meningioma_advanced():
    """get_medication() for Meningioma Advanced Stage includes follow_up and note."""
    med = get_medication("Meningioma Tumor", "Advanced Stage")
    assert "approach" in med
    assert "follow_up" in med
    assert "note" in med
    assert "URGENT" in med.get("note", "")


def test_medication_pituitary_intermediate():
    """get_medication() for Pituitary Intermediate Stage has medication list."""
    med = get_medication("Pituitary Tumor", "Intermediate Stage")
    assert isinstance(med.get("medications"), list)
    assert len(med["medications"]) >= 1


def test_medication_no_tumor_returns_empty():
    """get_medication() for unknown/no-tumor type returns empty dict."""
    med = get_medication("No Tumor", "")
    assert med == {}


# ══════════════════════════════════════════════════════════════════════════════
#  9. UNIT TESTS — preprocess_image()
# ══════════════════════════════════════════════════════════════════════════════

def test_preprocess_image_valid_png():
    """A valid PNG should be preprocessed to shape (1, 224, 224, 3)."""
    import numpy as np

    png_bytes = _make_png_bytes()
    file_obj = io.BytesIO(png_bytes)
    result = preprocess_image(file_obj)
    assert result is not None
    assert result.shape == (1, 224, 224, 3)
    assert result.dtype == np.float32


def test_preprocess_image_invalid_bytes():
    """Garbage bytes should return None."""
    file_obj = io.BytesIO(b"this is not an image")
    result = preprocess_image(file_obj)
    assert result is None


# ══════════════════════════════════════════════════════════════════════════════
#  10. API ENDPOINTS — History & Profile
# ══════════════════════════════════════════════════════════════════════════════

def test_api_history_unauthenticated(client):
    """GET /api/history without session should return 401."""
    resp = client.get("/api/history")
    assert resp.status_code == 401


def test_api_history_authenticated_empty(client):
    """GET /api/history for a new user returns empty list."""
    _register(client)
    _login(client)
    resp = client.get("/api/history")
    assert resp.status_code == 200
    data = resp.get_json()
    assert isinstance(data, list)
    assert len(data) == 0


def test_api_profile_stats_unauthenticated(client):
    """GET /api/profile-stats without session should return 401."""
    resp = client.get("/api/profile-stats")
    assert resp.status_code == 401


def test_api_profile_stats_authenticated(client):
    """GET /api/profile-stats returns correct structure for logged-in user."""
    _register(client)
    _login(client)
    resp = client.get("/api/profile-stats")
    assert resp.status_code == 200
    data = resp.get_json()
    assert "username" in data
    assert "email" in data
    assert "total" in data
    assert "tumor" in data
    assert "clear" in data
    assert "most_common" in data
    assert "scans" in data
    assert data["total"] == 0


def test_api_scan_detail_unauthenticated(client):
    """GET /api/scan-detail/<id> without session should return 401."""
    resp = client.get("/api/scan-detail/1")
    assert resp.status_code == 401


def test_api_scan_detail_not_found(client):
    """GET /api/scan-detail/<id> for non-existent scan returns 404."""
    _register(client)
    _login(client)
    resp = client.get("/api/scan-detail/99999")
    assert resp.status_code == 404


def test_api_save_and_fetch_scan(client):
    """Save a scan then fetch it via /api/history and /api/scan-detail."""
    _register(client)
    _login(client)

    # Save a scan
    resp = client.post(
        "/api/save-scan",
        json={
            "patient_name": "Test Patient",
            "patient_id": "PT-001",
            "patient_age": "35",
            "patient_gender": "Female",
            "tumor_type": "Glioma Tumor",
            "confidence": 88.5,
            "stage": "Advanced Stage",
            "area_pct": 42.0,
            "location": "Frontal Lobe, Left Hemisphere",
            "image_b64": "",
        },
        content_type="application/json",
    )
    assert resp.status_code == 200
    assert resp.get_json().get("ok") is True

    # History should have 1 record
    history = client.get("/api/history").get_json()
    assert len(history) == 1
    assert history[0]["patient_name"] == "Test Patient"
    assert history[0]["patient_id"] == "PT-001"

    # Fetch detail
    scan_id = history[0]["id"]
    detail = client.get(f"/api/scan-detail/{scan_id}").get_json()
    assert detail["tumor_type"] == "Glioma Tumor"
    assert "medication" in detail
    assert detail["medication"].get("approach") is not None
    assert "precautions" in detail
    assert len(detail["precautions"]) > 0
    assert "stage_description" in detail


# ══════════════════════════════════════════════════════════════════════════════
#  11. DATABASE TEST
# ══════════════════════════════════════════════════════════════════════════════

def test_db_init_creates_users_table(tmp_path):
    """init_db() should create a 'users' table in the SQLite database."""
    db_file = tmp_path / "test_db.db"
    original = app_module.DB_PATH
    app_module.DB_PATH = str(db_file)

    try:
        init_db()
        conn = sqlite3.connect(str(db_file))
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='users';"
        ).fetchall()
        conn.close()
        assert len(tables) == 1
    finally:
        app_module.DB_PATH = original


def test_db_init_creates_scan_history_table(tmp_path):
    """init_db() should create 'scan_history' table with patient_id column."""
    db_file = tmp_path / "test_db2.db"
    original = app_module.DB_PATH
    app_module.DB_PATH = str(db_file)

    try:
        init_db()
        conn = sqlite3.connect(str(db_file))
        # Check table exists
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND name='scan_history';"
        ).fetchall()
        assert len(tables) == 1
        # Check patient_id column exists
        cols = [row[1] for row in conn.execute("PRAGMA table_info(scan_history);").fetchall()]
        assert "patient_id" in cols
        conn.close()
    finally:
        app_module.DB_PATH = original


# ══════════════════════════════════════════════════════════════════════════════
#  12. PROFILE & HISTORY PAGES
# ══════════════════════════════════════════════════════════════════════════════

def test_history_page_redirects_unauthenticated(client):
    """GET /history without session should redirect to login."""
    resp = client.get("/history", follow_redirects=False)
    assert resp.status_code == 302
    assert "/login" in resp.headers.get("Location", "")


def test_profile_page_redirects_unauthenticated(client):
    """GET /profile without session should redirect to login."""
    resp = client.get("/profile", follow_redirects=False)
    assert resp.status_code == 302
    assert "/login" in resp.headers.get("Location", "")


def test_history_page_loads_authenticated(client):
    """GET /history for logged-in user should return 200."""
    _register(client)
    _login(client)
    resp = client.get("/history")
    assert resp.status_code == 200


def test_profile_page_loads_authenticated(client):
    """GET /profile for logged-in user should return 200."""
    _register(client)
    _login(client)
    resp = client.get("/profile")
    assert resp.status_code == 200
