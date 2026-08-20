"""
test_app.py
Comprehensive test suite for the Water Quality Detection application.
Verifies OpenCV feature extraction, ML prediction, SQLite persistence, and FastAPI routes.
"""

import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Add project root to sys.path
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from app import app, extract_features, predict_water_ph, init_db, get_db_connection

def test_pipeline():
    print("=== 1. Testing Feature Extraction & Prediction on Sample Images ===")
    samples_dir = PROJECT_DIR / "samples"
    
    test_cases = [
        ("sample_acidic_ph4.png", 3.0, 5.5, "Acidic"),
        ("sample_safe_neutral_ph7.png", 6.5, 8.5, "Safe / Neutral"),
        ("sample_alkaline_ph10.png", 8.6, 12.0, "Alkaline")
    ]
    
    for filename, expected_min_ph, expected_max_ph, expected_class in test_cases:
        sample_path = samples_dir / filename
        assert sample_path.exists(), f"Sample image {sample_path} not found!"
        
        # Test feature extraction
        mean_h, mean_s, mean_v, used_fallback = extract_features(str(sample_path))
        print(f"[*] {filename} -> Mean HSV: (H={mean_h}, S={mean_s}, V={mean_v}), Fallback used: {used_fallback}")
        
        # Test ML prediction
        ph, classification, confidence = predict_water_ph(mean_h, mean_s, mean_v)
        print(f"    Predicted pH: {ph}, Class: {classification}, Confidence: {confidence}%")
        
        assert expected_min_ph <= ph <= expected_max_ph, f"Expected pH between {expected_min_ph} and {expected_max_ph}, got {ph}"
        assert classification == expected_class, f"Expected class {expected_class}, got {classification}"
    
    print("[OK] Feature extraction and ML predictions verified successfully!")

def test_fastapi_endpoints():
    print("\n=== 2. Testing FastAPI Endpoints with TestClient ===")
    client = TestClient(app)
    
    # Test GET /
    res = client.get("/")
    assert res.status_code == 200
    assert "AI-Based Water Quality Detection" in res.text
    print("[OK] GET / returned 200 and rendered dashboard HTML")
    
    # Test GET /stats before uploads
    res = client.get("/stats")
    assert res.status_code == 200
    stats = res.json()
    assert stats["success"] is True
    print(f"[OK] GET /stats returned 200: {stats}")
    
    # Test POST /analyze with sample image upload
    samples_dir = PROJECT_DIR / "samples"
    test_img = samples_dir / "sample_safe_neutral_ph7.png"
    
    with open(test_img, "rb") as f:
        res = client.post(
            "/analyze",
            files={"file": ("test_strip.png", f, "image/png")}
        )
    
    assert res.status_code == 200
    analyze_data = res.json()
    assert analyze_data["success"] is True
    assert "predicted_ph" in analyze_data
    assert "classification" in analyze_data
    assert "mean_h" in analyze_data
    print(f"[OK] POST /analyze returned 200: pH={analyze_data['predicted_ph']}, Class={analyze_data['classification']}")
    
    # Test GET /history
    res = client.get("/history")
    assert res.status_code == 200
    hist = res.json()
    assert hist["success"] is True
    assert len(hist["records"]) > 0
    print(f"[OK] GET /history returned {len(hist['records'])} records")
    
    # Test GET /samples-list
    res = client.get("/samples-list")
    assert res.status_code == 200
    assert res.json()["success"] is True
    print(f"[OK] GET /samples-list returned 200: {len(res.json()['samples'])} sample images available")
    
    print("\n[ALL TESTS PASSED SUCCESSFULLY!]")

if __name__ == "__main__":
    init_db()
    test_pipeline()
    test_fastapi_endpoints()
