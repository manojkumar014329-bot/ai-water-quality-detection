"""
generate_model.py
Generates the pre-trained water_quality_model.pkl for the college mini-project:
'AI-Based Water Quality Detection Using pH Strip and Computer Vision'.

Trained features: Mean_H, Mean_S, Mean_V
Target: pH Value (0.0 to 14.0)
Model: RandomForestRegressor (scikit-learn)
"""

import os
import joblib
import numpy as np
import cv2
from sklearn.ensemble import RandomForestRegressor

def build_dataset():
    """
    Constructs a calibrated dataset of universal pH strip color values
    in OpenCV HSV color space (H: 0-179, S: 0-255, V: 0-255).
    """
    # Calibration anchors: (pH, Mean_H, Mean_S, Mean_V)
    anchors = [
        (1.0, 3.0, 220.0, 225.0),    # Strong Red
        (2.0, 7.0, 225.0, 230.0),    # Red
        (3.0, 12.0, 230.0, 235.0),   # Red-Orange
        (4.0, 18.0, 235.0, 240.0),   # Orange (Acidic)
        (5.0, 26.0, 220.0, 240.0),   # Yellow-Orange
        (6.0, 35.0, 200.0, 230.0),   # Lime / Yellow-Green
        (7.0, 55.0, 185.0, 215.0),   # Green (Neutral / Safe)
        (8.0, 72.0, 175.0, 200.0),   # Dark Green / Cyan-Green (Safe)
        (9.0, 88.0, 180.0, 195.0),   # Teal / Cyan
        (10.0, 102.0, 190.0, 195.0), # Blue (Alkaline)
        (11.0, 116.0, 205.0, 185.0), # Deep Blue
        (12.0, 130.0, 210.0, 170.0), # Indigo / Violet
        (13.0, 144.0, 200.0, 150.0), # Purple
        (14.0, 158.0, 190.0, 130.0)  # Dark Violet
    ]

    np.random.seed(42)
    X = []
    y = []

    # Generate interpolated and noisy samples for robust generalization
    for i in range(len(anchors) - 1):
        ph1, h1, s1, v1 = anchors[i]
        ph2, h2, s2, v2 = anchors[i + 1]

        # Generate 150 variations between each anchor step
        for _ in range(150):
            alpha = np.random.uniform(0.0, 1.0)
            ph = ph1 + alpha * (ph2 - ph1)
            h = h1 + alpha * (h2 - h1) + np.random.normal(0, 1.2)
            s = s1 + alpha * (s2 - s1) + np.random.normal(0, 4.0)
            v = v1 + alpha * (v2 - v1) + np.random.normal(0, 4.0)

            # Clamp HSV bounds
            h = np.clip(h, 0.0, 179.0)
            s = np.clip(s, 10.0, 255.0)
            v = np.clip(v, 10.0, 255.0)

            X.append([h, s, v])
            y.append(ph)

    return np.array(X), np.array(y)

def train_and_save_model(output_path="water_quality_model.pkl"):
    print("[*] Generating training data based on universal pH strip HSV calibration...")
    X, y = build_dataset()
    print(f"[*] Training RandomForestRegressor on {len(X)} data points with 3 features: Mean_H, Mean_S, Mean_V...")

    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=12,
        min_samples_split=3,
        random_state=42
    )
    model.fit(X, y)

    score = model.score(X, y)
    print(f"[*] Model R2 score on calibration dataset: {score:.4f}")

    joblib.dump(model, output_path)
    print(f"[OK] Saved model successfully to: {output_path}")
    return model

def create_sample_images(samples_dir="samples"):
    """
    Creates realistic sample pH strip images for instant testing/demoing.
    """
    os.makedirs(samples_dir, exist_ok=True)

    # 1. Acidic Strip (Orange/Red, pH ~4.0)
    # In HSV: H=18, S=235, V=240 -> convert to BGR
    hsv_acidic = np.uint8([[[18, 235, 240]]])
    bgr_acidic = cv2.cvtColor(hsv_acidic, cv2.COLOR_HSV2BGR)[0][0]

    img_acidic = np.full((300, 300, 3), (245, 245, 245), dtype=np.uint8)
    cv2.rectangle(img_acidic, (110, 30), (190, 270), (225, 225, 220), -1)
    cv2.rectangle(img_acidic, (120, 60), (180, 130), (int(bgr_acidic[0]), int(bgr_acidic[1]), int(bgr_acidic[2])), -1)
    cv2.putText(img_acidic, "Acidic pH 4.0", (80, 288), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)
    cv2.imwrite(os.path.join(samples_dir, "sample_acidic_ph4.png"), img_acidic)

    # 2. Safe / Neutral Strip (Green, pH ~7.0)
    # In HSV: H=55, S=185, V=215
    hsv_neutral = np.uint8([[[55, 185, 215]]])
    bgr_neutral = cv2.cvtColor(hsv_neutral, cv2.COLOR_HSV2BGR)[0][0]

    img_neutral = np.full((300, 300, 3), (245, 245, 245), dtype=np.uint8)
    cv2.rectangle(img_neutral, (110, 30), (190, 270), (225, 225, 220), -1)
    cv2.rectangle(img_neutral, (120, 60), (180, 130), (int(bgr_neutral[0]), int(bgr_neutral[1]), int(bgr_neutral[2])), -1)
    cv2.putText(img_neutral, "Safe pH 7.0", (85, 288), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)
    cv2.imwrite(os.path.join(samples_dir, "sample_safe_neutral_ph7.png"), img_neutral)

    # 3. Alkaline Strip (Deep Blue, pH ~10.0)
    # In HSV: H=102, S=190, V=195
    hsv_alkaline = np.uint8([[[102, 190, 195]]])
    bgr_alkaline = cv2.cvtColor(hsv_alkaline, cv2.COLOR_HSV2BGR)[0][0]

    img_alkaline = np.full((300, 300, 3), (245, 245, 245), dtype=np.uint8)
    cv2.rectangle(img_alkaline, (110, 30), (190, 270), (225, 225, 220), -1)
    cv2.rectangle(img_alkaline, (120, 60), (180, 130), (int(bgr_alkaline[0]), int(bgr_alkaline[1]), int(bgr_alkaline[2])), -1)
    cv2.putText(img_alkaline, "Alkaline pH 10.0", (70, 288), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (50, 50, 50), 1)
    cv2.imwrite(os.path.join(samples_dir, "sample_alkaline_ph10.png"), img_alkaline)

    # 4. Realistic Raw Strip Photo simulation
    raw_img = np.full((400, 400, 3), (230, 235, 238), dtype=np.uint8)
    cv2.rectangle(raw_img, (150, 40), (250, 360), (210, 215, 210), -1)
    cv2.rectangle(raw_img, (160, 80), (240, 170), (int(bgr_neutral[0]), int(bgr_neutral[1]), int(bgr_neutral[2])), -1)
    cv2.rectangle(raw_img, (150, 40), (250, 360), (170, 170, 170), 2)
    cv2.imwrite(os.path.join(samples_dir, "sample_strip_raw.jpg"), raw_img)

    print(f"[OK] Created demo sample images in: {samples_dir}")

if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(current_dir, "water_quality_model.pkl")
    samples_path = os.path.join(current_dir, "samples")

    train_and_save_model(model_file)
    create_sample_images(samples_path)
