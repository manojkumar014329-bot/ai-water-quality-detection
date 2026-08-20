# AI-Based Water Quality Detection Using pH Strip and Computer Vision

A complete, local web application designed for a college mini-project demonstration. The application leverages **Computer Vision (OpenCV)** and **Machine Learning (scikit-learn RandomForestRegressor)** to estimate water pH levels from photos of standard chemical pH test strips, classify water quality (Acidic, Safe / Neutral, Alkaline), and track results with a local **SQLite** database and interactive **Chart.js** analytics dashboard.

---

## 🌟 Features

- **Automated pH Estimation**: Extracts Region of Interest (ROI) and analyzes HSV colorimetry (`Mean_H`, `Mean_S`, `Mean_V`) from uploaded pH strip images.
- **Machine Learning Inference**: Employs a `RandomForestRegressor` trained on universal pH color calibration to predict numerical pH values (0.0 to 14.0).
- **Water Quality Categorization**:
  - **Acidic**: $\text{pH} < 6.5$ (Warning badge with plumbing/taste guidance)
  - **Safe / Neutral**: $6.5 \le \text{pH} \le 8.5$ (EPA/WHO standard potable drinking water range)
  - **Alkaline**: $\text{pH} > 8.5$ (Alkaline caution badge)
- **Robust Image Processing & Fallback**: Uses Gaussian blur, Canny edge detection, and contour filtering with an automatic central-region fallback to ensure demonstrations never fail or crash.
- **Model Confidence Metric**: Computes an estimator consistency score labeled as *Model Confidence*.
- **Local SQLite Database**: Stores all analysis records (`predicted_ph`, `classification`, `mean_h`, `mean_s`, `mean_v`, `confidence`, `created_at`, and image reference).
- **Interactive Web Dashboard**:
  - Drag-and-drop & click image uploader with live preview.
  - Quick-test demo sample buttons (Acidic, Safe/Neutral, Alkaline).
  - Summary metric cards (Total tests, Average pH, Latest pH, Water Status).
  - Historical trend line chart with safe reference zone ($6.5 - 8.5$).
  - Distribution doughnut chart (Acidic vs Safe vs Alkaline).
  - Searchable / refreshable history table with image thumbnails and record deletion.
- **100% Offline & Local**: No Firebase, no cloud database, no external API keys required.

---

## 🛠️ Technology Stack

### Backend
- **Python 3.10+**
- **FastAPI**: Modern, asynchronous web framework for API routing and file handling.
- **Uvicorn**: High-performance ASGI web server.
- **OpenCV (`opencv-python`)**: Computer vision pipeline for edge detection, contours, ROI extraction, and BGR-to-HSV conversion.
- **NumPy**: Numerical feature array computations.
- **scikit-learn & Joblib**: Machine learning regression model (`RandomForestRegressor`).
- **SQLite3**: Lightweight, zero-configuration local relational database.
- **Jinja2 & python-multipart**: HTML template rendering and form data parsing.

### Frontend
- **HTML5 & CSS3**: Responsive UI layout with Deep Blue (`#0B192C`), Cyan (`#00ADB5`), and Slate styling.
- **JavaScript (ES6+)**: Asynchronous API fetch handling, drag-and-drop file processing, dynamic UI updating, and toast notifications.
- **Chart.js**: Real-time historical timeline trend and classification distribution charts.
- **Font Awesome 6**: UI iconography.

---

## 📁 Project Structure

```
water_quality_project/
│
├── app.py                     # FastAPI server, routing, OpenCV pipeline, SQLite CRUD
├── water_quality_model.pkl    # Pre-trained RandomForestRegressor model (3 HSV features)
├── generate_model.py          # Script to generate/re-train model & sample test images
├── requirements.txt           # Python dependency specification
├── database.db                # SQLite database (auto-generated on startup)
├── uploads/                   # Uploaded pH strip images directory
├── samples/                   # Pre-generated sample test strips for quick demonstration
│   ├── sample_acidic_ph4.png
│   ├── sample_safe_neutral_ph7.png
│   └── sample_alkaline_ph10.png
│
├── templates/
│   └── index.html             # Dashboard frontend single-page interface
│
├── static/
│   ├── style.css              # Custom styling & responsive layouts
│   └── script.js              # Client-side controller & Chart.js logic
│
└── README.md                  # Project documentation
```

---

## 🚀 Installation & Setup

### 1. Navigate to the project directory
Open your Windows PowerShell or Command Prompt:

```powershell
cd C:\Users\manoj\.gemini\antigravity\scratch\water_quality_project
```

### 2. Install Required Dependencies
Ensure Python is installed, then run:

```powershell
pip install -r requirements.txt
```

### 3. Verify ML Model & Sample Data
The pre-trained model `water_quality_model.pkl` and sample images are already generated. If you ever need to re-generate them:

```powershell
python generate_model.py
```

---

## 🖥️ Running the Application

Start the local FastAPI server using Uvicorn:

```powershell
uvicorn app:app --reload
```

Or run directly via python:

```powershell
python app.py
```

Open your web browser and navigate to:
**[http://127.0.0.1:8000](http://127.0.0.1:8000)**

---

## 🔬 How the Computer Vision & ML Pipeline Works

```
User uploads pH strip image
           │
           ▼
FastAPI receives & validates image (JPG/PNG <= 10MB)
           │
           ▼
OpenCV reads image safely (cv2.imdecode / cv2.imread)
           │
           ▼
Image Resizing (standard scale, max 600px)
           │
           ▼
Gaussian Blur (5x5 kernel smoothing)
           │
           ▼
Grayscale Conversion & Canny Edge Detection (50, 150)
           │
           ▼
Contour Detection & Rectangular Strip ROI filtering
           │
           ├─► (If reliable ROI found) ──► Crop strip pad
           │
           └─► (If contour fails) ──────► Fallback to central region (Prevents Crashes!)
           │
           ▼
Color Conversion: BGR -> HSV (Hue: 0–179, Saturation: 0–255, Value: 0–255)
           │
           ▼
Calculate Feature Vector: [Mean_H, Mean_S, Mean_V]
           │
           ▼
Load pre-trained water_quality_model.pkl (RandomForestRegressor)
           │
           ▼
Predict pH: model.predict([[Mean_H, Mean_S, Mean_V]])[0]
           │
           ▼
Classify Water Quality:
  - pH < 6.5  ==> Acidic
  - 6.5 <= pH <= 8.5 ==> Safe / Neutral
  - pH > 8.5  ==> Alkaline
           │
           ▼
Save record in local SQLite database (analyses table)
           │
           ▼
Return JSON payload & update Dashboard, Charts, and History Table
```

---

## 🗄️ SQLite Database Schema

The application uses an SQLite database `database.db` containing the `analyses` table:

| Column | Data Type | Description |
|---|---|---|
| `id` | `INTEGER PRIMARY KEY AUTOINCREMENT` | Unique test ID |
| `filename` | `TEXT` | Stored image filename in `uploads/` |
| `predicted_ph` | `REAL` | Predicted numerical pH value (0.00 – 14.00) |
| `classification` | `TEXT` | "Acidic", "Safe / Neutral", or "Alkaline" |
| `mean_h` | `REAL` | OpenCV Mean Hue feature (0 – 179) |
| `mean_s` | `REAL` | OpenCV Mean Saturation feature (0 – 255) |
| `mean_v` | `REAL` | OpenCV Mean Value/Brightness feature (0 – 255) |
| `confidence` | `REAL` | Model Confidence percentage |
| `created_at` | `TIMESTAMP` | Timestamp of test (Default `CURRENT_TIMESTAMP`) |

---

## 📡 API Endpoints Reference

- `GET /` : Renders the web application dashboard.
- `POST /analyze` : Uploads an image file (`multipart/form-data`), runs OpenCV feature extraction and ML prediction, stores the record in SQLite, and returns result JSON.
- `GET /history` : Returns a list of past analysis records.
- `GET /stats` : Returns aggregate statistics (total analyses, average pH, latest pH, category counts).
- `DELETE /history/{id}` : Deletes an individual test record from SQLite and disk.
- `DELETE /history/clear` : Clears the history database table for demonstration reset.
- `GET /samples-list` : Lists demo sample test images available on the server.

---

## ⚠️ Academic Mini-Project Limitations & Disclaimer

1. **Local Demonstration Scope**: This project is built as a college mini-project demonstration. It operates purely offline on a local SQLite database and does not use Firebase or cloud infrastructure.
2. **Model Training Data**: The machine learning model is trained on calibrated universal pH colorimetric indicator values across the standard HSV color space.
3. **Lighting & Environment**: Variations in camera exposure, ambient lighting, shadows, and camera white-balance can impact the extracted HSV values. For best results, capture strips under balanced, neutral lighting against a plain background.
4. **Not a Laboratory Medical/Certified Device**: Predictions should be used for educational demonstration and preliminary estimation, not for laboratory certification or official municipal water safety sign-off.
