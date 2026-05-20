# SkyAuth

Payment-flow authentication server that uses a live sky photo, device sensors, and a solar-position check to verify the user is physically outdoors at the claimed location. Gemini AI scores image authenticity (real photo vs. screenshot / AI-generated / indoor). OpenWeather provides cloud context.

## Requirements

- Python 3.11+ (tested on 3.13 on Windows)
- A Gemini API key — https://aistudio.google.com/apikey
- An OpenWeather API key — https://home.openweathermap.org/api_keys

Both keys have free tiers. Without them the server still boots but the AI and weather checks run in **mock mode** (clear-sky, generic confidence values).

## Setup

```powershell
git clone https://github.com/SnehalJainn/skyauth.git
cd skyauth
pip install -r requirements.txt
```

## Configure API keys

The server reads keys from environment variables — they are **not** stored in the repo. See `.env.example` for the variable names.

Set them in your current PowerShell session before running:

```powershell
$env:GEMINI_API_KEY = "your-gemini-key"
$env:OPENWEATHER_API_KEY = "your-openweather-key"
# Optional: override the model
# $env:GEMINI_MODEL = "gemini-2.0-flash"
```

## Run

```powershell
python main.py
```

The server prints something like:

```
Local:    http://localhost:8000
LAN:      http://<your-LAN-IP>:8000
API docs: http://localhost:8000/docs
```

Open `http://localhost:8000` in a browser for the frontend, or `/docs` for the interactive Swagger UI.

To test from your phone on the same Wi-Fi, use the `LAN` URL the server prints. You may need to allow Python through Windows Firewall the first time.

## Project layout

```
.
├── main.py              # uvicorn entrypoint and app wiring
├── skyauth/
│   ├── config.py        # env vars, paths, logger, in-memory state
│   ├── solar.py         # NOAA sun-position calculations
│   ├── image_analysis.py# OpenCV sun detection and fake-image scoring
│   ├── external_apis.py # Gemini + OpenWeather HTTP clients
│   ├── ml_model.py      # random-forest ensemble
│   ├── models.py        # request schemas
│   └── routes.py        # FastAPI endpoints
└── templates/
    └── index.html       # bundled frontend
```

## Models and algorithms

| Component | Model / algorithm | Role |
|---|---|---|
| Image authenticity & scene | Gemini 2.0-flash (multimodal LLM) | Classifies scene (sky / ground / indoor / screenshot), estimates sun position from visual cues, flags AI-generated content. Primary judge. |
| Sun ground truth | NOAA General Solar Position Calculation (Spencer 1971) | Analytical azimuth and elevation from latitude, longitude, and UTC time. No external service required. |
| Fake-image detection | OpenCV pipeline, eight weighted signals | Error-level analysis, sensor-noise floor, DFT high-frequency energy, sky-saturation uniformity, bytes-per-pixel, top-half gradient smoothness, Laplacian sharpness, and Gemini's AI probability. Fused with a weighted average; flagged when > 0.48. |
| Final approval | Random Forest (scikit-learn, 150 trees, max depth 7) | Combines 12 normalised features into an approval probability. Approves at probability >= 0.58. |
| Weather context | OpenWeather Current Weather API | Cloud cover and visibility relax the fake-detector's smoothness thresholds so hazy real photos are not misclassified. |

## Data and features

`POST /api/verify` consumes a JSON body containing:

- `sky_image_base64` — JPEG sky photo, base64-encoded
- `latitude`, `longitude` — GPS at capture time
- `compass_heading`, `tilt_angle` — phone orientation in degrees
- `timestamp_client` — capture time as unix seconds
- `session_id`, `transaction_id`, `user_id` — session identifiers from `/api/initiate`

The Random Forest receives twelve features, all in `[0, 1]`:

| # | Feature | Source |
|---|---|---|
| 1 | GPS Valid | Coordinates in valid range and drift < 1 km from session origin |
| 2 | Daytime | Solar elevation >= 3 deg at verification time |
| 3 | Direction Match | Gemini verdict is `genuine_sun_visible` or `genuine_sun_obscured` |
| 4 | Tilt Match | Gemini's verdict confidence / 100 |
| 5 | Sun Detected (CV) | OpenCV brightest-blob heuristic or Gemini sees the sun |
| 6 | Sun Position Match (CV) | Pixel-derived sun direction agrees with the solar ground truth |
| 7 | Not Fake (ML) | 1 minus the fake-image fusion probability |
| 8 | Gemini AI Score | Gemini's `ai_score` / 100 |
| 9 | Timestamp Fresh | Client timestamp within 120 s of the server clock |
| 10 | Is Sky Image (CV) | Sky-pixel ratio above 12% |
| 11 | Gemini Sun Match | Gemini's visual sun estimate agrees with the solar ground truth |
| 12 | Gemini No Block | Gemini did not raise a hard block (not-sky, night, or fake) |

**Training data:** the Random Forest is trained on **synthetic data**, not real labelled approvals. `build_random_forest` in `skyauth/ml_model.py` draws 600 positive samples from a uniform high-feature distribution and 600 negative samples from a low-feature one, then fits the trees. This is a deliberate prototype choice: it produces a calibrated boundary without needing a labelled dataset. Replace this function with a fit on real verifications once you have logged enough of them.

## Endpoints

| Method | Path | Purpose |
|---|---|---|
| GET | `/` | Frontend (`templates/index.html`) |
| GET | `/health` | Liveness check |
| POST | `/api/initiate` | Begin a payment session, get sun-aware challenge |
| POST | `/api/verify` | Submit photo + sensor data for verification |
| GET | `/api/transactions` | List recent transactions |


## Deployed link 
The app is live on : https://skyauth-1973.onrender.com/


## Notes on the free tiers

- **OpenWeather:** newly created keys can take up to ~2 hours to activate. Until then you'll see `401 Unauthorized` from `api.openweathermap.org`; the server logs the failure and falls back to a degraded weather record (no crash).
- **Gemini:** if you see `429 RESOURCE_EXHAUSTED` with `limit: 0`, the Google Cloud project tied to your key has no free-tier allocation. Easiest fix: delete the key, recreate it in AI Studio, and choose **"Create API key in new project"**.
