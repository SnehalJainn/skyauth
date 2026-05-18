"""Third-party HTTP clients: Gemini for image analysis, OpenWeather for current conditions.

Both clients silently fall back to mock data when their API key is unset; we never raise
to the caller because verification should still produce a result (in mock mode) for demos
and CI runs.
"""

import json
import time
from datetime import datetime

import requests

from skyauth.config import (
    GEMINI_API_KEY,
    GEMINI_MODEL,
    OPENWEATHER_API_KEY,
    log,
)


GEMINI_REQUEST_TIMEOUT_S = 25
WEATHER_REQUEST_TIMEOUT_S = 5


def analyze_image_with_gemini(img_b64: str, context: dict) -> dict:
    """Send the image plus solar/weather context to Gemini and parse its JSON verdict.

    Falls back to a mock 'genuine' response when no API key is set.
    """
    if not GEMINI_API_KEY:
        return _gemini_mock_response()

    try:
        url = (
            f"https://generativelanguage.googleapis.com/v1beta/models/"
            f"{GEMINI_MODEL}:generateContent?key={GEMINI_API_KEY}"
        )
        prompt = _build_gemini_prompt(context)
        payload = {
            "contents": [{
                "parts": [
                    {"inline_data": {"mime_type": "image/jpeg",
                                     "data": img_b64[:len(img_b64) - len(img_b64) % 4]}},
                    {"text": prompt},
                ]
            }],
            "generationConfig": {"temperature": 0.05, "maxOutputTokens": 900},
        }

        r = requests.post(url, json=payload, timeout=GEMINI_REQUEST_TIMEOUT_S)
        data = r.json()
        text = (
            data.get("candidates", [{}])[0]
            .get("content", {})
            .get("parts", [{}])[0]
            .get("text", "{}")
        )
        # Gemini sometimes wraps JSON in markdown fences despite the instruction not to.
        text = text.strip().lstrip("```json").lstrip("```").rstrip("```").strip()
        result = json.loads(text)
        result["source"] = "gemini_ai"
        result["gemini_hard_block"], result["gemini_block_reason"] = _hard_block_reason(result)
        return result

    except Exception as e:
        log.warning("gemini call failed: %s", e)
        return _gemini_error_response(e)


def compare_gemini_sun_estimate(gemini_result: dict, solar_azimuth: float,
                                 solar_elevation: float) -> dict:
    """Compare Gemini's estimated sun position against the solar API ground truth."""
    g_az = gemini_result.get("estimated_sun_azimuth")
    g_el = gemini_result.get("estimated_sun_elevation")

    if g_az is None or g_el is None:
        return {
            "match": False,
            "reason": "Gemini could not estimate sun position",
            "azimuth_error": None,
            "elevation_error": None,
            "score": 0,
        }

    az_err = abs(g_az - solar_azimuth)
    if az_err > 180:
        az_err = 360 - az_err
    el_err = abs(g_el - solar_elevation)

    match = az_err < 40 and el_err < 30
    score = max(0.0, 100 - az_err * 1.2 - el_err * 1.8)

    return {
        "match": match,
        "gemini_azimuth": round(g_az, 1),
        "solar_azimuth": round(solar_azimuth, 1),
        "gemini_elevation": round(g_el, 1),
        "solar_elevation": round(solar_elevation, 1),
        "azimuth_error": round(az_err, 1),
        "elevation_error": round(el_err, 1),
        "score": round(score, 1),
        "reason": (
            f"Gemini sun estimate matches solar API (az +-{az_err:.1f} deg, el +-{el_err:.1f} deg)"
            if match else
            f"Gemini sun estimate mismatch (az +-{az_err:.1f} deg, el +-{el_err:.1f} deg)"
        ),
    }


def get_weather(lat: float, lon: float) -> dict:
    """Fetch current conditions from OpenWeather. Returns a fixed mock dict if no key is set."""
    if not OPENWEATHER_API_KEY:
        hour = datetime.now().hour
        return {
            "source": "mock",
            "description": "clear sky",
            "temperature_c": 28.0,
            "humidity": 55,
            "clouds_pct": 10,
            "wind_speed_kmh": 8.0,
            "visibility_m": 10000,
            "weather_code": 800,
            "is_daytime": 6 <= hour <= 18,
            "city": "Unknown",
            "lat": lat,
            "lon": lon,
        }
    try:
        url = (
            f"https://api.openweathermap.org/data/2.5/weather"
            f"?lat={lat}&lon={lon}&appid={OPENWEATHER_API_KEY}&units=metric"
        )
        r = requests.get(url, timeout=WEATHER_REQUEST_TIMEOUT_S)
        d = r.json()
        sunrise = d["sys"].get("sunrise", 0)
        sunset = d["sys"].get("sunset", 0)
        now_ts = int(time.time())
        return {
            "source": "openweathermap",
            "description": d["weather"][0]["description"],
            "temperature_c": d["main"]["temp"],
            "humidity": d["main"]["humidity"],
            "clouds_pct": d["clouds"]["all"],
            "wind_speed_kmh": round(d["wind"]["speed"] * 3.6, 1),
            "visibility_m": d.get("visibility", 10000),
            "weather_code": d["weather"][0]["id"],
            "is_daytime": sunrise < now_ts < sunset,
            "city": d.get("name", "Unknown"),
            "lat": lat,
            "lon": lon,
        }
    except Exception as e:
        return {
            "source": "error",
            "description": "unavailable",
            "temperature_c": 0,
            "humidity": 0,
            "clouds_pct": 0,
            "wind_speed_kmh": 0,
            "visibility_m": 0,
            "weather_code": 0,
            "is_daytime": True,
            "city": "Unknown",
            "lat": lat,
            "lon": lon,
            "error": str(e),
        }


def _hard_block_reason(result: dict):
    """Decide whether Gemini's verdict should be a non-negotiable rejection."""
    scene = result.get("scene_type", "sky")
    verdict = result.get("overall_verdict", "genuine_sun_visible")

    if scene not in ("sky",):
        return True, f"Image classified as '{scene}' - SkyAuth requires a real outdoor sky photo."
    if result.get("is_night_image"):
        return True, "Night-time image detected."
    if verdict == "fake":
        return True, (
            f"Image flagged as fake/screenshot "
            f"({result.get('authenticity_reason', 'authenticity check failed')})"
        )
    if result.get("ai_generated_probability", 0) > 80:
        return True, (
            f"AI-generated image detected "
            f"(probability: {result.get('ai_generated_probability')}%)"
        )
    return False, None


def _build_gemini_prompt(context: dict) -> str:
    """Render the long verification prompt with the per-request context interpolated in."""
    lat = context.get("lat", "?")
    lon = context.get("lon", "?")
    s_az = context.get("solar_azimuth", "?")
    s_el = context.get("solar_elevation", "?")
    weather_desc = context.get("weather_desc", "unknown")
    cloud_pct = context.get("cloud_pct", "unknown")

    if isinstance(s_el, (int, float)):
        if s_el > 60:
            elev_hint = "very high overhead"
            position_hint = "the sun should appear very high, near the top of the frame if the camera is tilted up"
        elif s_el > 30:
            elev_hint = "moderately high"
            position_hint = "the sun should appear in the middle-upper area of the sky"
        else:
            elev_hint = "relatively low in the sky"
            position_hint = "the sun should be relatively low in the sky"
    else:
        elev_hint = "at an unknown elevation"
        position_hint = "use visual cues to estimate where the sun should be"

    return f"""You are the primary verification AI for SkyAuth - a payment system that requires
the user to photograph the real sky with the sun visible. You are the MAIN judge. There is no
compass or tilt challenge. Your analysis is the core of the decision.

CONTEXT:
- User GPS: lat={lat}, lon={lon}
- Solar API ground truth: sun azimuth={s_az} deg (0=North, clockwise), elevation={s_el} deg above horizon
- Current weather: {weather_desc}, cloud cover: {cloud_pct}%
- Note: at elevation={s_el} deg, the sun is {elev_hint}

YOUR TASK: Critically analyze this image and respond ONLY with valid JSON (no markdown, no backticks, no preamble).
{{
  "scene_type": "sky | ground | indoor | vehicle | screenshot | not_sky | other",
  "sky_visible": true/false,
  "sun_visible": true/false,
  "sun_behind_cloud": true/false,
  "sun_position_consistent": true/false,
  "clouds_present": true/false,
  "cloud_coverage_percent": 0-100,
  "sky_condition": "Clear | Partly Cloudy | Overcast | Hazy | Indoor | Not Sky | Night",

  "sun_visibility_reason": "one sentence: explain what you see regarding the sun - visible glowing disc, bright glare region, hidden behind clouds, not present, etc.",

  "estimated_sun_azimuth": null or number,
  "estimated_sun_elevation": null or number,
  "sun_position_reasoning": "explain how you estimated the sun position from visual cues: shadows, light direction, brightness gradients, lens flare angle, position of bright region",

  "is_fake_or_screenshot": true/false,
  "authenticity_confidence": 0-100,
  "ai_generated_probability": 0-100,
  "authenticity_reason": "one sentence: explain your authenticity judgment - natural noise, compression artifacts, lighting consistency, sky texture, etc.",

  "overall_verdict": "genuine_sun_visible | genuine_sun_obscured | suspicious | fake | not_sky",
  "verdict_confidence": 0-100,
  "ai_score": 0-100,

  "notes": "max 150 chars: any important observation for this specific image"
}}

DETAILED RULES:

SUN DETECTION:
- If the sun is directly visible: sun_visible=true, describe the glowing disc or intense bright spot
- If sky is bright/hazy but no clear disc: sun_visible can still be true if there's an obvious bright glare region
- If overcast/cloudy but sun is partially visible or creating a bright patch: sun_visible=true, sun_behind_cloud=true
- If fully overcast with NO bright region at all: sun_visible=false (but image may still be genuine sky)
- Sun position clues: direction of shadows (opposite sun), lens flare streaks, brightest sky region, color temperature gradient

SUN POSITION CROSS-CHECK:
- Solar API says sun is at azimuth {s_az} deg and elevation {s_el} deg
- In the image, does the bright region / sun disc appear in a position consistent with this?
- For elevation {s_el} deg: {position_hint}
- estimated_sun_azimuth: your best compass direction estimate (0=North, clockwise) based on visual cues - use null only if truly impossible
- estimated_sun_elevation: your best angle-above-horizon estimate based on where the bright region sits

AUTHENTICITY CHECK:
- Real camera photos: natural noise, slightly uneven sky texture, real atmospheric haze, natural vignetting
- AI-generated images: too-perfect sky gradients, unnaturally smooth clouds, mathematically perfect blue
- Screenshots: visible UI elements, screen pixels, status bars, app chrome, unnatural sharpness
- Edited photos: mismatched lighting, copy-paste artifacts, inconsistent shadow directions
- If cloud cover is high ({cloud_pct}%), a hazy or uniform sky is EXPECTED and NORMAL - do NOT penalize for this

VERDICT GUIDE:
- genuine_sun_visible: real photo, sky visible, sun clearly present as disc or bright glare region
- genuine_sun_obscured: real photo, sky visible, sun completely hidden BUT a distinct bright patch is still visible through clouds indicating solar position - NOT just a uniformly grey sky
- suspicious: real photo but sun position inconsistent with solar data, OR sky is uniformly grey/overcast with NO visible bright region at all
- fake: AI-generated, screenshot, heavily edited, or not a real camera photo
- not_sky: not a sky image at all (ground, indoor, etc.)

IMPORTANT: High cloud cover and haze are valid, but the user MUST be photographing toward the sun's direction - there should be a visibly brighter region in the sky (even through thick haze) indicating the sun's position. A completely uniform grey sky with NO brighter region should be rated 'suspicious'. The sun must leave some trace of brightness - diffused glow, bright patch, or glare - for genuine_sun_visible or genuine_sun_obscured.
"""


def _gemini_mock_response() -> dict:
    return {
        "source": "mock_no_key",
        "scene_type": "sky",
        "sky_visible": True,
        "sun_visible": True,
        "sun_position_consistent": True,
        "clouds_present": False,
        "cloud_coverage_percent": 10,
        "sky_condition": "Clear sky",
        "is_fake_or_screenshot": False,
        "authenticity_confidence": 70,
        "is_night_image": False,
        "ai_generated_probability": 20,
        "estimated_sun_azimuth": None,
        "estimated_sun_elevation": None,
        "sun_visibility_reason": "Mock mode - no Gemini key set",
        "authenticity_reason": "Mock mode",
        "overall_verdict": "likely_genuine",
        "ai_score": 70,
        "notes": "Gemini key not set - set GEMINI_API_KEY in your environment",
        "gemini_hard_block": False,
        "gemini_block_reason": None,
    }


def _gemini_error_response(err: Exception) -> dict:
    return {
        "source": "gemini_error",
        "error": str(err),
        "scene_type": "sky",
        "sky_visible": True,
        "sun_visible": None,
        "sun_behind_cloud": False,
        "is_fake_or_screenshot": False,
        "authenticity_confidence": 50,
        "is_night_image": False,
        "ai_generated_probability": 30,
        "estimated_sun_azimuth": None,
        "estimated_sun_elevation": None,
        "sun_visibility_reason": f"Gemini call failed: {err}",
        "authenticity_reason": "Analysis unavailable",
        "overall_verdict": "genuine_sun_visible",
        "verdict_confidence": 40,
        "ai_score": 50,
        "notes": f"Gemini call failed: {err}",
        "gemini_hard_block": False,
        "gemini_block_reason": None,
    }
