"""HTTP endpoints for the SkyAuth payment-verification flow.

The router exposes four routes:

    GET  /                  - serve the bundled frontend (templates/index.html)
    GET  /health            - liveness probe
    POST /api/initiate      - open a verification session, compute solar ground truth
    POST /api/verify        - submit photo + sensors, get APPROVED/REJECTED

Verification fuses six signals (GPS, daytime, Gemini verdict, OpenCV sun detection,
fake-image probability, Gemini sun-position cross-check) and gates the final decision
behind both a random-forest probability and a classical weighted-score threshold.
"""

import hashlib
import json
import math
import os
import time
from datetime import datetime, timezone

from fastapi import APIRouter, HTTPException
from fastapi.responses import HTMLResponse

from skyauth.config import (
    MAX_GPS_DRIFT_KM,
    MIN_SUN_ELEVATION_DEG,
    SESSION_TTL_SECONDS,
    TEMPLATE_DIR,
    TIMESTAMP_FRESH_SECONDS,
    TIMESTAMP_STALE_SECONDS,
    log,
    sessions,
    transactions,
)
from skyauth.external_apis import (
    analyze_image_with_gemini,
    compare_gemini_sun_estimate,
    get_weather,
)
from skyauth.image_analysis import (
    compare_sun_position,
    detect_fake_image,
    detect_sun_in_image,
)
from skyauth.ml_model import rf_decision
from skyauth.models import InitiateRequest, VerifyRequest
from skyauth.solar import azimuth_to_direction, solar_position


router = APIRouter()


def _sha256_payload(data: dict) -> str:
    """Stable SHA-256 of the given dict, used as a verification receipt."""
    return hashlib.sha256(
        json.dumps(data, sort_keys=True, default=str).encode()
    ).hexdigest()


def _haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Great-circle distance in kilometres."""
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2
    )
    return round(6371 * 2 * math.asin(math.sqrt(a)), 3)


@router.get("/", response_class=HTMLResponse)
async def serve_frontend():
    """Serve the bundled single-page frontend if present, otherwise a placeholder."""
    index_path = TEMPLATE_DIR / "index.html"
    if index_path.exists():
        return HTMLResponse(index_path.read_text(encoding="utf-8"))
    # Also accept index.html at the project root for backward compatibility.
    if os.path.exists("index.html"):
        with open("index.html", "r", encoding="utf-8") as f:
            return HTMLResponse(f.read())
    return HTMLResponse(
        "<h1>SkyAuth</h1><p>Place index.html in the templates/ directory.</p>"
    )


@router.get("/health")
async def health():
    return {"status": "ok", "version": "3.0", "server_time": datetime.now().isoformat()}


@router.post("/api/initiate")
async def initiate_auth(req: InitiateRequest):
    """Open a verification session and return the current solar position.

    The session expires after `SESSION_TTL_SECONDS`. The caller's GPS plus the
    server's clock determine the ground-truth sun position that the photo will
    be checked against.
    """
    if req.transaction_amount <= 0:
        raise HTTPException(400, "Transaction amount must be positive")
    if req.latitude is None or req.longitude is None:
        raise HTTPException(400, "GPS coordinates are required to compute sun position.")
    if not (-90 <= req.latitude <= 90) or not (-180 <= req.longitude <= 180):
        raise HTTPException(400, "Invalid GPS coordinates.")

    utc_now = datetime.now(timezone.utc)
    sun = solar_position(req.latitude, req.longitude, utc_now)

    if not sun["is_above_horizon"]:
        raise HTTPException(
            403,
            f"SkyAuth requires daylight. "
            f"Sun is {abs(sun['elevation_deg']):.1f} deg below horizon at your location. "
            f"Please try again during daylight hours.",
        )
    if sun["elevation_deg"] < MIN_SUN_ELEVATION_DEG:
        raise HTTPException(
            403,
            f"Sun elevation {sun['elevation_deg']:.1f} deg is too close to the horizon. "
            f"Please try when the sun is higher in the sky.",
        )

    nonce = hashlib.sha256(os.urandom(32)).hexdigest()[:16]
    session_id = hashlib.sha256(
        f"{req.transaction_id}{req.user_id}{time.time()}".encode()
    ).hexdigest()[:24]

    now_ts = int(time.time())
    sessions[session_id] = {
        "sun_azimuth": sun["azimuth_deg"],
        "sun_elevation": sun["elevation_deg"],
        "nonce": nonce,
        "issued_at": now_ts,
        "expires_at": now_ts + SESSION_TTL_SECONDS,
        "transaction_amount": req.transaction_amount,
        "transaction_id": req.transaction_id,
        "user_id": req.user_id,
        "initiated_at": now_ts,
        "init_lat": req.latitude,
        "init_lon": req.longitude,
    }

    return {
        "session_id": session_id,
        "sun": sun,
        "instruction": (
            "Point your camera at the sky where you can see the sun and take a photo. "
            "SkyAuth's AI will verify the image. You have 120 seconds."
        ),
        "sun_info": {
            "azimuth_deg": round(sun["azimuth_deg"], 1),
            "elevation_deg": round(sun["elevation_deg"], 1),
            "direction": azimuth_to_direction(sun["azimuth_deg"]),
        },
    }


@router.post("/api/verify")
async def verify_auth(req: VerifyRequest):
    """Run all six verification signals on the submitted photo and decide APPROVED or REJECTED."""
    if req.session_id not in sessions:
        raise HTTPException(404, "Session not found or expired")
    session = sessions[req.session_id]

    if int(time.time()) > session["expires_at"]:
        del sessions[req.session_id]
        raise HTTPException(410, "Challenge expired. Please start a new transaction.")

    log.info("=" * 60)
    log.info("VERIFY user=%s txn=%s", req.user_id, req.transaction_id)
    log.info("   GPS: %.4f,%.4f  heading=%s deg  tilt=%s deg",
             req.latitude, req.longitude, req.compass_heading, req.tilt_angle)

    # 1. Weather context (drives the fake-detector's smoothness tolerance).
    weather = get_weather(req.latitude, req.longitude)
    log.info("   weather: %s clouds=%s%% vis=%sm temp=%s C",
             weather.get("description"), weather.get("clouds_pct"),
             weather.get("visibility_m"), weather.get("temperature_c"))

    # 2. Solar ground truth at the verification time (not the initiate time).
    utc_now = datetime.now(timezone.utc)
    sun = solar_position(req.latitude, req.longitude, utc_now)
    is_daytime = sun["is_above_horizon"] and sun["elevation_deg"] >= MIN_SUN_ELEVATION_DEG
    log.info("   sun az=%s el=%s daytime=%s",
             sun["azimuth_deg"], sun["elevation_deg"], is_daytime)

    # 3. OpenCV pass: sky/ground classification and brightest-blob localisation.
    log.info("--- opencv sun detection ---")
    cv_result = detect_sun_in_image(req.sky_image_base64)
    log.info("   sun_detected=%s is_sky=%s sky_ratio=%s brightness_ratio=%s overexposed=%s",
             cv_result.get("sun_detected"), cv_result.get("is_sky_image"),
             cv_result.get("sky_pixel_ratio"), cv_result.get("brightness_ratio"),
             cv_result.get("overexposed_fraction"))
    if cv_result.get("error"):
        log.warning("   opencv error: %s", cv_result["error"])

    sun_compare = compare_sun_position(
        cv_result,
        device_heading=req.compass_heading,
        device_tilt=req.tilt_angle,
        solar_azimuth=sun["azimuth_deg"],
        solar_elevation=sun["elevation_deg"],
    )
    log.info("   sun position match=%s | %s",
             sun_compare.get("match"), sun_compare.get("reason", ""))

    # 4. Gemini analysis - the primary judge.
    log.info("--- gemini ai analysis ---")
    ai_result = analyze_image_with_gemini(
        req.sky_image_base64,
        {
            "lat": req.latitude,
            "lon": req.longitude,
            "solar_azimuth": sun["azimuth_deg"],
            "solar_elevation": sun["elevation_deg"],
            "weather_desc": weather.get("description", "unknown"),
            "cloud_pct": weather.get("clouds_pct", 50),
        },
    )
    log.info("   verdict=%s confidence=%s%% sun_visible=%s ai_prob=%s%% ai_score=%s",
             ai_result.get("overall_verdict"), ai_result.get("verdict_confidence"),
             ai_result.get("sun_visible"), ai_result.get("ai_generated_probability"),
             ai_result.get("ai_score"))
    log.info("   scene=%s sky_condition=%s",
             ai_result.get("scene_type"), ai_result.get("sky_condition"))
    log.info("   sun reason: %s", ai_result.get("sun_visibility_reason", ""))
    log.info("   auth reason: %s", ai_result.get("authenticity_reason", ""))
    if ai_result.get("gemini_hard_block"):
        log.warning("   gemini hard block: %s", ai_result.get("gemini_block_reason"))

    # 5. Gemini's own visual sun-position estimate vs the solar API.
    gemini_sun_cmp = compare_gemini_sun_estimate(
        ai_result, sun["azimuth_deg"], sun["elevation_deg"]
    )
    log.info("   gemini sun position: %s", gemini_sun_cmp.get("reason", ""))

    # 6. Multi-signal fake/AI-render detection (Gemini's AI prob feeds in as one signal).
    log.info("--- fake-image ml pipeline ---")
    gemini_ai_prob = ai_result.get("ai_generated_probability", 30)
    fake_result = detect_fake_image(
        req.sky_image_base64,
        gemini_ai_prob,
        weather_context={
            "clouds_pct": weather.get("clouds_pct", 0),
            "description": weather.get("description", ""),
            "visibility_m": weather.get("visibility_m", 10000),
        },
    )
    log.info("   fake_prob=%s is_fake=%s relax=%s flags=%s",
             fake_result.get("fake_probability"), fake_result.get("is_likely_fake"),
             fake_result.get("weather_relaxation", "?"), fake_result.get("flags", []))

    # 7. GPS sanity and timestamp freshness.
    gps_valid = (-90 <= req.latitude <= 90) and (-180 <= req.longitude <= 180)
    gps_drift_km = 0.0
    gps_drift_ok = True
    init_lat = session.get("init_lat")
    init_lon = session.get("init_lon")
    if init_lat is not None and init_lon is not None:
        gps_drift_km = _haversine_km(req.latitude, req.longitude, init_lat, init_lon)
        gps_drift_ok = gps_drift_km < MAX_GPS_DRIFT_KM

    ts_diff = abs(int(time.time()) - (req.timestamp_client or int(time.time())))
    ts_fresh = ts_diff < TIMESTAMP_FRESH_SECONDS
    log.info("   gps: valid=%s drift=%.3fkm ok=%s", gps_valid, gps_drift_km, gps_drift_ok)
    log.info("   timestamp diff=%ss fresh=%s", ts_diff, ts_fresh)

    # 8. Project Gemini's verdict onto the binary feature space the RF expects.
    gemini_verdict = ai_result.get("overall_verdict", "genuine_sun_visible")
    gemini_sun_ok = gemini_verdict in ("genuine_sun_visible", "genuine_sun_obscured")
    gemini_verdict_conf = ai_result.get("verdict_confidence", 50)

    # 9. Random forest. The old compass/tilt features are repurposed to carry
    #    Gemini's verdict signal so the model shape stays stable.
    rf_features = {
        "gps_valid": float(gps_valid and gps_drift_ok),
        "is_daytime": float(is_daytime),
        "direction_match": float(gemini_sun_ok),
        "tilt_match": min(1.0, gemini_verdict_conf / 100.0),
        "sun_detected": float(
            cv_result.get("sun_detected", False) or ai_result.get("sun_visible", False)
        ),
        "sun_pos_match_cv": float(sun_compare.get("match", False)),
        "not_fake": 1.0 - fake_result.get("fake_probability", 0.5),
        "gemini_ai_score": ai_result.get("ai_score", 50) / 100.0,
        "ts_freshness": 1.0 if ts_fresh else (0.5 if ts_diff < TIMESTAMP_STALE_SECONDS else 0.0),
        "is_sky_image": float(
            cv_result.get("is_sky_image", False) or ai_result.get("sky_visible", False)
        ),
        "gemini_sun_match": float(gemini_sun_cmp.get("match", False)),
        "gemini_no_block": 0.0 if ai_result.get("gemini_hard_block") else 1.0,
    }
    rf = rf_decision(rf_features)

    # 10. Classical weighted score (the demo-friendly explainable number).
    gemini_sun_any = (
        ai_result.get("sun_visible")
        or (ai_result.get("sun_behind_cloud") and gemini_verdict in
            ("genuine_sun_visible", "genuine_sun_obscured"))
    )
    sun_detected_final = cv_result.get("sun_detected") or gemini_sun_any

    breakdown = {
        "gps": 25 if (gps_valid and gps_drift_ok) else (10 if gps_valid else 0),
        "daytime": 15 if is_daytime else 0,
        "gemini_verdict": (
            15 if gemini_sun_ok else (8 if gemini_verdict == "suspicious" else 0)
        ),
        "sun_detected": 10 if sun_detected_final else 0,
        "not_fake": 15 if not fake_result.get("is_likely_fake") else 0,
        "gemini_score": int(ai_result.get("ai_score", 50) * 0.10),
        "is_sky_image": 5 if (
            cv_result.get("is_sky_image") or ai_result.get("sky_visible")
        ) else 0,
        "timestamp": 10 if ts_fresh else (5 if ts_diff < TIMESTAMP_STALE_SECONDS else 0),
    }
    classic_score = sum(breakdown.values())

    # 11. Hard blockers - any one of these forces a rejection regardless of scores.
    denial_reasons: list = []

    if not is_daytime:
        denial_reasons.append(
            f"Sun is below horizon (elevation: {sun['elevation_deg']:.1f} deg) "
            f"- SkyAuth requires daylight"
        )
    if not gps_drift_ok:
        denial_reasons.append(
            f"Location mismatch: moved {gps_drift_km:.2f} km from session origin "
            f"(max: {MAX_GPS_DRIFT_KM} km)"
        )
    if not gps_valid:
        denial_reasons.append("Invalid GPS coordinates")
    if fake_result.get("is_likely_fake"):
        denial_reasons.append(
            f"Image flagged as fake/AI-generated "
            f"(probability: {fake_result['fake_probability'] * 100:.0f}%, "
            f"flags: {', '.join(fake_result.get('flags', [])[:3])})"
        )
    if ai_result.get("gemini_hard_block"):
        denial_reasons.append(
            f"Gemini AI: {ai_result.get('gemini_block_reason', 'Not a sky image')}"
        )

    # Reject only if both detectors agree the sun is absent. Daytime overcast
    # photos with a visible bright patch should still pass.
    no_sun_trace = (
        not sun_detected_final
        and gemini_verdict not in ("genuine_sun_visible", "genuine_sun_obscured")
        and not ai_result.get("sun_behind_cloud", False)
    )
    if no_sun_trace and is_daytime and not ai_result.get("gemini_hard_block"):
        denial_reasons.append(
            "No sun or bright region detected - point camera toward the sun "
            "(even through haze or clouds the sun should leave a visible bright patch)"
        )

    hard_fail = any([
        not is_daytime,
        not gps_drift_ok,
        not gps_valid,
        fake_result.get("is_likely_fake"),
        ai_result.get("gemini_hard_block"),
        no_sun_trace,
    ])

    rf_approved = rf["approved"]
    classic_approved = classic_score >= 55
    approved = (not hard_fail) and rf_approved and classic_approved

    if not rf_approved and not hard_fail:
        denial_reasons.append(
            f"AI ensemble confidence insufficient ({rf['confidence']:.1f}%)"
        )
    if not classic_approved and not hard_fail and not denial_reasons:
        denial_reasons.append(
            f"Verification score too low ({classic_score}/100, need 55)"
        )

    log.info("--- final verdict ---")
    log.info("   classic_score=%d/100 (need 55) -> %s",
             classic_score, "pass" if classic_approved else "fail")
    log.info("   rf_confidence=%s%% -> %s",
             rf["confidence"], "pass" if rf_approved else "fail")
    log.info("   hard_fail=%s", hard_fail)
    log.info("   breakdown=%s", breakdown)
    if approved:
        log.info("   APPROVED")
    else:
        log.warning("   REJECTED")
        for r in denial_reasons:
            log.warning("      reason: %s", r)
    log.info("=" * 60)

    # 12. Receipt: hash that pins this verification to its inputs.
    hash_payload = {
        "session_id": req.session_id,
        "transaction_id": req.transaction_id,
        "user_id": req.user_id,
        "gps": {"lat": req.latitude, "lon": req.longitude},
        "nonce": session["nonce"],
        "solar_azimuth": sun["azimuth_deg"],
        "solar_elevation": sun["elevation_deg"],
        "gemini_verdict": gemini_verdict,
        "server_ts": int(time.time()),
    }
    crypto_hash = _sha256_payload(hash_payload)

    tx_record = {
        "transaction_id": req.transaction_id,
        "user_id": req.user_id,
        "amount": session["transaction_amount"],
        "timestamp": utc_now.isoformat(),
        "result": "APPROVED" if approved else "REJECTED",
        "rf_confidence": rf["confidence"],
        "classic_score": classic_score,
        "crypto_hash": crypto_hash,
        "location": {"lat": req.latitude, "lon": req.longitude},
        "weather": weather,
        "denial_reasons": denial_reasons,
        "gemini_verdict": gemini_verdict,
    }
    transactions.append(tx_record)
    del sessions[req.session_id]

    return {
        "status": "APPROVED" if approved else "REJECTED",
        "approved": approved,
        "denial_reasons": denial_reasons,

        "rf_decision": rf,
        "classic_score": classic_score,
        "classic_threshold": 55,
        "breakdown": breakdown,

        "solar_position": sun,

        "ai_analysis": ai_result,
        "gemini_verdict": gemini_verdict,
        "gemini_verdict_confidence": gemini_verdict_conf,
        "gemini_sun_reasoning": ai_result.get("sun_visibility_reason", ""),
        "gemini_auth_reasoning": ai_result.get("authenticity_reason", ""),
        "gemini_sun_comparison": gemini_sun_cmp,

        "sun_cv_analysis": cv_result,
        "sun_comparison_cv": sun_compare,

        "fake_detection": fake_result,

        "weather": weather,
        "crypto_hash": crypto_hash,
        "transaction": tx_record,
        "gps_drift_km": gps_drift_km,

        "message": (
            f"Payment APPROVED - AI Confidence: {rf['confidence']}% | Score: {classic_score}/100"
            if approved else
            f"ACCESS DENIED - {denial_reasons[0] if denial_reasons else 'Verification failed'}"
        ),
    }


@router.get("/api/transactions")
async def get_transactions():
    return {"transactions": transactions[-20:], "count": len(transactions)}
