"""Image-side analysis: sky/ground classification, sun blob detection, and a
multi-signal fake/AI-render detector tuned for mobile-camera input.

All thresholds here were tuned against real WhatsApp-pipeline photos taken in
Delhi (smog, low contrast, heavy JPEG recompression). If you tighten them for
a cleaner image source, expect the recall on AI-generated images to go up but
the false-positive rate on hazy real photos to go up too.
"""

import base64
import io

import cv2
import numpy as np
import PIL.Image as PILImage

from skyauth.config import log


# Phone-camera field of view used to convert a pixel offset into degrees.
HFOV_DEG = 60.0
VFOV_DEG = 45.0

# Minimum fraction of "sky-like" pixels for the image to count as a sky photo.
MIN_SKY_PIXEL_RATIO = 0.12

# Fake-image fusion: weights for the eight component scores.
FAKE_SCORE_WEIGHTS = [2, 2, 1, 1, 1, 2, 1, 3]
FAKE_PROBABILITY_THRESHOLD = 0.48


def detect_sun_in_image(img_b64: str) -> dict:
    """Locate the brightest blob in the photo and decide whether it's plausibly the sun.

    Also returns the sky-pixel ratio (used elsewhere to reject ground/indoor shots) and
    the pixel-to-degree azimuth/elevation offset the blob implies.
    """
    try:
        img_bytes = base64.b64decode(img_b64 + "==")
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            return {"error": "Could not decode image", "sun_detected": False, "is_sky_image": False}

        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Sky-pixel ratio: blue daytime sky plus light-grey overcast sky.
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        sky_mask = (
            ((hsv[:, :, 0] >= 90) & (hsv[:, :, 0] <= 140) &
             (hsv[:, :, 1] >= 40) & (hsv[:, :, 2] >= 60)) |
            ((hsv[:, :, 1] < 40) & (hsv[:, :, 2] > 120))
        )
        sky_ratio = float(np.sum(sky_mask)) / (h * w)
        is_sky_image = sky_ratio > MIN_SKY_PIXEL_RATIO

        # Sun blob: pick the brightest spot after a heavy blur to suppress noise.
        blurred = cv2.GaussianBlur(gray, (31, 31), 0)
        _, max_val, _, max_loc = cv2.minMaxLoc(blurred)
        sun_x, sun_y = max_loc
        mean_brightness = float(np.mean(gray))
        max_brightness = float(max_val)
        brightness_ratio = max_brightness / max(mean_brightness, 1)
        overexposed = float(np.sum(gray > 230)) / (h * w)

        norm_x = (sun_x - w / 2) / (w / 2)
        norm_y = (sun_y - h / 2) / (h / 2)
        pixel_az_offset = norm_x * (HFOV_DEG / 2)
        pixel_el_offset = -norm_y * (VFOV_DEG / 2)

        # Three detection modes cover clear, hazy, and pure-glare conditions.
        mode_clear = (
            brightness_ratio > 3.0
            and overexposed > 0.015
            and sun_y < h * 0.65
        )
        top_region = gray[:int(h * 0.70), :]
        top_bright_frac = float(np.sum(top_region > 180)) / max(top_region.size, 1)
        mode_hazy = (
            brightness_ratio > 1.4
            and top_bright_frac > 0.04
            and max_brightness > 200
            and sun_y < h * 0.70
        )
        mode_glare = (
            overexposed > 0.08
            and sun_y < h * 0.55
            and mean_brightness > 120
        )
        sun_likely = mode_clear or mode_hazy or mode_glare
        sun_mode = (
            "clear" if mode_clear else
            "hazy" if mode_hazy else
            "glare" if mode_glare else "none"
        )

        return {
            "sun_detected": sun_likely,
            "sun_detection_mode": sun_mode,
            "is_sky_image": is_sky_image,
            "sky_pixel_ratio": round(sky_ratio, 3),
            "sun_pixel": {"x": int(sun_x), "y": int(sun_y)},
            "image_size": {"w": w, "h": h},
            "max_brightness": round(max_brightness, 1),
            "mean_brightness": round(mean_brightness, 1),
            "brightness_ratio": round(brightness_ratio, 2),
            "overexposed_fraction": round(overexposed, 4),
            "top_bright_frac": round(top_bright_frac, 4),
            "pixel_az_offset_deg": round(pixel_az_offset, 1),
            "pixel_el_offset_deg": round(pixel_el_offset, 1),
            "norm_x": round(float(norm_x), 3),
            "norm_y": round(float(norm_y), 3),
        }
    except Exception as e:
        return {"error": str(e), "sun_detected": False, "is_sky_image": False}


def compare_sun_position(
    cv_result: dict,
    device_heading: float,
    device_tilt: float,
    solar_azimuth: float,
    solar_elevation: float,
) -> dict:
    """Cross-check the in-image sun against the solar API.

    The image gives us a pixel offset; combined with the phone's compass heading and
    tilt we can reconstruct an absolute azimuth and elevation for the bright blob.
    A close match to the solar API value is evidence the photo is genuine.
    """
    if not cv_result.get("sun_detected"):
        return {
            "match": False,
            "score": 0,
            "reason": "Sun not detected in image by OpenCV",
            "estimated_azimuth": None,
            "estimated_elevation": None,
            "solar_azimuth": round(solar_azimuth, 1),
            "solar_elevation": round(solar_elevation, 1),
            "azimuth_error_deg": None,
            "elevation_error_deg": None,
        }

    estimated_az = (device_heading + cv_result["pixel_az_offset_deg"]) % 360
    estimated_el = device_tilt + cv_result["pixel_el_offset_deg"]

    az_err = abs(estimated_az - solar_azimuth)
    if az_err > 180:
        az_err = 360 - az_err
    el_err = abs(estimated_el - solar_elevation)

    match = az_err < 30 and el_err < 25
    score = max(0.0, 100 - az_err * 1.5 - el_err * 2)

    return {
        "match": match,
        "estimated_azimuth": round(estimated_az, 1),
        "estimated_elevation": round(estimated_el, 1),
        "solar_azimuth": round(solar_azimuth, 1),
        "solar_elevation": round(solar_elevation, 1),
        "azimuth_error_deg": round(az_err, 1),
        "elevation_error_deg": round(el_err, 1),
        "score": round(score, 1),
        "reason": (
            f"Sun position matches solar API (az err {az_err:.1f} deg, el err {el_err:.1f} deg)"
            if match else
            f"Sun position mismatch (az err {az_err:.1f} deg, el err {el_err:.1f} deg)"
        ),
    }


def _weather_relaxation(weather_context: dict) -> tuple:
    """Translate weather context into a multiplier for the smoothness-based thresholds.

    Hazy or overcast scenes are genuinely low-texture, so a flat threshold would
    flag every Delhi-winter photo as AI-generated. The multiplier loosens the
    smoothness checks proportionally.
    """
    clouds_pct = weather_context.get("clouds_pct", 0)
    vis_m = weather_context.get("visibility_m", 10000)
    desc = (weather_context.get("description") or "").lower()

    is_hazy = vis_m < 5000 or any(k in desc for k in ("haze", "fog", "mist", "smoke", "dust", "sand"))
    is_overcast = clouds_pct >= 70 or any(
        k in desc for k in ("overcast", "broken", "few clouds", "scattered", "clouds")
    )

    if is_hazy and is_overcast:
        relax = 2.2
    elif is_hazy:
        relax = 1.8
    elif is_overcast:
        relax = 1.5
    else:
        relax = 1.0
    return relax, is_hazy, is_overcast


def detect_fake_image(img_b64: str, gemini_ai_prob: float = 0.5,
                      weather_context: dict = None) -> dict:
    """Score how likely the image is AI-generated, a screenshot, or heavily edited.

    Eight component checks feed into a weighted average. The weather context is used
    to relax smoothness-sensitive thresholds; a thresholds-only approach would
    misclassify legitimately hazy photos.
    """
    log.info("[fake-detect] starting image authenticity analysis")

    weather_context = weather_context or {}
    smooth_relax, is_hazy, is_overcast = _weather_relaxation(weather_context)
    log.info(
        "  weather context: clouds=%s vis=%s desc=%r hazy=%s overcast=%s relax=%.2f",
        weather_context.get("clouds_pct"),
        weather_context.get("visibility_m"),
        weather_context.get("description", ""),
        is_hazy, is_overcast, smooth_relax,
    )

    try:
        img_bytes = base64.b64decode(img_b64 + "==")
        arr = np.frombuffer(img_bytes, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is None:
            log.error("  could not decode image bytes")
            return {
                "fake_probability": 0.9, "is_likely_fake": True,
                "flags": ["decode_fail"], "ela_mean": 0, "noise_std": 0,
            }

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape[:2]
        log.info("  image %dx%d px, payload %.1f KB", w, h, len(img_bytes) / 1024)

        flags: list = []
        scores: list = []

        # 1. Error-level analysis. WhatsApp already recompresses photos, so a
        #    near-identical second pass means the source was lossless (PNG
        #    screenshot or AI render saved as JPEG).
        buf = io.BytesIO()
        pil = PILImage.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
        pil.save(buf, "JPEG", quality=92)
        buf.seek(0)
        recompressed = cv2.cvtColor(np.array(PILImage.open(buf)), cv2.COLOR_RGB2BGR)
        ela = cv2.absdiff(img, recompressed).astype(np.float32)
        ela_mean = float(np.mean(ela))
        ela_std = float(np.std(ela))

        ela_zero_threshold = 0.005
        ela_high_threshold = 20.0
        if ela_mean < ela_zero_threshold:
            flags.append("ela_near_zero_lossless_source")
            scores.append(0.70)
            log.warning("  [1] ela=%.4f near-zero, lossless source suspected", ela_mean)
        elif ela_mean > ela_high_threshold:
            flags.append("ela_high_possible_editing")
            scores.append(0.55)
            log.warning("  [1] ela=%.4f high, possible editing", ela_mean)
        else:
            scores.append(0.1)
            log.info("  [1] ela=%.4f within normal range", ela_mean)

        # 2. Sensor-noise floor. Real CMOS sensors emit ISO noise even at low ISO;
        #    AI renders are completely smooth at the pixel level.
        denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
        noise = img.astype(np.float32) - denoised.astype(np.float32)
        noise_std = float(np.std(noise))
        n_low1 = 1.5 / smooth_relax
        n_low2 = 3.0 / smooth_relax
        if noise_std < n_low1:
            flags.append("near_zero_sensor_noise_ai_render")
            scores.append(0.80)
            log.warning("  [2] noise=%.3f near-zero (threshold %.2f), likely AI render",
                        noise_std, n_low1)
        elif noise_std < n_low2:
            flags.append("low_sensor_noise_suspicious")
            scores.append(0.45)
            log.warning("  [2] noise=%.3f low (threshold %.2f)", noise_std, n_low2)
        else:
            scores.append(0.1)
            log.info("  [2] noise=%.3f", noise_std)

        # 3. DFT high-frequency energy fraction. AI images lack natural high-frequency
        #    texture; the high-band energy ratio collapses below ~0.15.
        f_transform = np.fft.fft2(gray.astype(np.float32))
        f_shifted = np.fft.fftshift(f_transform)
        rows_f, cols_f = gray.shape
        crow, ccol = rows_f // 2, cols_f // 2
        lf_radius = min(rows_f, cols_f) // 6
        mask_low = np.zeros((rows_f, cols_f), np.uint8)
        cv2.circle(mask_low, (ccol, crow), lf_radius, 1, -1)
        mask_high = 1 - mask_low
        abs_f = np.abs(f_shifted)
        low_energy = float(np.sum(abs_f * mask_low))
        high_energy = float(np.sum(abs_f * mask_high))
        high_frac = high_energy / max(low_energy + high_energy, 1)

        hf_hard = 0.10
        hf_soft = 0.15
        if high_frac < hf_hard:
            flags.append("dft_missing_high_freq_ai_generated")
            scores.append(0.70)
            log.warning("  [3] dft_high_frac=%.4f very low, AI-generated likely", high_frac)
        elif high_frac < hf_soft:
            flags.append("dft_low_high_freq_suspicious")
            scores.append(0.40)
            log.warning("  [3] dft_high_frac=%.4f low", high_frac)
        else:
            scores.append(0.1)
            log.info("  [3] dft_high_frac=%.4f", high_frac)

        # 4. Sky saturation uniformity. Stable diffusion etc. produce skies with
        #    very low saturation variance; real skies have natural texture.
        sky_region = img[:h // 2, :]
        hsv_sky = cv2.cvtColor(sky_region, cv2.COLOR_BGR2HSV)
        sat_std = float(np.std(hsv_sky[:, :, 1]))
        sat_thresh = 7.0 / smooth_relax
        if sat_std < sat_thresh:
            flags.append("unnaturally_uniform_sky_ai_render")
            scores.append(0.70)
            log.warning("  [4] sat_std=%.2f too uniform (threshold %.2f)", sat_std, sat_thresh)
        else:
            scores.append(0.1)
            log.info("  [4] sat_std=%.2f (threshold %.2f)", sat_std, sat_thresh)

        # 5. Bytes-per-pixel. Pure screenshots and renders compress far smaller
        #    than camera output, even after WhatsApp's pipeline.
        file_size = len(img_bytes)
        compression_ratio = file_size / (w * h)
        compress_threshold = 0.005
        if compression_ratio < compress_threshold:
            flags.append("suspiciously_compressed_possible_screenshot")
            scores.append(0.60)
            log.warning("  [5] compression=%.4f below threshold %.4f", compression_ratio,
                        compress_threshold)
        else:
            scores.append(0.1)
            log.info("  [5] compression=%.4f", compression_ratio)

        # 6. Top-half gradient smoothness. Perfect linear sky gradients indicate
        #    rendered content.
        sky_gray = gray[:h // 2, :]
        row_means = np.mean(sky_gray, axis=1)
        row_diffs = np.diff(row_means)
        rd_std = float(np.std(row_diffs))
        rd_mean = float(np.mean(np.abs(row_diffs)))
        grad_std_t = 1.2 / smooth_relax
        grad_mean_t = 1.5 / smooth_relax
        if rd_std < grad_std_t and rd_mean < grad_mean_t:
            flags.append("perfect_gradient_sky_ai_or_render")
            scores.append(0.85)
            log.warning("  [6] gradient std=%.3f mean=%.3f (thresholds %.3f/%.3f)",
                        rd_std, rd_mean, grad_std_t, grad_mean_t)
        else:
            scores.append(0.1)
            log.info("  [6] gradient std=%.3f mean=%.3f", rd_std, rd_mean)

        # 7. Laplacian variance as a sharpness proxy. Real haze and JPEG
        #    compression both flatten this number, so the threshold has to be lenient.
        lap_var = float(cv2.Laplacian(gray, cv2.CV_64F).var())
        lap_thresh = 40.0 / smooth_relax
        if lap_var < lap_thresh:
            flags.append("too_smooth_possibly_ai")
            scores.append(0.60)
            log.warning("  [7] laplacian=%.2f below threshold %.2f", lap_var, lap_thresh)
        elif lap_var > 10000:
            flags.append("over_sharpened_possible_post_processing")
            scores.append(0.45)
            log.warning("  [7] laplacian=%.2f very high, over-sharpened", lap_var)
        else:
            scores.append(0.1)
            log.info("  [7] laplacian=%.2f", lap_var)

        # 8. Gemini's standalone AI-generation probability, normalised.
        gemini_norm = max(0.0, min(1.0, gemini_ai_prob / 100.0))
        scores.append(gemini_norm)
        if gemini_norm > 0.6:
            flags.append(f"gemini_flagged_ai_{int(gemini_ai_prob)}pct")
            log.warning("  [8] gemini ai_prob=%.0f%% flagged", gemini_ai_prob)
        else:
            log.info("  [8] gemini ai_prob=%.0f%%", gemini_ai_prob)

        weighted = [s * wt for s, wt in zip(scores, FAKE_SCORE_WEIGHTS)]
        fake_prob = round(min(1.0, float(sum(weighted) / sum(FAKE_SCORE_WEIGHTS))), 3)
        is_fake = fake_prob > FAKE_PROBABILITY_THRESHOLD

        log.info("  components=%s", [round(s, 3) for s in scores])
        log.info("  weighted_sum=%.3f / %d", sum(weighted), sum(FAKE_SCORE_WEIGHTS))
        log.info("  fake_probability=%.3f threshold=%.2f", fake_prob, FAKE_PROBABILITY_THRESHOLD)
        log.info("  flags=%s", flags or "none")
        if is_fake:
            log.warning("  verdict: likely fake (prob=%.3f)", fake_prob)
        else:
            log.info("  verdict: likely genuine (prob=%.3f)", fake_prob)

        return {
            "fake_probability": fake_prob,
            "is_likely_fake": is_fake,
            "flags": flags,
            "ela_mean": round(ela_mean, 4),
            "ela_std": round(ela_std, 4),
            "noise_std": round(noise_std, 3),
            "sky_saturation_std": round(sat_std, 1),
            "dft_high_freq_frac": round(high_frac, 4),
            "compression_ratio": round(compression_ratio, 4),
            "sharpness_laplacian": round(lap_var, 2),
            "gemini_ai_prob": round(gemini_ai_prob, 1),
            "weather_relaxation": round(smooth_relax, 2),
            "score_components": {
                "ela": round(scores[0], 3),
                "noise": round(scores[1], 3),
                "freq": round(scores[2], 3),
                "color": round(scores[3], 3),
                "compress": round(scores[4], 3),
                "gradient": round(scores[5], 3),
                "sharpness": round(scores[6], 3),
                "gemini": round(scores[7], 3),
            },
        }

    except Exception as e:
        log.error("  detect_fake_image exception: %s", e, exc_info=True)
        return {
            "fake_probability": 0.5,
            "is_likely_fake": False,
            "flags": [f"analysis_error:{str(e)}"],
            "ela_mean": 0,
            "noise_std": 0,
            "gemini_ai_prob": gemini_ai_prob,
        }
