"""Random-forest ensemble that fuses twelve normalised features into an approval probability.

The training data is synthetic — drawn from two well-separated uniform clusters — so the
classifier learns a smooth boundary that approximates "all features high" vs "any feature low".
Replace `build_random_forest` with a fit on real labelled data when you have it.
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier


FEATURE_NAMES = [
    "GPS Valid",
    "Daytime",
    "Direction Match",
    "Tilt Match",
    "Sun Detected (CV)",
    "Sun Position Match (CV)",
    "Not Fake (ML)",
    "Gemini AI Score",
    "Timestamp Fresh",
    "Is Sky Image (CV)",
    "Gemini Sun Match",
    "Gemini No Block",
]

APPROVAL_THRESHOLD = 0.58


def build_random_forest() -> RandomForestClassifier:
    """Train a 150-tree forest on synthetic positive/negative clusters and return it."""
    rng = np.random.RandomState(42)
    n = 600

    # Positive class: every feature is comfortably high.
    pos = np.column_stack([
        rng.uniform(0.9, 1.0, n),
        rng.uniform(0.9, 1.0, n),
        rng.uniform(0.7, 1.0, n),
        rng.uniform(0.7, 1.0, n),
        rng.uniform(0.7, 1.0, n),
        rng.uniform(0.6, 1.0, n),
        rng.uniform(0.7, 1.0, n),
        rng.uniform(0.7, 1.0, n),
        rng.uniform(0.8, 1.0, n),
        rng.uniform(0.8, 1.0, n),
        rng.uniform(0.6, 1.0, n),
        rng.uniform(0.7, 1.0, n),
    ])
    # Negative class: features land in the lower half of the unit interval.
    neg = np.column_stack([
        rng.uniform(0.0, 0.5, n),
        rng.uniform(0.0, 0.3, n),
        rng.uniform(0.0, 0.4, n),
        rng.uniform(0.0, 0.4, n),
        rng.uniform(0.0, 0.3, n),
        rng.uniform(0.0, 0.3, n),
        rng.uniform(0.0, 0.4, n),
        rng.uniform(0.0, 0.4, n),
        rng.uniform(0.0, 0.5, n),
        rng.uniform(0.0, 0.4, n),
        rng.uniform(0.0, 0.3, n),
        rng.uniform(0.0, 0.3, n),
    ])

    X = np.vstack([pos, neg])
    y = np.array([1] * n + [0] * n)
    clf = RandomForestClassifier(n_estimators=150, random_state=42, max_depth=7)
    clf.fit(X, y)
    return clf


_MODEL = build_random_forest()


def rf_decision(features: dict) -> dict:
    """Run the random forest on the feature dict and return approval + per-feature importance."""
    row = np.array([[
        features["gps_valid"],
        features["is_daytime"],
        features["direction_match"],
        features["tilt_match"],
        features["sun_detected"],
        features["sun_pos_match_cv"],
        features["not_fake"],
        features["gemini_ai_score"],
        features["ts_freshness"],
        features["is_sky_image"],
        features["gemini_sun_match"],
        features["gemini_no_block"],
    ]])
    prob = _MODEL.predict_proba(row)[0]
    importances = _MODEL.feature_importances_.tolist()
    return {
        "approved": bool(prob[1] >= APPROVAL_THRESHOLD),
        "confidence": round(float(prob[1]) * 100, 1),
        "rejection_risk": round(float(prob[0]) * 100, 1),
        "feature_importance": dict(
            zip(FEATURE_NAMES, [round(i * 100, 1) for i in importances])
        ),
    }
