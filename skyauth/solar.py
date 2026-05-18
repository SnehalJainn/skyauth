"""Solar position math (NOAA algorithm) and compass-direction helpers."""

import math
from datetime import datetime


def solar_position(lat: float, lon: float, utc_dt: datetime) -> dict:
    """Sun azimuth and elevation in degrees for the given latitude, longitude, and UTC time."""
    n = utc_dt.timetuple().tm_yday
    hour_utc = utc_dt.hour + utc_dt.minute / 60.0 + utc_dt.second / 3600.0
    gamma = 2 * math.pi / 365 * (n - 1 + (hour_utc - 12) / 24)

    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.04089 * math.sin(2 * gamma)
    )
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )

    time_offset = eqtime + 4 * lon
    tst = hour_utc * 60 + time_offset
    ha = (tst / 4) - 180

    lat_r = math.radians(lat)
    decl_r = decl
    ha_r = math.radians(ha)

    sin_elev = (
        math.sin(lat_r) * math.sin(decl_r)
        + math.cos(lat_r) * math.cos(decl_r) * math.cos(ha_r)
    )
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))

    cos_az = (
        (math.sin(decl_r) - math.sin(lat_r) * sin_elev)
        / (math.cos(lat_r) * math.cos(math.radians(max(0.001, abs(elevation)))) + 1e-9)
    )
    cos_az = max(-1.0, min(1.0, cos_az))
    azimuth = math.degrees(math.acos(cos_az))
    if ha > 0:
        azimuth = 360 - azimuth

    return {
        "azimuth_deg": round(azimuth, 2),
        "elevation_deg": round(elevation, 2),
        "is_above_horizon": elevation > 0,
        "hour_angle": round(ha, 2),
        "declination_deg": round(math.degrees(decl_r), 2),
    }


_COMPASS_BOUNDARIES = [
    (22.5, "North"),
    (67.5, "NorthEast"),
    (112.5, "East"),
    (157.5, "SouthEast"),
    (202.5, "South"),
    (247.5, "SouthWest"),
    (292.5, "West"),
    (337.5, "NorthWest"),
    (360.0, "North"),
]


def azimuth_to_direction(az: float) -> str:
    """Map a compass azimuth (degrees, 0=North, clockwise) to a cardinal label."""
    az = az % 360
    for boundary, name in _COMPASS_BOUNDARIES:
        if az < boundary:
            return name
    return "North"
