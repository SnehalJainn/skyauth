"""Solar position math (NOAA algorithm) and compass-direction helpers."""

import math
from datetime import datetime


def solar_position(lat: float, lon: float, utc_dt: datetime) -> dict:
    """Sun azimuth and elevation in degrees for the given latitude, longitude, and UTC time.

    Implements the General Solar Position Calculation published by NOAA's
    Earth System Research Laboratory:
        https://gml.noaa.gov/grad/solcalc/solareqns.PDF

    The Fourier-series coefficients below are taken verbatim from that document
    (Spencer 1971) and are accurate to roughly +/-0.01 deg for civil use. Inputs
    must be a timezone-aware UTC datetime; passing local time will silently
    skew the result by the timezone offset.
    """
    # Day-of-year and decimal UTC hour are the only time inputs the algorithm needs.
    day_of_year = utc_dt.timetuple().tm_yday
    hour_utc = utc_dt.hour + utc_dt.minute / 60.0 + utc_dt.second / 3600.0

    # Fractional year (radians). Subtracting 1 from day_of_year and shifting by
    # the half-day term centres the series on solar noon UTC.
    gamma = 2 * math.pi / 365 * (day_of_year - 1 + (hour_utc - 12) / 24)

    # Equation of time (minutes): the correction between apparent solar time
    # and mean solar time. Varies by up to ~16 minutes across the year.
    eqtime = 229.18 * (
        0.000075
        + 0.001868 * math.cos(gamma)
        - 0.032077 * math.sin(gamma)
        - 0.014615 * math.cos(2 * gamma)
        - 0.04089 * math.sin(2 * gamma)
    )

    # Solar declination (radians): the latitude on Earth where the sun is
    # directly overhead at solar noon. Swings between +/-23.45 deg over the year.
    decl = (
        0.006918
        - 0.399912 * math.cos(gamma)
        + 0.070257 * math.sin(gamma)
        - 0.006758 * math.cos(2 * gamma)
        + 0.000907 * math.sin(2 * gamma)
        - 0.002697 * math.cos(3 * gamma)
        + 0.00148 * math.sin(3 * gamma)
    )

    # True solar time (minutes since solar midnight), then the hour angle:
    # 0 deg at solar noon, +15 deg per hour after noon, -15 deg per hour before.
    time_offset = eqtime + 4 * lon
    true_solar_time = hour_utc * 60 + time_offset
    hour_angle = (true_solar_time / 4) - 180

    lat_r = math.radians(lat)
    ha_r = math.radians(hour_angle)

    # Elevation angle above the horizon (spherical-law-of-cosines on the
    # celestial triangle: pole, zenith, sun).
    sin_elev = (
        math.sin(lat_r) * math.sin(decl)
        + math.cos(lat_r) * math.cos(decl) * math.cos(ha_r)
    )
    elevation = math.degrees(math.asin(max(-1.0, min(1.0, sin_elev))))

    # Azimuth measured clockwise from due North. The acos branch always returns
    # 0..180 deg, so we flip to 180..360 when the sun is west of the meridian
    # (positive hour angle = afternoon). The max(0.001, ...) and +1e-9 guards
    # avoid a divide-by-zero when the sun is exactly at the zenith.
    cos_az = (
        (math.sin(decl) - math.sin(lat_r) * sin_elev)
        / (math.cos(lat_r) * math.cos(math.radians(max(0.001, abs(elevation)))) + 1e-9)
    )
    cos_az = max(-1.0, min(1.0, cos_az))
    azimuth = math.degrees(math.acos(cos_az))
    if hour_angle > 0:
        azimuth = 360 - azimuth

    return {
        "azimuth_deg": round(azimuth, 2),
        "elevation_deg": round(elevation, 2),
        "is_above_horizon": elevation > 0,
        "hour_angle": round(hour_angle, 2),
        "declination_deg": round(math.degrees(decl), 2),
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
