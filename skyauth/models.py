"""Request models for the two API endpoints."""

from typing import Optional

from pydantic import BaseModel


class InitiateRequest(BaseModel):
    transaction_amount: float
    transaction_id: str
    user_id: str
    latitude: Optional[float] = None
    longitude: Optional[float] = None


class VerifyRequest(BaseModel):
    session_id: str
    user_id: str
    transaction_id: str
    latitude: float
    longitude: float
    altitude: Optional[float] = 0.0
    compass_heading: float
    tilt_angle: float
    roll_angle: Optional[float] = 0.0
    sky_image_base64: str
    device_id: Optional[str] = "unknown"
    timestamp_client: Optional[int] = None
