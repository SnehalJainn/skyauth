"""Entrypoint: wires the FastAPI app and launches it under uvicorn."""

import socket

import uvicorn
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from skyauth.config import GEMINI_API_KEY, OPENWEATHER_API_KEY
from skyauth.routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="SkyAuth Payment Server", version="3.0.0")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()


def _detect_lan_ip() -> str:
    """Best-effort LAN IP discovery for the startup banner."""
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect(("8.8.8.8", 80))
        return s.getsockname()[0]
    except Exception:
        return "localhost"
    finally:
        s.close()


def _print_startup_banner() -> None:
    ip = _detect_lan_ip()
    gemini = "configured" if GEMINI_API_KEY else "not set (mock mode)"
    weather = "configured" if OPENWEATHER_API_KEY else "not set (mock mode)"
    print()
    print("SkyAuth Payment Server")
    print(f"  Local:    http://localhost:8000")
    print(f"  LAN:      http://{ip}:8000")
    print(f"  API docs: http://localhost:8000/docs")
    print()
    print(f"  GEMINI_API_KEY:      {gemini}")
    print(f"  OPENWEATHER_API_KEY: {weather}")
    print()


if __name__ == "__main__":
    _print_startup_banner()
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
