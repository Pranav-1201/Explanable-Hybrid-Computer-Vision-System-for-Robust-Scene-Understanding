"""Environment configuration for serving. Invalid values fail loudly."""
import os

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"", "0", "false", "no", "off"}


def env_flag(name: str, environ=os.environ) -> bool:
    raw = environ.get(name, "").strip().lower()
    if raw in _TRUE:
        return True
    if raw in _FALSE:
        return False
    raise ValueError(f"{name}={raw!r} is not a boolean (use 1/0, true/false, yes/no, on/off)")


def env_int(name: str, default: int, environ=os.environ) -> int:
    raw = environ.get(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{name}={raw!r} is not an integer") from None


def cors_origins(environ=os.environ):
    """'*' (today's behaviour) unless CORS_ORIGINS lists comma-separated origins."""
    raw = environ.get("CORS_ORIGINS", "*").strip()
    if raw in ("", "*"):
        return "*"
    return [o.strip() for o in raw.split(",") if o.strip()]


def predict_rate_limit(environ=os.environ):
    """flask-limiter limit string for the inference endpoints (/predict,
    /predict_batch). These do real model forward passes -- unlike /health,
    /classes and /, which stay unlimited so the Docker HEALTHCHECK and the
    frontend shell can never be starved by this."""
    raw = environ.get("PREDICT_RATE_LIMIT", "").strip()
    return raw or "30 per minute"
