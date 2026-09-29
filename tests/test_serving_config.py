import pytest

from serving.config import cors_origins, env_flag, env_int, env_opt_float, predict_rate_limit


@pytest.mark.parametrize("raw,expected", [("1", True), ("true", True), ("ON", True),
                                          ("", False), ("0", False), ("no", False)])
def test_env_flag(raw, expected):
    assert env_flag("X", {"X": raw}) is expected


def test_env_flag_unset_is_false():
    assert env_flag("X", {}) is False


def test_env_flag_rejects_garbage():
    with pytest.raises(ValueError):
        env_flag("X", {"X": "maybe"})


def test_env_int():
    assert env_int("P", 5000, {}) == 5000
    assert env_int("P", 5000, {"P": "7860"}) == 7860
    with pytest.raises(ValueError):
        env_int("P", 5000, {"P": "abc"})


def test_env_opt_float():
    assert env_opt_float("M", {}) is None
    assert env_opt_float("M", {"M": "  "}) is None
    assert env_opt_float("M", {"M": "0.25"}) == 0.25
    with pytest.raises(ValueError):
        env_opt_float("M", {"M": "abc"})


def test_cors_origins():
    assert cors_origins({}) == "*"
    assert cors_origins({"CORS_ORIGINS": "*"}) == "*"
    assert cors_origins({"CORS_ORIGINS": "https://a.example, https://b.example"}) == [
        "https://a.example", "https://b.example"]


def test_predict_rate_limit_default():
    assert predict_rate_limit({}) == "30 per minute"


def test_predict_rate_limit_override():
    assert predict_rate_limit({"PREDICT_RATE_LIMIT": "5 per second"}) == "5 per second"
