import pytest

from serving.config import cors_origins, env_flag, env_int


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


def test_cors_origins():
    assert cors_origins({}) == "*"
    assert cors_origins({"CORS_ORIGINS": "*"}) == "*"
    assert cors_origins({"CORS_ORIGINS": "https://a.example, https://b.example"}) == [
        "https://a.example", "https://b.example"]
