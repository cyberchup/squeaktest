from pathlib import Path

import pytest

from squeaktest.config import DEFAULT_DATA_DIR, MB, ServerSettings, Settings


def test_defaults_match_the_brief():
    s = Settings()
    assert s.max_file_bytes == 25 * MB
    assert s.max_duration_s == 300.0
    assert s.decode_timeout_s == 30.0
    assert s.temp_dir is None
    assert s.data_dir == DEFAULT_DATA_DIR


def test_empty_environment_gives_defaults():
    assert Settings.from_env({}) == Settings()


def test_environment_overrides():
    s = Settings.from_env(
        {
            "SQUEAKTEST_MAX_FILE_MB": "10",
            "SQUEAKTEST_MAX_DURATION_S": "60",
            "SQUEAKTEST_DECODE_TIMEOUT_S": "5.5",
            "SQUEAKTEST_FFMPEG": "/opt/ffmpeg/bin/ffmpeg",
            "SQUEAKTEST_FFPROBE": "/opt/ffmpeg/bin/ffprobe",
            "SQUEAKTEST_TEMP_DIR": "/run/squeaktest",
            "SQUEAKTEST_DATA_DIR": "/srv/squeaktest-data",
        }
    )
    assert s.data_dir == Path("/srv/squeaktest-data")
    assert s.max_file_bytes == 10 * MB
    assert s.max_duration_s == 60.0
    assert s.decode_timeout_s == 5.5
    assert s.ffmpeg == "/opt/ffmpeg/bin/ffmpeg"
    assert s.ffprobe == "/opt/ffmpeg/bin/ffprobe"
    assert s.temp_dir == Path("/run/squeaktest")


def test_blank_variables_are_ignored():
    assert Settings.from_env({"SQUEAKTEST_MAX_DURATION_S": ""}) == Settings()


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("SQUEAKTEST_MAX_DURATION_S", "five"),
        ("SQUEAKTEST_MAX_FILE_MB", "0"),
        ("SQUEAKTEST_MAX_FILE_MB", "-1"),
        ("SQUEAKTEST_DECODE_TIMEOUT_S", "nan"),
        ("SQUEAKTEST_MAX_DURATION_S", "inf"),
        ("SQUEAKTEST_MAX_FILE_MB", "inf"),
    ],
)
def test_invalid_values_are_rejected(var: str, value: str):
    with pytest.raises(ValueError, match=r"SQUEAKTEST|must be a positive number"):
        Settings.from_env({var: value})


def test_direct_construction_validates():
    with pytest.raises(ValueError, match="max_duration_s"):
        Settings(max_duration_s=0)


def test_server_defaults():
    s = ServerSettings()
    assert s.rate_limit_per_minute == 6
    assert s.max_waiting == 4
    assert s.proxy_hops == 0
    assert s.frame_ancestors == "'none'"
    assert ServerSettings.from_env({}) == s


def test_server_environment_overrides():
    s = ServerSettings.from_env(
        {
            "SQUEAKTEST_RATE_LIMIT_PER_MINUTE": "3",
            "SQUEAKTEST_MAX_WAITING": "0",
            "SQUEAKTEST_UPLOAD_TIMEOUT_S": "20",
            "SQUEAKTEST_PROXY_HOPS": "1",
            "SQUEAKTEST_FRAME_ANCESTORS": "https://huggingface.co",
        }
    )
    assert s == ServerSettings(3, 0, 20.0, 1, "https://huggingface.co")


@pytest.mark.parametrize(
    ("var", "value"),
    [
        ("SQUEAKTEST_RATE_LIMIT_PER_MINUTE", "0"),
        ("SQUEAKTEST_RATE_LIMIT_PER_MINUTE", "1.5"),
        ("SQUEAKTEST_MAX_WAITING", "-1"),
        ("SQUEAKTEST_PROXY_HOPS", "-1"),
        ("SQUEAKTEST_UPLOAD_TIMEOUT_S", "0"),
        # Anything that could end the CSP directive or inject another header is refused.
        ("SQUEAKTEST_FRAME_ANCESTORS", "'self'; script-src *"),
        ("SQUEAKTEST_FRAME_ANCESTORS", "https://a.example, https://b.example"),
        ("SQUEAKTEST_FRAME_ANCESTORS", "   "),
    ],
)
def test_invalid_server_values_are_rejected(var: str, value: str):
    with pytest.raises(ValueError, match=r"SQUEAKTEST|frame_ancestors|must"):
        ServerSettings.from_env({var: value})
