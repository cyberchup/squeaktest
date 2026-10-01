from pathlib import Path

import pytest

from squeaktest.config import MB, Settings


def test_defaults_match_the_brief():
    s = Settings()
    assert s.max_file_bytes == 25 * MB
    assert s.max_duration_s == 300.0
    assert s.decode_timeout_s == 30.0
    assert s.temp_dir is None


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
        }
    )
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
