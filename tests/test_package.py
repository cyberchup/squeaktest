import squeaktest


def test_version_matches_package_metadata():
    assert squeaktest.__version__ == "0.1.0.dev0"
