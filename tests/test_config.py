from scurve.config import load_config


def test_load_config_resolves_paths():
    cfg = load_config()
    assert cfg["paths"]["raw_dir"].is_absolute()
    assert cfg["sample"]["nonevent_keep_pct"] == 5
    assert cfg["regimes"]["lockin"][0] == "2022-07"
