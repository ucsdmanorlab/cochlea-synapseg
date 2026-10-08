from napari_cochlea_synapse_seg._settings import load_settings, save_settings


def test_settings_round_trip(tmp_path):
    path = tmp_path / "settings.json"
    settings = {"GTWidget": {"xy_res": 0.25}}

    assert save_settings(settings, settings_path=path) is True

    loaded = load_settings(settings_path=path)
    assert loaded["GTWidget"] == {"xy_res": 0.25}
    assert loaded["version"] == "1.0"
    # sections missing from the file are filled in with empty dicts
    assert loaded["PredWidget"] == {}
    assert loaded["CropWidget"] == {}
    assert loaded["PreProcessWidget"] == {}
