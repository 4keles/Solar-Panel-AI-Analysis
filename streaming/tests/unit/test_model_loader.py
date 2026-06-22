import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch

from src.core.model_loader import ModelLoader
from src.core.exceptions import ModelLoadError

@pytest.fixture
def mock_yolo_cls():
    with patch("src.core.model_loader.YOLO") as MockYOLO:
        mock_instance = MagicMock()
        mock_instance.names = {0: "physical_damage", 1: "dust_particle"}
        mock_instance.device = "mock-device"
        MockYOLO.return_value = mock_instance
        yield MockYOLO

@pytest.fixture
def sample_pt_file(tmp_path: Path) -> Path:
    pt = tmp_path / "best.pt"
    pt.write_bytes(b"fake_model_weights")
    return pt

def test_load_valid_model(mock_yolo_cls, sample_pt_file):
    loader = ModelLoader()
    model = loader.load(sample_pt_file)
    assert model is not None
    mock_yolo_cls.assert_called_once_with(str(sample_pt_file), task="detect")
    
def test_load_missing_file_raises():
    loader = ModelLoader()
    missing_path = Path("/invalid/path/missing_model.pt")
    with pytest.raises(ModelLoadError):
         loader.load(missing_path)

def test_get_class_names_after_load(mock_yolo_cls, sample_pt_file):
    loader = ModelLoader()
    loader.load(sample_pt_file)
    names = loader.get_class_names()
    assert names == {0: "physical_damage", 1: "dust_particle"}

def test_get_class_names_before_load():
    loader = ModelLoader()
    names = loader.get_class_names()
    assert names == {}

def test_load_latest_finds_best_pt(mock_yolo_cls, tmp_path):
    models_dir = tmp_path / "models"
    latest_dir = models_dir / "latest"
    latest_dir.mkdir(parents=True)
    best_pt = latest_dir / "best.pt"
    best_pt.write_bytes(b"temp")
    
    loader = ModelLoader()
    model = loader.load_latest(models_dir)
    assert model is not None
    mock_yolo_cls.assert_called_once_with(str(best_pt), task="detect")

def test_metadata_missing_returns_empty(mock_yolo_cls, sample_pt_file):
    loader = ModelLoader()
    loader.load(sample_pt_file)
    metadata = loader.get_metadata()
    assert metadata == {}

def test_metadata_present_returns_data(mock_yolo_cls, sample_pt_file):
    import json
    metadata_path = sample_pt_file.parent / "metadata.json"
    metadata_path.write_text(json.dumps({"version": "v1.0"}), encoding="utf-8")
    
    loader = ModelLoader()
    loader.load(sample_pt_file)
    metadata = loader.get_metadata()
    assert metadata == {"version": "v1.0"}
