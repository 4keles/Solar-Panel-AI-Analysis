import sys
from pathlib import Path
from ultralytics import YOLO

from .exceptions import ModelLoadError

# Ana projedeki logglerı sistem pathi aracılığıyla import etme kuralı
_scripts_dir = Path(__file__).resolve().parent.parent.parent.parent / "scripts"
if str(_scripts_dir) not in sys.path:
    sys.path.insert(0, str(_scripts_dir))
from utils.logger import get_logger

logger = get_logger(__name__)


class ModelLoader:
    def __init__(self) -> None:
        self._model: YOLO | None = None
        self._model_path: Path | None = None

    def load(self, model_path: Path) -> YOLO:
        """Load YOLO model from path.
        Args:
            model_path: Absolute path to .pt file.
        Returns:
            Loaded YOLO instance.
        Raises:
            ModelLoadError: If file not found or load fails.
        """
        if not model_path.exists():
            logger.error("model_file_missing", path=str(model_path))
            raise ModelLoadError(f"Model dosyası bulunamadı: {model_path}")

        try:
            logger.info("loading_model", path=str(model_path))
            self._model = YOLO(str(model_path), task="detect")
            self._model_path = model_path
            
            # Simple warmup or device check (dummy check based on ultralytics YOLO structure)
            device = self._model.device if hasattr(self._model, "device") else "unknown"
            logger.info("model_loaded_successfully", device=str(device))
            return self._model
        except Exception as e:
            logger.error("model_load_failed", path=str(model_path), error=str(e))
            raise ModelLoadError(f"Model yüklenirken bir hata oluştu: {e}") from e

    def load_latest(self, models_dir: Path, modality: str = "rgb") -> YOLO:
        """Loads the latest best model of the given modality."""
        latest_dir = models_dir / "latest"
        
        # Check specific modality best.pt or general best.pt
        modality_pt = latest_dir / f"{modality}_best.pt"
        if modality_pt.exists():
            return self.load(modality_pt)
        
        best_pt = latest_dir / "best.pt"
        if best_pt.exists():
            return self.load(best_pt)

        logger.error("latest_model_not_found", search_dir=str(latest_dir))
        raise ModelLoadError(f"{latest_dir} altında 'best.pt' veya '{modality}_best.pt' bulunamadı.")

    def get_class_names(self) -> dict[int, str]:
        """Tüm sınıf eşleşmelerini dict tipinde geri döndürür."""
        if not self._model:
            return {}
        return self._model.names if hasattr(self._model, "names") else {}

    def get_metadata(self) -> dict:
        """Sadece model_path bazlı metadata json dosyasını okuyup dict döner."""
        if not self._model_path:
            return {}

        metadata_path = self._model_path.parent / "metadata.json"
        if not metadata_path.exists():
            return {}
            
        import json
        try:
            content = json.loads(metadata_path.read_text(encoding="utf-8"))
            return content
        except Exception:
            return {}
