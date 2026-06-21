"""Entry point — delegates to streaming launcher (run from project root)."""
import runpy, pathlib, sys

sys.path.insert(0, str(pathlib.Path(__file__).parent))
runpy.run_path(str(pathlib.Path(__file__).parent / "streaming" / "run_from_pyc.py"), run_name="__main__")
