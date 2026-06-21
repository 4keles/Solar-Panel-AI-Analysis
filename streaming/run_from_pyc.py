"""
Temporary launcher — runs streaming from .pyc bytecode while source .py files are missing.
Usage: cd ~/git/solar_panel_od && python3.13 streaming/run_from_pyc.py
"""
import sys
import os
import importlib.util
import importlib.machinery
import importlib.abc

STREAMING_DIR = os.path.dirname(os.path.abspath(__file__))
VENV_SITE = os.path.join(os.path.dirname(STREAMING_DIR), ".venv", "lib", "python3.13", "site-packages")
SCRIPTS_DIR = os.path.join(os.path.dirname(STREAMING_DIR), "scripts")

sys.path.insert(0, VENV_SITE)
sys.path.insert(0, SCRIPTS_DIR)
sys.path.insert(0, STREAMING_DIR)

PYTHON_TAG = "cpython-313"


def _pkg_dir(parts):
    return os.path.join(STREAMING_DIR, *parts)


def _find_pyc(parts):
    """Find .pyc for a dotted module name split into parts.
    Handles both regular modules and packages (__init__.pyc).
    """
    if not parts:
        return None
    *pkg, mod = parts
    pkg_dir = _pkg_dir(pkg) if pkg else STREAMING_DIR

    # Regular module: <pkg_dir>/__pycache__/<mod>.cpython-313.pyc
    pyc = os.path.join(pkg_dir, "__pycache__", f"{mod}.{PYTHON_TAG}.pyc")
    if os.path.isfile(pyc):
        return pyc

    # Package: <pkg_dir>/<mod>/__pycache__/__init__.cpython-313.pyc
    pkg_init = os.path.join(pkg_dir, mod, "__pycache__", f"__init__.{PYTHON_TAG}.pyc")
    if os.path.isfile(pkg_init):
        return pkg_init

    return None


def _is_package(parts):
    """Return True if this module is a package (has a sub-directory for it)."""
    return os.path.isdir(_pkg_dir(parts))


class PycFallbackFinder(importlib.abc.MetaPathFinder):
    """Load modules from __pycache__/*.pyc when the source .py is missing."""

    def find_spec(self, fullname, path, target=None):
        parts = fullname.split(".")
        pyc_path = _find_pyc(parts)
        if not pyc_path:
            return None

        loader = importlib.machinery.SourcelessFileLoader(fullname, pyc_path)
        is_pkg = _is_package(parts)

        spec = importlib.util.spec_from_loader(fullname, loader, origin=pyc_path)
        spec.has_location = True

        if is_pkg:
            spec.submodule_search_locations = [_pkg_dir(parts)]

        return spec


sys.meta_path.insert(0, PycFallbackFinder())


_loading = set()  # guard against circular recursion


def _preload(dotted_name):
    """Force-load a module from .pyc. Parents loaded first (from their __init__.pyc)."""
    if dotted_name in sys.modules:
        return sys.modules[dotted_name]
    if dotted_name in _loading:
        return sys.modules.get(dotted_name)

    _loading.add(dotted_name)
    try:
        parts = dotted_name.split(".")
        # Ensure parent exists first
        if len(parts) > 1:
            _preload(".".join(parts[:-1]))

        # Already loaded by parent init? return it
        if dotted_name in sys.modules:
            return sys.modules[dotted_name]

        pyc = _find_pyc(parts)
        if not pyc:
            raise ImportError(f"Cannot find .pyc for {dotted_name}")

        loader = importlib.machinery.SourcelessFileLoader(dotted_name, pyc)
        spec = importlib.util.spec_from_loader(dotted_name, loader, origin=pyc)
        spec.has_location = True
        if _is_package(parts):
            spec.submodule_search_locations = [_pkg_dir(parts)]

        mod = importlib.util.module_from_spec(spec)
        mod.__package__ = dotted_name if _is_package(parts) else ".".join(parts[:-1])
        sys.modules[dotted_name] = mod
        spec.loader.exec_module(mod)
        return mod
    finally:
        _loading.discard(dotted_name)


# Load leaf modules first, then packages (so __init__ finds submodules already in sys.modules)
_load_order = [
    # utils leaves
    "src.utils.class_colors",
    "src.utils.fps_counter",
    # core leaves (exceptions first — others import from it)
    "src.core.exceptions",
    "src.core.model_loader",
    "src.core.recorder",
    "src.core.frame_processor",
    "src.core.annotator",
    "src.core.source_manager",
    # ui widget leaves
    "src.ui.widgets.video_widget",
    "src.ui.widgets.stats_panel",
    "src.ui.widgets.source_selector",
    "src.ui.widgets.thermal_control",
    "src.ui.widgets.path_manager_bar",
    "src.ui.widgets.help_window",
    "src.ui.widgets.playback_control",
    "src.ui.widgets.image_gallery",
    # packages (their __init__ runs now with submodules already loaded)
    "src.utils",
    "src.core",
    "src.ui.widgets",
    "src.ui",
    "src",
]
for _mod in _load_order:
    try:
        _preload(_mod)
    except Exception as e:
        print(f"[preload warning] {_mod}: {e}", file=sys.stderr)

# Run main
_main_pyc = os.path.join(STREAMING_DIR, "__pycache__", f"main.{PYTHON_TAG}.pyc")
loader = importlib.machinery.SourcelessFileLoader("__main__", _main_pyc)
spec = importlib.util.spec_from_loader("__main__", loader)
mod = importlib.util.module_from_spec(spec)
sys.modules["__main__"] = mod
spec.loader.exec_module(mod)
