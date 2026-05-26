"""Entry point — launches the FastAPI web UI."""
import os
import sys

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

# Import from absolute path to avoid 'app' package conflict
import importlib.util, pathlib

_server_path = pathlib.Path(ROOT) / "app" / "web" / "server.py"
_spec = importlib.util.spec_from_file_location("web_server", _server_path)
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)

fastapi_app = _mod.app
get_pipeline = _mod.get_pipeline

import uvicorn

if __name__ == "__main__":
    print("[BOM Detector] Loading pipeline...")
    get_pipeline()
    print("[BOM Detector] → http://localhost:8000")
    uvicorn.run(fastapi_app, host="0.0.0.0", port=8000, log_level="info")
