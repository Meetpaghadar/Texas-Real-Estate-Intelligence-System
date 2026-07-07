"""Repo-root Streamlit entrypoint (requirements.txt co-located for Cloud)."""
from __future__ import annotations

import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
APP = ROOT / "real_estate_texas" / "app" / "streamlit_app.py"

if str(ROOT / "real_estate_texas") not in sys.path:
    sys.path.insert(0, str(ROOT / "real_estate_texas"))

runpy.run_path(str(APP), run_name="__main__")
