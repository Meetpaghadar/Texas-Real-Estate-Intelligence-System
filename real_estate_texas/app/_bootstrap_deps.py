"""Install runtime packages when Streamlit Cloud skips requirements.txt."""
from __future__ import annotations

import importlib.util
import subprocess
import sys

# Keep this lean — heavy extras (xgboost, wordcloud) are optional and imported lazily.
_PACKAGES: tuple[tuple[str, str], ...] = (
    ("joblib", "joblib==1.4.2"),
    ("numpy", "numpy==1.26.4"),
    ("pandas", "pandas==2.2.3"),
    ("scipy", "scipy==1.14.1"),
    ("sklearn", "scikit-learn==1.7.1"),
    ("plotly", "plotly>=5.24.0"),
)


def ensure_runtime_packages() -> None:
    missing = [pip for mod, pip in _PACKAGES if importlib.util.find_spec(mod) is None]
    if not missing:
        return
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "--quiet", *missing],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.STDOUT,
    )
