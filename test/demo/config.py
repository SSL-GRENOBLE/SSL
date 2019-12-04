import sys

from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

# Path to directory with sslearn folder if not in the PATH already.
path = str(Path(__file__).resolve().parents[2])
sys.path.append(path)

# Import algorithm to test.
from sslearn.models.sla._customs import BinarySLARandomForestClassifier  # noqa


configs = {
    "sla": {
        "model_cls": BinarySLARandomForestClassifier,
        "baseline_cls": RandomForestClassifier,
        "model_inits": {
            "n_estimators": 100,
            "adaptive": True,
            "margin_mode": "soft",
            "max_iter": 150,
        },
        "baseline_inits": {"n_estimators": 100},
    }
}
