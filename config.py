import os
import sys

from sklearn.ensemble import RandomForestClassifier

# Path to directory with sslearn folder if not in the PATH already.
sslearn_root = os.path.dirname(__file__)
sys.path.append(sslearn_root)

# Import algorithm to test.
from sslearn.models.sla._customs import BinarySLARandomForestClassifier  # noqa
from sslearn.models.lda import (
    SemiSupervisedLinearDiscriminantAnalysis,
    LinearDiscriminantAnalysis,
)  # noqa
from sslearn.models.lgc import LGC  # noqa
from sslearn.models.rf import (
    SemiSupervisedRandomForest,
    RandomForest,
)  # noqa

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
    },
    "lda": {
        "model_cls": SemiSupervisedLinearDiscriminantAnalysis,
        "baseline_cls": LinearDiscriminantAnalysis,
    },
    "lgc": {"model_cls": LGC},
    "rf": {"model_cls": SemiSupervisedRandomForest, "baseline_cls": RandomForest},
}
