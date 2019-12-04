from typing import Optional

from sklearn.ensemble import RandomForestClassifier

from .sla import BinarySLA


class BinarySLARandomForestClassifier(BinarySLA):
    def __init__(
        self,
        adaptive: bool = False,
        theta: Optional[float] = None,
        margin_mode: str = "soft",
        max_iter: int = 100,
        random_state: int = 0,
        **kwargs
    ) -> None:
        super(BinarySLARandomForestClassifier, self).__init__(
            RandomForestClassifier(random_state=random_state, **kwargs),
            adaptive,
            theta,
            margin_mode,
            max_iter,
        )
