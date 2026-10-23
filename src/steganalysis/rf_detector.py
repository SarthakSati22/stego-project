"""A second, independent detector family, used only for evaluation."""
import numpy as np
from sklearn.ensemble import RandomForestClassifier


class RFDetector:
    """Random-forest steganalyzer.

    The allocator never sees it, so it gives an unbiased view of detectability for
    strategies that were tuned against the main (gradient-boosting) detectors.
    """

    def __init__(self, seed: int = 1):
        self.model = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                            n_jobs=-1, random_state=seed)

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def score(self, X) -> np.ndarray:
           
        self.model.set_params(n_jobs=1)     # one clip at a time: extra threads only add overhead
        return self.model.predict_proba(X)[:, 1]
        