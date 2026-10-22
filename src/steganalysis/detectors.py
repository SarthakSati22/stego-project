"""Audio, video and fusion steganalyzers (scikit-learn, CPU-friendly)."""
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def _logit(p):
    p = np.clip(p, 1e-4, 1 - 1e-4)
    return np.log(p / (1 - p))


class Detector:
    """One-modality detector: features in, probability of 'contains hidden data' out."""

    def __init__(self, seed: int = 0):
        self.model = make_pipeline(
            StandardScaler(),
            HistGradientBoostingClassifier(max_iter=200, learning_rate=0.08,
                                           max_leaf_nodes=15, l2_regularization=1.0,
                                           random_state=seed))

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def score(self, X) -> np.ndarray:
        return self.model.predict_proba(X)[:, 1]


class FusionDetector:
    """Late fusion: combines the audio and video scores of one clip.

    The fusion weights are learned on held-out scores (validation split) so the
    two base detectors cannot simply have memorised the training clips.
    """

    def __init__(self):
        self.model = LogisticRegression(C=10.0)
        self.neutral = np.zeros(2)

    def fit(self, score_a, score_v, y):
        X = np.column_stack([_logit(score_a), _logit(score_v)])
        self.neutral = X.mean(axis=0)
        self.model.fit(X, y)
        return self

    def score(self, score_a, score_v, mask_audio=False, mask_video=False) -> np.ndarray:
        """mask_* replaces that modality's score with an uninformative value (ablation)."""
        X = np.column_stack([_logit(score_a), _logit(score_v)])
        if mask_audio:
            X[:, 0] = self.neutral[0]
        if mask_video:
            X[:, 1] = self.neutral[1]
        return self.model.predict_proba(X)[:, 1]

    @property
    def weights(self):
        return self.model.coef_[0]


def auc(y, s) -> float:
    return float(roc_auc_score(y, s)) if len(set(y)) > 1 else float("nan")