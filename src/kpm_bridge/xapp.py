"""A fixed portable QoS-risk xApp used for downstream stability evaluation."""

from __future__ import annotations

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


class PortableRiskXApp:
    def __init__(self, random_state: int = 20260712, action_threshold: float = 0.20):
        self.action_threshold = float(action_threshold)
        self.logit_threshold = float(np.log(self.action_threshold / (1.0 - self.action_threshold)))
        self.scaler = StandardScaler()
        self.classifier = LogisticRegression(
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            random_state=random_state,
        )

    def fit(self, values: np.ndarray, risk: np.ndarray) -> "PortableRiskXApp":
        standard = self.scaler.fit_transform(values)
        self.classifier.fit(standard, risk)
        return self

    def logits(self, values: np.ndarray) -> np.ndarray:
        return self.classifier.decision_function(self.scaler.transform(values))

    def probabilities(self, values: np.ndarray) -> np.ndarray:
        return self.classifier.predict_proba(self.scaler.transform(values))[:, 1]

    def actions(self, values: np.ndarray) -> np.ndarray:
        return (self.logits(values) >= self.logit_threshold).astype(int)

    def uncertainty_margin(self, radius: float, feature_scale: np.ndarray) -> float:
        weights = self.classifier.coef_[0]
        scaled_radius = feature_scale / self.scaler.scale_
        return float(radius * np.linalg.norm(weights * scaled_radius, ord=2))


class DeploymentSpecificRiskXApp:
    """Deployment-specific retraining upper bound; it is not portable."""

    def __init__(self, random_state: int = 20260712, action_threshold: float = 0.20):
        self.action_threshold = float(action_threshold)
        self.model = make_pipeline(
            SimpleImputer(strategy="median", add_indicator=True),
            StandardScaler(),
            LogisticRegression(
                C=1.0,
                class_weight="balanced",
                max_iter=2000,
                random_state=random_state,
            ),
        )

    def fit(self, values: np.ndarray, risk: np.ndarray) -> "DeploymentSpecificRiskXApp":
        self.model.fit(values, risk)
        return self

    def actions(self, values: np.ndarray) -> np.ndarray:
        return (self.probabilities(values) >= self.action_threshold).astype(int)

    def probabilities(self, values: np.ndarray) -> np.ndarray:
        return self.model.predict_proba(values)[:, 1]


def decision_regret(actions: np.ndarray, risk: np.ndarray) -> np.ndarray:
    """Regret to a clairvoyant QoS action under 0.2 intervention cost."""
    actions = np.asarray(actions, dtype=int)
    risk = np.asarray(risk, dtype=int)
    return 0.8 * ((risk == 1) & (actions == 0)) + 0.2 * ((risk == 0) & (actions == 1))
