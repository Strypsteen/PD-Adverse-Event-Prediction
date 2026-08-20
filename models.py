"""
Beta-calibrated random forest.

A 5-fold internal split of the training data: each fold trains a
RandomForestClassifier on 4/5 and fits a BetaCalibration on its held-out 1/5.
predict_proba averages the 5 calibrated probabilities. IPCW sample weights are
passed to both the forest and the calibrator.

Note on the fold splitter. The research code exposed a `stratified` flag and the
paper runs were launched with it set to False, but the class overrode it to True
internally, so every reported result used a shuffled StratifiedKFold. That is
what is hard-coded here; making the flag effective would change the numbers.
"""
import numpy as np
from betacal import BetaCalibration
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.model_selection import StratifiedKFold


class BetaCalibratedClassifierCV(BaseEstimator, ClassifierMixin):
    def __init__(self, base_estimator, X_fields, cv=5, parameters='abm', seed=42):
        self.base_estimator = base_estimator
        self.X_fields = X_fields
        self.cv = cv
        self.parameters = parameters
        self.seed = seed
        self.models_ = []
        self.calibrators_ = []

    def fit(self, X, y, sample_weight=None):
        self.models_ = []
        self.calibrators_ = []

        X = np.asarray(X)
        y = np.asarray(y)
        if sample_weight is not None:
            sample_weight = np.asarray(sample_weight)

        skf = StratifiedKFold(n_splits=self.cv, shuffle=True, random_state=self.seed)
        for train_idx, calib_idx in skf.split(X, y):
            sw_train = sample_weight[train_idx] if sample_weight is not None else None
            sw_calib = sample_weight[calib_idx] if sample_weight is not None else None

            model = clone(self.base_estimator)
            model.fit(X[train_idx], y[train_idx], sample_weight=sw_train)
            self.models_.append(model)

            probs = model.predict_proba(X[calib_idx])[:, 1]
            calibrator = BetaCalibration(parameters=self.parameters)
            calibrator.fit(probs.reshape(-1, 1), y[calib_idx], sample_weight=sw_calib)
            self.calibrators_.append(calibrator)

        return self

    def predict_proba(self, X):
        probs = np.zeros(len(X))
        for model, calibrator in zip(self.models_, self.calibrators_):
            raw = model.predict_proba(X)[:, 1]
            probs += calibrator.predict(raw.reshape(-1, 1))
        probs /= len(self.models_)
        return np.column_stack([1 - probs, probs])

    def predict(self, X):
        return self.predict_proba(X)[:, 1] >= 0.5
