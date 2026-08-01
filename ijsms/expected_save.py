from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

NUMERIC_FEATURES = [
    "shot_xg",
    "shot_distance",
    "shot_angle",
    "goalkeeper_distance_to_goal_center",
    "end_z",
    "under_pressure",
    "one_on_one",
    "open_goal",
    "first_time",
    "deflected",
    "minute",
    "score_diff_before_shot",
]
CATEGORICAL_FEATURES = ["shot_type", "body_part", "technique", "play_pattern"]


def classify_save_outcome(actual_save: int, expected_save_probability: float) -> str:
    if int(actual_save) == 1:
        return "heroic_save" if expected_save_probability < 0.5 else "routine_save"
    return "understandable_goal" if expected_save_probability < 0.5 else "unexpected_goal"


def _model() -> Pipeline:
    numeric = Pipeline(
        [
            ("impute", SimpleImputer(strategy="median")),
            ("scale", StandardScaler()),
        ]
    )
    categorical = Pipeline(
        [
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", min_frequency=2)),
        ]
    )
    preprocess = ColumnTransformer(
        [
            ("numeric", numeric, NUMERIC_FEATURES),
            ("categorical", categorical, CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocess),
            ("model", LogisticRegression(max_iter=3000, C=0.5)),
        ]
    )


def cross_fit_expected_save(shots: pd.DataFrame, *, n_splits: int = 8) -> tuple[pd.DataFrame, dict[str, Any]]:
    frame = shots.copy().reset_index(drop=True)
    for column in NUMERIC_FEATURES:
        if column not in frame:
            frame[column] = 0.0
    for column in CATEGORICAL_FEATURES:
        if column not in frame:
            frame[column] = "Unknown"
    if "sb_match_id" not in frame or "actual_save" not in frame:
        raise ValueError("Shots require sb_match_id and actual_save")
    groups = frame["sb_match_id"]
    unique_groups = groups.nunique()
    if unique_groups < 2:
        raise ValueError("At least two matches are required for cross-fitting")
    splitter = GroupKFold(n_splits=min(n_splits, unique_groups))
    probabilities = np.full(len(frame), np.nan)
    folds = np.full(len(frame), -1, dtype=int)
    leakage_count = 0
    features = frame[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    target = frame["actual_save"].astype(int)
    for fold, (train_idx, test_idx) in enumerate(splitter.split(features, target, groups)):
        train_groups = set(groups.iloc[train_idx])
        test_groups = set(groups.iloc[test_idx])
        leakage_count += len(train_groups.intersection(test_groups))
        model = _model()
        model.fit(features.iloc[train_idx], target.iloc[train_idx])
        probabilities[test_idx] = model.predict_proba(features.iloc[test_idx])[:, 1]
        folds[test_idx] = fold
    if np.isnan(probabilities).any():
        raise RuntimeError("Cross-fitting failed to predict every shot")
    frame["expected_save_probability"] = np.clip(probabilities, 1e-6, 1 - 1e-6)
    frame["prediction_fold"] = folds
    frame["performance_category"] = [
        classify_save_outcome(save, probability)
        for save, probability in zip(frame["actual_save"], frame["expected_save_probability"])
    ]
    prevalence = float(target.mean())
    intercept_predictions = np.repeat(prevalence, len(target))
    diagnostics: dict[str, Any] = {
        "observations": int(len(frame)),
        "matches": int(unique_groups),
        "save_rate": prevalence,
        "brier_score": float(brier_score_loss(target, probabilities)),
        "intercept_brier_score": float(brier_score_loss(target, intercept_predictions)),
        "log_loss": float(log_loss(target, probabilities)),
        "group_leakage_count": int(leakage_count),
        "folds": int(len(np.unique(folds))),
        "category_counts": frame["performance_category"].value_counts().to_dict(),
    }
    if target.nunique() == 2:
        diagnostics["roc_auc"] = float(roc_auc_score(target, probabilities))
    return frame, diagnostics
