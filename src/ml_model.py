"""
XGBoost model for football match outcome prediction.
"""
import numpy as np
import pandas as pd
import streamlit as st
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, log_loss
)
from sklearn.calibration import calibration_curve

FEATURES = [
    "HomeElo", "AwayElo", "EloDiff",
    "Form3Home", "Form5Home", "Form3Away", "Form5Away",
    "FairHome", "FairDraw", "FairAway",
]

FEATURE_LABELS = {
    "HomeElo": "Home Elo Rating",
    "AwayElo": "Away Elo Rating",
    "EloDiff": "Elo Difference (H-A)",
    "Form3Home": "Home Form (last 3)",
    "Form5Home": "Home Form (last 5)",
    "Form3Away": "Away Form (last 3)",
    "Form5Away": "Away Form (last 5)",
    "FairHome": "Fair Prob – Home Win",
    "FairDraw": "Fair Prob – Draw",
    "FairAway": "Fair Prob – Away Win",
}


@st.cache_resource(show_spinner="Treinando modelo XGBoost...")
def train_model(df: pd.DataFrame):
    """Train XGBoost on available features. Returns model + metadata dict."""
    sub = df.dropna(subset=FEATURES + ["ResultCode"]).copy()

    X = sub[FEATURES].values
    y = sub["ResultCode"].values  # 0=H, 1=D, 2=A

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        eval_metric="mlogloss",
        random_state=42,
        n_jobs=-1,
    )
    model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False,
    )

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)

    report = classification_report(y_test, y_pred, target_names=["Home Win", "Draw", "Away Win"], output_dict=True)
    cm = confusion_matrix(y_test, y_pred)
    ll = log_loss(y_test, y_prob)
    auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="macro")

    importance = pd.DataFrame({
        "Feature": [FEATURE_LABELS.get(f, f) for f in FEATURES],
        "Importance": model.feature_importances_,
    }).sort_values("Importance", ascending=False)

    # Calibration curves per class
    calib = {}
    for cls_idx, cls_name in enumerate(["Home Win", "Draw", "Away Win"]):
        frac_pos, mean_pred = calibration_curve(
            (y_test == cls_idx).astype(int),
            y_prob[:, cls_idx],
            n_bins=10,
        )
        calib[cls_name] = {"frac_pos": frac_pos, "mean_pred": mean_pred}

    # Prediction on all data with features
    X_all = sub[FEATURES].values
    probs_all = model.predict_proba(X_all)
    sub = sub.copy()
    sub["PredHome"] = probs_all[:, 0]
    sub["PredDraw"] = probs_all[:, 1]
    sub["PredAway"] = probs_all[:, 2]
    sub["PredResult"] = model.predict(X_all)

    # Model-based value: where model prob > fair prob
    sub["ModelEdgeHome"] = sub["PredHome"] - sub["FairHome"]
    sub["ModelEdgeAway"] = sub["PredAway"] - sub["FairAway"]
    sub["ModelEdgeDraw"]  = sub["PredDraw"]  - sub["FairDraw"]

    return {
        "model": model,
        "report": report,
        "cm": cm,
        "log_loss": ll,
        "auc": auc,
        "importance": importance,
        "calib": calib,
        "predictions": sub,
        "n_train": len(X_train),
        "n_test": len(X_test),
    }


def model_roi(predictions: pd.DataFrame, edge_threshold: float = 0.05) -> pd.DataFrame:
    """ROI when betting only on games where model edge > threshold."""
    rows = []
    for market, odd_col, edge_col, result_val in [
        ("Home Win", "OddHome", "ModelEdgeHome", "H"),
        ("Draw",     "OddDraw",  "ModelEdgeDraw",  "D"),
        ("Away Win", "OddAway",  "ModelEdgeAway",  "A"),
    ]:
        sub = predictions.dropna(subset=[odd_col, edge_col, "FTResult"])
        sub = sub[sub[edge_col] > edge_threshold]
        n = len(sub)
        if n == 0:
            continue
        wins = (sub["FTResult"] == result_val).sum()
        profit = (
            (sub["FTResult"] == result_val).astype(float) * (sub[odd_col] - 1)
        ).sum() - ((sub["FTResult"] != result_val).astype(float)).sum()
        rows.append({
            "Market": market,
            "Bets": n,
            "Win%": round(wins / n * 100, 2),
            "ROI%": round(profit / n * 100, 2),
            "Profit": round(profit, 2),
        })
    return pd.DataFrame(rows)
