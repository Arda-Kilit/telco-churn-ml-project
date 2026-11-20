"""
BTK Uçtan Uca Makine Öğrenmesi Projesi — Telco Customer Churn
Aşama 5–12: Sızıntısız Pipeline, Feature Engineering, Baseline, 7 Güçlü Model
Karşılaştırması (6+ Güçlü Rakip), Adil Hiperparametre Optimizasyonu,
Eşik (Threshold) Optimizasyonu, Final Test Değerlendirmesi ve Hata Analizi.

Kullanım:
    python train.py --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv
"""

import argparse
import json
import os
import time
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.features import CATEGORICAL_FEATURES, NUMERIC_FEATURES, TelcoFeatureEngineer

# Optional boosting libraries (recommended in requirements.txt)
try:
    from xgboost import XGBClassifier
    HAS_XGB = True
except ImportError:
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier
    HAS_LGBM = True
except ImportError:
    HAS_LGBM = False

try:
    from catboost import CatBoostClassifier
    HAS_CAT = True
except ImportError:
    HAS_CAT = False

try:
    import shap
    HAS_SHAP = True
except ImportError:
    HAS_SHAP = False

warnings.filterwarnings("ignore")
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "figure.dpi": 200,
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})


def build_preprocessor():
    """Creates zero-leakage ColumnTransformer inside Pipeline."""
    return ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def make_pipeline(estimator):
    return Pipeline(steps=[
        ("fe", TelcoFeatureEngineer()),
        ("prep", build_preprocessor()),
        ("model", estimator),
    ])


def evaluate_predictions(y_true, y_prob, threshold=0.5):
    y_pred = (y_prob >= threshold).astype(int)
    return {
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "threshold": float(threshold),
    }


def find_best_threshold(y_true, y_prob):
    thresholds = np.arange(0.15, 0.86, 0.01)
    best_t, best_f1 = 0.50, -1.0
    curve = []
    for t in thresholds:
        y_pred = (y_prob >= t).astype(int)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        prec = precision_score(y_true, y_pred, zero_division=0)
        curve.append({"threshold": float(t), "f1": float(f1), "recall": float(rec), "precision": float(prec)})
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    return best_t, float(best_f1), curve


def get_models(scale_pos_weight: float):
    models = {
        "Baseline (Dummy Most-Frequent)": DummyClassifier(strategy="most_frequent"),
        "Baseline (Dummy Stratified)": DummyClassifier(strategy="stratified", random_state=RANDOM_STATE),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=300, max_depth=10, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=300, max_depth=12, class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32), max_iter=300, early_stopping=True, random_state=RANDOM_STATE
        ),
    }

    if HAS_XGB:
        models["XGBoost"] = XGBClassifier(
            n_estimators=300,
            learning_rate=0.05,
            max_depth=4,
            scale_pos_weight=scale_pos_weight,
            eval_metric="logloss",
            random_state=RANDOM_STATE,
            n_jobs=-1,
        )
    if HAS_LGBM:
        models["LightGBM"] = LGBMClassifier(
            n_estimators=300,
            learning_rate=0.05,
            num_leaves=31,
            class_weight="balanced",
            random_state=RANDOM_STATE,
            verbose=-1,
            n_jobs=-1,
        )
    if HAS_CAT:
        models["CatBoost"] = CatBoostClassifier(
            iterations=350,
            learning_rate=0.05,
            depth=5,
            auto_class_weights="Balanced",
            random_seed=RANDOM_STATE,
            verbose=0,
        )
    return models


def get_param_distributions(scale_pos_weight: float):
    """Fair hyperparameter search spaces (equal n_iter budget across tuned contenders)."""
    return {
        "Logistic Regression": {
            "model__C": [0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0],
            "model__penalty": ["l2"],
            "model__solver": ["lbfgs", "liblinear"],
            "model__class_weight": ["balanced", None],
        },
        "Random Forest": {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [6, 8, 10, 14, None],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
            "model__max_features": ["sqrt", "log2", 0.5],
        },
        "Extra Trees": {
            "model__n_estimators": [200, 300, 500],
            "model__max_depth": [8, 12, 16, None],
            "model__min_samples_split": [2, 5, 10, 20],
            "model__min_samples_leaf": [1, 2, 4, 8],
        },
        "MLP Neural Network": {
            "model__hidden_layer_sizes": [(64,), (64, 32), (128, 64), (32, 16)],
            "model__alpha": [1e-4, 1e-3, 1e-2, 1e-1],
            "model__learning_rate_init": [5e-4, 1e-3, 5e-3],
        },
        "XGBoost": {
            "model__n_estimators": [200, 300, 450],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__max_depth": [3, 4, 5, 6],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__min_child_weight": [1, 3, 5, 10],
            "model__reg_lambda": [1.0, 3.0, 10.0],
            "model__scale_pos_weight": [1.0, scale_pos_weight],
        },
        "LightGBM": {
            "model__n_estimators": [200, 300, 450],
            "model__learning_rate": [0.01, 0.03, 0.05, 0.1],
            "model__num_leaves": [15, 31, 63],
            "model__max_depth": [-1, 4, 6, 8],
            "model__min_child_samples": [10, 20, 40, 60],
            "model__subsample": [0.7, 0.85, 1.0],
            "model__colsample_bytree": [0.7, 0.85, 1.0],
            "model__reg_lambda": [0.0, 1.0, 5.0],
        },
        "CatBoost": {
            "model__iterations": [250, 400, 600],
            "model__learning_rate": [0.02, 0.04, 0.06, 0.1],
            "model__depth": [3, 4, 5, 6],
            "model__l2_leaf_reg": [1, 3, 5, 10],
        },
    }


def main(data_path: str, out_dir: str, n_iter: int = 20):
    fig_dir = os.path.join(out_dir, "figures")
    res_dir = os.path.join(out_dir, "results")
    mod_dir = os.path.join(out_dir, "models")
    for d in [fig_dir, res_dir, mod_dir]:
        os.makedirs(d, exist_ok=True)

    print(f"[1/7] Veri yükleniyor: {data_path}")
    df = pd.read_csv(data_path)
    y = (df["Churn"] == "Yes").astype(int)
    X = df.drop(columns=["Churn"])

    # Stratified 70% Train / 15% Validation / 15% Test
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.30, stratify=y, random_state=RANDOM_STATE
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=RANDOM_STATE
    )

    scale_pos_weight = float((y_train == 0).sum() / (y_train == 1).sum())
    print(f"Split boyutları -> Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"Sınıf dengesizliği (Train neg/pos): {scale_pos_weight:.2f}")

    # ==========================================================
    # AŞAMA 7 & 8: Baseline & İlk Model Karşılaştırması (5-Fold CV)
    # ==========================================================
    print("\n[2/7] 5-Fold Stratified CV ile Baseline + Tüm Modeller karşılaştırılıyor...")
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "pr_auc": "average_precision",
        "roc_auc": "roc_auc",
        "f1": "f1",
        "recall": "recall",
        "precision": "precision",
        "accuracy": "accuracy",
    }

    models = get_models(scale_pos_weight)
    cv_results = []
    for name, est in models.items():
        t0 = time.time()
        pipe = make_pipeline(est)
        scores = cross_validate(pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1)
        elapsed = time.time() - t0
        row = {
            "model": name,
            "is_baseline": name.startswith("Baseline"),
            "cv_pr_auc_mean": float(scores["test_pr_auc"].mean()),
            "cv_pr_auc_std": float(scores["test_pr_auc"].std()),
            "cv_roc_auc_mean": float(scores["test_roc_auc"].mean()),
            "cv_roc_auc_std": float(scores["test_roc_auc"].std()),
            "cv_f1_mean": float(scores["test_f1"].mean()),
            "cv_recall_mean": float(scores["test_recall"].mean()),
            "cv_precision_mean": float(scores["test_precision"].mean()),
            "cv_accuracy_mean": float(scores["test_accuracy"].mean()),
            "fit_seconds": round(elapsed, 2),
        }
        cv_results.append(row)
        print(
            f"  {name:32s} | PR-AUC: {row['cv_pr_auc_mean']:.4f}±{row['cv_pr_auc_std']:.4f} "
            f"| ROC-AUC: {row['cv_roc_auc_mean']:.4f} | F1: {row['cv_f1_mean']:.4f} ({elapsed:.1f}s)"
        )

    cv_df = pd.DataFrame(cv_results).sort_values("cv_pr_auc_mean", ascending=False)

    # Plot Fig 4: CV Model Comparison
    fig, ax = plt.subplots(figsize=(10.5, 5.5))
    plot_df = cv_df.sort_values("cv_pr_auc_mean", ascending=True)
    y_pos = np.arange(len(plot_df))
    ax.barh(y_pos - 0.18, plot_df["cv_pr_auc_mean"], height=0.35, color="#E11D48", label="PR-AUC (Birincil Metrik)")
    ax.barh(y_pos + 0.18, plot_df["cv_roc_auc_mean"], height=0.35, color="#2563EB", label="ROC-AUC")
    ax.set_yticks(y_pos)
    ax.set_yticklabels(plot_df["model"], fontsize=9.5)
    ax.set_xlabel("5-Fold Stratified CV Skoru")
    ax.set_xlim(0, 1.02)
    ax.set_title("Baseline ve Güçlü Rakip Modellerin 5-Fold CV Karşılaştırması", fontsize=12, fontweight="bold")
    ax.legend(frameon=False, loc="lower right")
    ax.grid(axis="x", linestyle="--", alpha=0.35)
    sns.despine(ax=ax)
    for i, (_, r) in enumerate(plot_df.iterrows()):
        ax.text(r["cv_pr_auc_mean"] + 0.01, i - 0.18, f"{r['cv_pr_auc_mean']:.3f}", va="center", fontsize=8)
        ax.text(r["cv_roc_auc_mean"] + 0.01, i + 0.18, f"{r['cv_roc_auc_mean']:.3f}", va="center", fontsize=8)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig4_model_comparison_cv.png"))
    plt.close()

    # ==========================================================
    # AŞAMA 9, 10 & 11: Aday Seçimi + Adil Hiperparametre Optimizasyonu (Top 3)
    # ==========================================================
    print(f"\n[3/7] En iyi 3 aday model için adil hiperparametre optimizasyonu (n_iter={n_iter})...")
    non_baseline = cv_df[~cv_df["is_baseline"]].copy()
    top3_names = non_baseline["model"].head(3).tolist()
    param_dists = get_param_distributions(scale_pos_weight)

    tuned_pipelines = {}
    tuning_summary = []

    for name in top3_names:
        t0 = time.time()
        base_pipe = make_pipeline(models[name])
        search = RandomizedSearchCV(
            estimator=base_pipe,
            param_distributions=param_dists[name],
            n_iter=n_iter,
            scoring="average_precision",
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            refit=True,
        )
        search.fit(X_train, y_train)
        elapsed = time.time() - t0
        before_pr = float(cv_df.loc[cv_df["model"] == name, "cv_pr_auc_mean"].values[0])
        after_pr = float(search.best_score_)

        # Validation evaluation & optimal threshold
        val_prob = search.best_estimator_.predict_proba(X_val)[:, 1]
        best_t, val_best_f1, _ = find_best_threshold(y_val, val_prob)
        val_metrics_05 = evaluate_predictions(y_val, val_prob, threshold=0.50)
        val_metrics_opt = evaluate_predictions(y_val, val_prob, threshold=best_t)

        tuned_pipelines[name] = {
            "pipeline": search.best_estimator_,
            "best_params": {k: (str(v) if isinstance(v, tuple) else v) for k, v in search.best_params_.items()},
            "best_threshold": best_t,
            "val_metrics_05": val_metrics_05,
            "val_metrics_opt": val_metrics_opt,
        }
        tuning_summary.append({
            "model": name,
            "cv_pr_auc_before": before_pr,
            "cv_pr_auc_after": after_pr,
            "delta_pr_auc": round(after_pr - before_pr, 5),
            "val_pr_auc": val_metrics_opt["pr_auc"],
            "val_roc_auc": val_metrics_opt["roc_auc"],
            "val_f1_at_05": val_metrics_05["f1"],
            "val_f1_at_opt": val_metrics_opt["f1"],
            "optimal_threshold": best_t,
            "best_params": tuned_pipelines[name]["best_params"],
            "tune_seconds": round(elapsed, 2),
        })
        print(
            f"  [TUNED] {name:20s} | CV PR-AUC: {before_pr:.4f} -> {after_pr:.4f} "
            f"| Val PR-AUC: {val_metrics_opt['pr_auc']:.4f} | Opt Thr: {best_t:.2f} (Val F1: {val_metrics_opt['f1']:.4f})"
        )

    # Select final champion strictly on Validation PR-AUC (test set STILL untouched!)
    tuning_df = pd.DataFrame(tuning_summary).sort_values("val_pr_auc", ascending=False)
    champion_name = tuning_df.iloc[0]["model"]
    champion_pipe = tuned_pipelines[champion_name]["pipeline"]
    champion_thr = tuned_pipelines[champion_name]["best_threshold"]
    print(f"\n>>> SEÇİLEN FİNAL MODEL (Validation PR-AUC'a göre): {champion_name} (Eşik: {champion_thr:.2f})")

    # ==========================================================
    # AŞAMA 12: KİLİTLİ TEST SETİ ÜZERİNDE FİNAL DEĞERLENDİRME
    # ==========================================================
    print("\n[4/7] Kilitli Test Seti (n=%d) üzerinde tüm modeller değerlendiriliyor..." % len(X_test))
    test_results = []
    test_probs_for_plots = {}

    for name, est in models.items():
        if name in tuned_pipelines:
            pipe = tuned_pipelines[name]["pipeline"]
            thr = tuned_pipelines[name]["best_threshold"]
            label = f"{name} (Tuned)"
        else:
            pipe = make_pipeline(est)
            pipe.fit(X_train, y_train)
            thr = 0.50
            label = name

        y_prob = pipe.predict_proba(X_test)[:, 1]
        m_05 = evaluate_predictions(y_test, y_prob, threshold=0.50)
        m_opt = evaluate_predictions(y_test, y_prob, threshold=thr)
        test_results.append({
            "model": label,
            "is_champion": name == champion_name,
            "test_pr_auc": m_opt["pr_auc"],
            "test_roc_auc": m_opt["roc_auc"],
            "test_f1_at_05": m_05["f1"],
            "test_f1_at_opt": m_opt["f1"],
            "test_recall_at_opt": m_opt["recall"],
            "test_precision_at_opt": m_opt["precision"],
            "test_accuracy_at_opt": m_opt["accuracy"],
            "threshold_used": thr,
        })
        if not name.startswith("Baseline"):
            test_probs_for_plots[label] = y_prob

    test_df = pd.DataFrame(test_results).sort_values("test_pr_auc", ascending=False)
    print(test_df[["model", "test_pr_auc", "test_roc_auc", "test_f1_at_opt", "test_recall_at_opt", "test_precision_at_opt"]].to_string(index=False))

    # Plot Fig 5: ROC & PR Curves on Test Set (Top 5 non-baseline models)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    top_plot_names = test_df[~test_df["model"].str.startswith("Baseline")]["model"].head(5).tolist()
    colors = ["#E11D48", "#2563EB", "#10B981", "#F59E0B", "#7C3AED"]

    for idx, label in enumerate(top_plot_names):
        prob = test_probs_for_plots[label]
        fpr, tpr, _ = roc_curve(y_test, prob)
        prec, rec, _ = precision_recall_curve(y_test, prob)
        auc_v = roc_auc_score(y_test, prob)
        ap_v = average_precision_score(y_test, prob)
        lw = 2.6 if idx == 0 else 1.6
        axes[0].plot(fpr, tpr, color=colors[idx], lw=lw, label=f"{label} (AUC={auc_v:.3f})")
        axes[1].plot(rec, prec, color=colors[idx], lw=lw, label=f"{label} (AP={ap_v:.3f})")

    axes[0].plot([0, 1], [0, 1], "k--", alpha=0.4, label="Rastgele (0.500)")
    axes[0].set_title("Test Seti ROC Eğrisi", fontsize=11.5, fontweight="bold")
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate (Recall)")
    axes[0].legend(frameon=False, fontsize=8.5, loc="lower right")
    axes[0].grid(alpha=0.3)

    baseline_ap = y_test.mean()
    axes[1].axhline(baseline_ap, color="k", linestyle="--", alpha=0.4, label=f"Baseline ({baseline_ap:.3f})")
    axes[1].set_title("Test Seti Precision-Recall Eğrisi", fontsize=11.5, fontweight="bold")
    axes[1].set_xlabel("Recall (Duyarlılık)")
    axes[1].set_ylabel("Precision (Kesinlik)")
    axes[1].legend(frameon=False, fontsize=8.5, loc="upper right")
    axes[1].grid(alpha=0.3)

    for ax in axes:
        sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig5_roc_pr_curves_test.png"))
    plt.close()

    # Plot Fig 6: Champion Confusion Matrix + Threshold Trade-off
    champ_test_prob = champion_pipe.predict_proba(X_test)[:, 1]
    champ_test_pred = (champ_test_prob >= champion_thr).astype(int)
    cm = confusion_matrix(y_test, champ_test_pred)
    _, _, thr_curve = find_best_threshold(y_val, champion_pipe.predict_proba(X_val)[:, 1])
    thr_df = pd.DataFrame(thr_curve)

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=axes[0],
        xticklabels=["Tahmin: Kaldı (0)", "Tahmin: Ayrıldı (1)"],
        yticklabels=["Gerçek: Kaldı (0)", "Gerçek: Ayrıldı (1)"],
        annot_kws={"size": 13, "weight": "bold"},
    )
    axes[0].set_title(f"{champion_name} — Test Karışıklık Matrisi (Eşik={champion_thr:.2f})", fontsize=11, fontweight="bold")

    axes[1].plot(thr_df["threshold"], thr_df["f1"], label="F1-Score", color="#E11D48", lw=2.3)
    axes[1].plot(thr_df["threshold"], thr_df["recall"], label="Recall", color="#2563EB", lw=1.8)
    axes[1].plot(thr_df["threshold"], thr_df["precision"], label="Precision", color="#10B981", lw=1.8)
    axes[1].axvline(champion_thr, color="#0F172A", linestyle="--", lw=1.4, label=f"Seçilen Eşik ({champion_thr:.2f})")
    axes[1].set_title("Validation Seti Eşik Optimizasyonu", fontsize=11, fontweight="bold")
    axes[1].set_xlabel("Karar Eşiği (Threshold)")
    axes[1].set_ylabel("Skor")
    axes[1].legend(frameon=False, fontsize=8.5)
    axes[1].grid(alpha=0.3)
    sns.despine(ax=axes[1])
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig6_confusion_matrix_threshold.png"))
    plt.close()

    # ==========================================================
    # Feature Importance / SHAP (Fig 7)
    # ==========================================================
    print("\n[5/7] Özellik önemi / SHAP analizi hesaplanıyor...")
    prep = champion_pipe.named_steps["prep"]
    fe = champion_pipe.named_steps["fe"]
    model_est = champion_pipe.named_steps["model"]
    feature_names = prep.get_feature_names_out()
    clean_names = [n.replace("num__", "").replace("cat__", "") for n in feature_names]

    top_features = []
    if hasattr(model_est, "feature_importances_"):
        importances = model_est.feature_importances_
    elif hasattr(model_est, "coef_"):
        importances = np.abs(model_est.coef_[0])
    else:
        importances = None

    if importances is not None:
        imp_df = pd.DataFrame({"feature": clean_names, "importance": importances}).sort_values(
            "importance", ascending=False
        )
        top15 = imp_df.head(15).iloc[::-1]
        top_features = imp_df.head(15).to_dict(orient="records")
        fig, ax = plt.subplots(figsize=(9.5, 6.0))
        ax.barh(top15["feature"], top15["importance"], color="#2563EB", height=0.65)
        ax.set_title(f"{champion_name} — En Önemli 15 Özellik", fontsize=12, fontweight="bold")
        ax.set_xlabel("Önem Skoru")
        ax.grid(axis="x", linestyle="--", alpha=0.35)
        sns.despine(ax=ax)
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir, "fig7_feature_importance.png"))
        plt.close()

    # ==========================================================
    # Error Analysis (Fig 8)
    # ==========================================================
    print("[6/7] Hata analizi (False Positive / False Negative kırılımı) hesaplanıyor...")
    err_df = X_test.copy()
    err_df["y_true"] = y_test.values
    err_df["y_pred"] = champ_test_pred
    err_df["y_prob"] = champ_test_prob
    err_df["error_type"] = np.where(
        (err_df["y_true"] == 1) & (err_df["y_pred"] == 0), "FN (Kaçırılan Churn)",
        np.where((err_df["y_true"] == 0) & (err_df["y_pred"] == 1), "FP (Yanlış Alarm)", "Doğru Tahmin")
    )

    err_by_contract = (
        err_df.groupby("Contract")["error_type"]
        .value_counts(normalize=True)
        .unstack(fill_value=0) * 100
    )

    fig, ax = plt.subplots(figsize=(8.5, 4.6))
    cols_to_plot = [c for c in ["FN (Kaçırılan Churn)", "FP (Yanlış Alarm)"] if c in err_by_contract.columns]
    err_by_contract[cols_to_plot].plot(kind="bar", ax=ax, color=["#E11D48", "#F59E0B"], width=0.55)
    ax.set_title("Sözleşme Tipine Göre Hata Oranları (Test Seti)", fontsize=11.5, fontweight="bold")
    ax.set_ylabel("Oran (%)")
    ax.set_xlabel("Sözleşme Tipi")
    plt.xticks(rotation=0)
    ax.legend(frameon=False)
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    sns.despine(ax=ax)
    plt.tight_layout()
    plt.savefig(os.path.join(fig_dir, "fig8_error_analysis.png"))
    plt.close()

    # ==========================================================
    # Save Model & Metrics JSON
    # ==========================================================
    print("[7/7] Final model ve metrikler kaydediliyor...")
    model_path = os.path.join(mod_dir, "best_model.joblib")
    joblib.dump(champion_pipe, model_path)

    metrics_payload = {
        "dataset": {
            "source": "https://www.kaggle.com/datasets/blastchar/telco-customer-churn",
            "n_rows": int(len(df)),
            "n_cols": int(df.shape[1]),
            "train_size": int(len(X_train)),
            "val_size": int(len(X_val)),
            "test_size": int(len(X_test)),
            "churn_rate_overall": float(y.mean()),
            "random_state": RANDOM_STATE,
        },
        "cv_comparison": cv_df.to_dict(orient="records"),
        "tuning_comparison": tuning_df.to_dict(orient="records"),
        "champion": {
            "name": champion_name,
            "optimal_threshold": float(champion_thr),
            "best_params": tuned_pipelines[champion_name]["best_params"],
            "test_metrics_at_opt": evaluate_predictions(y_test, champ_test_prob, threshold=champion_thr),
            "test_metrics_at_05": evaluate_predictions(y_test, champ_test_prob, threshold=0.50),
            "confusion_matrix": {
                "tn": int(cm[0, 0]),
                "fp": int(cm[0, 1]),
                "fn": int(cm[1, 0]),
                "tp": int(cm[1, 1]),
            },
            "error_rates_by_contract_pct": err_by_contract.round(2).to_dict(),
        },
        "test_comparison_all_models": test_df.to_dict(orient="records"),
        "top_features": top_features,
    }

    with open(os.path.join(res_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, ensure_ascii=False, indent=2)

    with open(os.path.join(mod_dir, "model_meta.json"), "w", encoding="utf-8") as f:
        json.dump({
            "champion_name": champion_name,
            "optimal_threshold": float(champion_thr),
            "test_pr_auc": metrics_payload["champion"]["test_metrics_at_opt"]["pr_auc"],
            "test_roc_auc": metrics_payload["champion"]["test_metrics_at_opt"]["roc_auc"],
            "test_f1": metrics_payload["champion"]["test_metrics_at_opt"]["f1"],
            "test_recall": metrics_payload["champion"]["test_metrics_at_opt"]["recall"],
        }, f, ensure_ascii=False, indent=2)

    print("\nTAMAMLANDI! Tüm sonuçlar kaydedildi:")
    print(f"  - Model   : {model_path}")
    print(f"  - Metrik  : {os.path.join(res_dir, 'metrics.json')}")
    print(f"  - Grafikler: {fig_dir}/fig4..fig8.png")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Telco Customer Churn — Uçtan Uca Eğitim Pipeline'ı")
    parser.add_argument("--data", type=str, default="data/WA_Fn-UseC_-Telco-Customer-Churn.csv")
    parser.add_argument("--out", type=str, default=".")
    parser.add_argument("--n_iter", type=int, default=20, help="RandomizedSearchCV deneme bütçesi (model başına)")
    args = parser.parse_args()
    main(args.data, args.out, args.n_iter)
