from __future__ import annotations

import json
import os
import platform
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import friedmanchisquare, wilcoxon
from sklearn import __version__ as sklearn_version
from sklearn.base import clone
from sklearn.calibration import calibration_curve
from sklearn.datasets import load_breast_cancer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    balanced_accuracy_score,
    brier_score_loss,
    confusion_matrix,
    matthews_corrcoef,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    GridSearchCV,
    RepeatedStratifiedKFold,
    StratifiedKFold,
    cross_val_predict,
)
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

try:
    import scipy
except Exception:
    scipy = None
try:
    import pandas
except Exception:
    pandas = None
try:
    import matplotlib
except Exception:
    matplotlib = None

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RESULTS_DIR = ROOT / "results"
FIG_DIR = ROOT / "figures"
for p in (DATA_DIR, RESULTS_DIR, FIG_DIR):
    p.mkdir(parents=True, exist_ok=True)

SEED = 513
N_SPLITS = 5
N_REPEATS = 3
INNER_SPLITS = 3
TARGET_SPECIFICITY = 0.95

FEATURE_NAMES_UCI = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave_points_mean", "symmetry_mean", "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave_points_se", "symmetry_se", "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave_points_worst", "symmetry_worst", "fractal_dimension_worst",
]


def load_data():
    """Load the official UCI WDBC raw file when network access is available.

    The raw file has 32 columns: ID, diagnosis, and 30 numeric predictors.
    The ID is explicitly removed before modelling. If the UCI host is not
    reachable (for example, in an offline grading sandbox), the function
    falls back to scikit-learn's packaged copy of the same 569 x 30 WDBC
    predictor matrix and diagnosis labels.
    """
    raw_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/wdbc.data"
    columns = ["id", "diagnosis"] + FEATURE_NAMES_UCI
    try:
        df = pd.read_csv(raw_url, header=None, names=columns)
        if df.shape != (569, 32):
            raise ValueError(f"Unexpected UCI WDBC shape: {df.shape}")
        df.to_csv(DATA_DIR / "wdbc.data", header=False, index=False)
        X = df[FEATURE_NAMES_UCI].astype(float).copy()
        y = (df["diagnosis"].astype(str).str.upper() == "M").astype(int).rename("malignant")
        source = {
            "source_used": "official UCI wdbc.data",
            "url": raw_url,
            "id_unique": bool(df["id"].is_unique),
            "raw_rows": int(df.shape[0]),
            "raw_columns": int(df.shape[1]),
        }
    except Exception as exc:
        bunch = load_breast_cancer(as_frame=True)
        X = bunch.data.copy()
        X.columns = FEATURE_NAMES_UCI
        y = (bunch.target == 0).astype(int).rename("malignant")
        source = {
            "source_used": "scikit-learn packaged WDBC fallback",
            "intended_official_url": raw_url,
            "fallback_reason": str(exc),
            "note": "scikit-learn contains the same 569 x 30 WDBC predictor matrix; the raw ID column is already absent",
        }
    (RESULTS_DIR / "data_source.json").write_text(json.dumps(source, indent=2))
    return X, y


def audit_data(X: pd.DataFrame, y: pd.Series):
    corr = X.corr(numeric_only=True).abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    high_pairs = (
        upper.stack()
        .reset_index()
        .rename(columns={"level_0": "feature_1", "level_1": "feature_2", 0: "abs_r"})
        .query("abs_r >= 0.90")
        .sort_values("abs_r", ascending=False)
    )
    audit = {
        "rows": int(X.shape[0]),
        "features": int(X.shape[1]),
        "missing_values": int(X.isna().sum().sum()),
        "duplicate_feature_rows": int(X.duplicated().sum()),
        "malignant": int(y.sum()),
        "benign": int((1-y).sum()),
        "malignant_percent": float(100*y.mean()),
        "high_corr_pairs_abs_r_ge_0.90": int(len(high_pairs)),
        "max_abs_correlation": float(high_pairs["abs_r"].max()) if len(high_pairs) else None,
    }
    (RESULTS_DIR / "data_audit.json").write_text(json.dumps(audit, indent=2))
    high_pairs.to_csv(RESULTS_DIR / "high_correlation_pairs.csv", index=False)
    return audit, high_pairs


def build_models(scale=True):
    scaler = StandardScaler() if scale else "passthrough"
    models = {
        "Logistic Regression": (
            Pipeline([
                ("scale", scaler),
                ("model", LogisticRegression(solver="lbfgs", max_iter=5000, random_state=SEED)),
            ]),
            {
                "model__C": [0.01, 0.1, 1, 10, 100, 300],
            },
        ),
        "k-NN": (
            Pipeline([
                ("scale", scaler),
                ("model", KNeighborsClassifier()),
            ]),
            {
                "model__n_neighbors": [3, 7, 15],
                "model__weights": ["uniform", "distance"],
            },
        ),
        "Random Forest": (
            Pipeline([
                ("scale", scaler),
                ("model", RandomForestClassifier(random_state=SEED, n_jobs=1)),
            ]),
            {
                "model__n_estimators": [30, 80],
                "model__max_depth": [None, 8, 16],
                "model__min_samples_leaf": [1],
            },
        ),
        "RBF SVM": (
            Pipeline([
                ("scale", scaler),
                ("model", SVC(kernel="rbf", probability=True, random_state=SEED)),
            ]),
            {
                "model__C": [0.1, 1, 10],
                "model__gamma": ["scale", 0.01],
            },
        ),
    }
    return models


def threshold_for_specificity(y_true, scores, target_spec=0.95):
    fpr, tpr, thresholds = roc_curve(y_true, scores, pos_label=1)
    specificity = 1 - fpr
    valid = np.where(specificity >= target_spec)[0]
    if len(valid) == 0:
        return float(np.inf)
    # Among thresholds satisfying target specificity, maximize sensitivity.
    max_tpr = np.max(tpr[valid])
    candidates = valid[np.isclose(tpr[valid], max_tpr)]
    # If tied, use the lowest threshold to be least conservative while preserving specificity.
    idx = candidates[np.argmin(thresholds[candidates])]
    return float(thresholds[idx])


def evaluate_threshold(y_true, scores, threshold):
    pred = (scores >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred, labels=[0, 1]).ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else np.nan
    specificity = tn / (tn + fp) if (tn + fp) else np.nan
    return sensitivity, specificity, pred


def run_nested_cv(X, y, models, label="main"):
    outer = RepeatedStratifiedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    outer_splits = list(outer.split(X, y))
    rows = []
    pred_rows = []
    best_rows = []

    for alg_idx, (alg_name, (pipeline, grid)) in enumerate(models.items()):
        print(f"\n=== {label}: {alg_name} ===", flush=True)
        for fold_idx, (train_idx, test_idx) in enumerate(outer_splits, start=1):
            repeat = (fold_idx - 1) // N_SPLITS + 1
            fold = (fold_idx - 1) % N_SPLITS + 1
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

            inner_seed = SEED + fold_idx
            inner = StratifiedKFold(n_splits=INNER_SPLITS, shuffle=True, random_state=inner_seed)
            search = GridSearchCV(
                estimator=clone(pipeline),
                param_grid=grid,
                scoring="roc_auc",
                cv=inner,
                n_jobs=1,
                refit=True,
                return_train_score=False,
            )

            t0 = time.perf_counter()
            search.fit(X_train, y_train)
            train_time = time.perf_counter() - t0
            best = search.best_estimator_

            t1 = time.perf_counter()
            prob = best.predict_proba(X_test)[:, 1]
            pred_time = time.perf_counter() - t1
            pred = (prob >= 0.5).astype(int)

            # Evaluation operating point: derive the ROC threshold on the untouched outer
            # fold only to estimate sensitivity at 95% specificity. This threshold is
            # descriptive and is NOT used as the deployment threshold. A separate
            # training-only OOF procedure below derives the deployment candidate.
            threshold = threshold_for_specificity(y_test.to_numpy(), prob, TARGET_SPECIFICITY)
            sens95, achieved_spec, pred95 = evaluate_threshold(y_test.to_numpy(), prob, threshold)

            bal_acc = balanced_accuracy_score(y_test, pred)
            roc_auc = roc_auc_score(y_test, prob)
            mcc = matthews_corrcoef(y_test, pred)
            brier = brier_score_loss(y_test, prob)

            rows.append({
                "experiment": label,
                "algorithm": alg_name,
                "repeat": repeat,
                "fold": fold,
                "outer_fold_index": fold_idx,
                "n_train": len(train_idx),
                "n_test": len(test_idx),
                "balanced_accuracy": bal_acc,
                "roc_auc": roc_auc,
                "sensitivity_at_95_specificity": sens95,
                "achieved_specificity": achieved_spec,
                "mcc": mcc,
                "brier_score": brier,
                "selected_threshold": threshold,
                "train_time_s": train_time,
                "prediction_time_s": pred_time,
                "prediction_time_ms_per_sample": pred_time / len(test_idx) * 1000,
                "best_inner_roc_auc": search.best_score_,
            })
            best_rows.append({
                "experiment": label,
                "algorithm": alg_name,
                "repeat": repeat,
                "fold": fold,
                "outer_fold_index": fold_idx,
                "best_params": json.dumps(search.best_params_, sort_keys=True),
                "best_inner_roc_auc": search.best_score_,
            })
            for local_i, global_i in enumerate(test_idx):
                pred_rows.append({
                    "experiment": label,
                    "algorithm": alg_name,
                    "repeat": repeat,
                    "fold": fold,
                    "row_index": int(global_i),
                    "y_true": int(y_test.iloc[local_i]),
                    "prob_malignant": float(prob[local_i]),
                    "pred_default_0_5": int(pred[local_i]),
                    "selected_threshold": threshold,
                    "pred_at_95_specificity": int(pred95[local_i]),
                })
            print(
                f"fold {fold_idx:02d}/15 BA={bal_acc:.4f} AUC={roc_auc:.4f} "
                f"Sens@95={sens95:.4f} MCC={mcc:.4f} time={train_time:.2f}s",
                flush=True,
            )

    results = pd.DataFrame(rows)
    predictions = pd.DataFrame(pred_rows)
    best_params = pd.DataFrame(best_rows)
    return results, predictions, best_params


def holm_adjust(pvals):
    pvals = np.asarray(pvals, dtype=float)
    m = len(pvals)
    order = np.argsort(pvals)
    adjusted = np.empty(m, dtype=float)
    running = 0.0
    for rank, idx in enumerate(order):
        val = (m - rank) * pvals[idx]
        running = max(running, val)
        adjusted[idx] = min(running, 1.0)
    return adjusted


def rank_biserial_from_pairs(x, y):
    d = np.asarray(x) - np.asarray(y)
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    ranks = pd.Series(np.abs(d)).rank(method="average").to_numpy()
    w_plus = ranks[d > 0].sum()
    w_minus = ranks[d < 0].sum()
    return float((w_plus - w_minus) / (w_plus + w_minus))


def statistical_tests(results: pd.DataFrame, metric="balanced_accuracy"):
    pivot = results.pivot(index="outer_fold_index", columns="algorithm", values=metric)
    algs = list(pivot.columns)
    stat, p = friedmanchisquare(*[pivot[a].to_numpy() for a in algs])
    omnibus = pd.DataFrame([{
        "metric": metric,
        "test": "Friedman",
        "statistic": stat,
        "p_value": p,
        "n_paired_folds": len(pivot),
        "algorithms": "; ".join(algs),
    }])

    pair_rows = []
    raw_p = []
    for i in range(len(algs)):
        for j in range(i+1, len(algs)):
            a, b = algs[i], algs[j]
            xa, xb = pivot[a].to_numpy(), pivot[b].to_numpy()
            try:
                w, pv = wilcoxon(xa, xb, alternative="two-sided", zero_method="wilcox")
            except ValueError:
                w, pv = 0.0, 1.0
            rb = rank_biserial_from_pairs(xa, xb)
            pair_rows.append({
                "metric": metric,
                "algorithm_a": a,
                "algorithm_b": b,
                "mean_a": xa.mean(),
                "mean_b": xb.mean(),
                "mean_difference_a_minus_b": (xa-xb).mean(),
                "wilcoxon_W": w,
                "p_raw": pv,
                "rank_biserial_effect": rb,
            })
            raw_p.append(pv)
    adj = holm_adjust(raw_p)
    for row, pa in zip(pair_rows, adj):
        row["p_holm"] = pa
        row["significant_0_05"] = bool(pa < 0.05)
    pairwise = pd.DataFrame(pair_rows).sort_values("p_holm")
    return omnibus, pairwise


def summarize_results(results: pd.DataFrame):
    metrics = [
        "balanced_accuracy", "roc_auc", "sensitivity_at_95_specificity", "mcc",
        "achieved_specificity", "brier_score", "selected_threshold", "train_time_s",
        "prediction_time_ms_per_sample",
    ]
    agg = results.groupby("algorithm")[metrics].agg(["mean", "std"])
    # flatten
    agg.columns = [f"{c}_{s}" for c, s in agg.columns]
    agg = agg.reset_index()
    # rank by balanced accuracy (higher is better)
    agg["rank_balanced_accuracy"] = agg["balanced_accuracy_mean"].rank(ascending=False, method="min").astype(int)
    agg = agg.sort_values("rank_balanced_accuracy")
    return agg


def plot_dispersion(results):
    metrics = [
        ("balanced_accuracy", "Balanced accuracy"),
        ("roc_auc", "ROC-AUC"),
        ("sensitivity_at_95_specificity", "Sensitivity at 95% specificity"),
        ("mcc", "Matthews correlation coefficient"),
    ]
    alg_order = (
        results.groupby("algorithm")["balanced_accuracy"].mean().sort_values(ascending=False).index.tolist()
    )
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.5), constrained_layout=True)
    for ax, (metric, title) in zip(axes.flat, metrics):
        data = [results.loc[results.algorithm == a, metric].to_numpy() for a in alg_order]
        ax.boxplot(data, tick_labels=alg_order, showmeans=True)
        ax.set_title(title)
        ax.set_ylabel("Score")
        ax.tick_params(axis="x", labelrotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.suptitle("Fold-level performance across repeated stratified 5-fold CV (n=15 per algorithm)")
    fig.savefig(FIG_DIR / "figure1_fold_level_dispersion.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure1_fold_level_dispersion.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_confusion(predictions, top_algs):
    fig, axes = plt.subplots(1, 2, figsize=(8.5, 3.8), constrained_layout=True)
    for ax, alg in zip(axes, top_algs):
        d = predictions[predictions.algorithm == alg]
        cm = confusion_matrix(d.y_true, d.pred_default_0_5, labels=[0, 1])
        im = ax.imshow(cm, cmap="Blues")
        for (i, j), v in np.ndenumerate(cm):
            ax.text(j, i, f"{v}", ha="center", va="center", fontsize=12)
        ax.set_xticks([0, 1], ["Benign", "Malignant"])
        ax.set_yticks([0, 1], ["Benign", "Malignant"])
        ax.set_xlabel("Predicted")
        ax.set_ylabel("True")
        ax.set_title(alg)
    fig.suptitle("Pooled out-of-fold confusion matrices at the default 0.5 threshold (1,707 predictions/model)")
    fig.savefig(FIG_DIR / "figure2_confusion_matrices_top2.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure2_confusion_matrices_top2.pdf", bbox_inches="tight")
    plt.close(fig)


def plot_calibration(predictions, top_algs):
    fig, ax = plt.subplots(figsize=(6.2, 5.2), constrained_layout=True)
    for alg in top_algs:
        d = predictions[predictions.algorithm == alg]
        frac_pos, mean_pred = calibration_curve(d.y_true, d.prob_malignant, n_bins=10, strategy="quantile")
        ax.plot(mean_pred, frac_pos, marker="o", label=alg)
    ax.plot([0, 1], [0, 1], linestyle="--", label="Perfect calibration")
    ax.set_xlabel("Mean predicted probability of malignancy")
    ax.set_ylabel("Observed malignant fraction")
    ax.set_title("Calibration of the two highest-ranked algorithms")
    ax.legend()
    ax.grid(alpha=0.25)
    fig.savefig(FIG_DIR / "figure3_calibration_top2.png", dpi=220, bbox_inches="tight")
    fig.savefig(FIG_DIR / "figure3_calibration_top2.pdf", bbox_inches="tight")
    plt.close(fig)


def final_thresholds(X, y, top_algs, models):
    rows = []
    for i, alg in enumerate(top_algs):
        pipeline, grid = models[alg]
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 700 + i)
        search = GridSearchCV(clone(pipeline), grid, scoring="roc_auc", cv=cv, n_jobs=1, refit=True)
        search.fit(X, y)
        best = search.best_estimator_
        oof_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED + 900 + i)
        prob = cross_val_predict(clone(best), X, y, cv=oof_cv, method="predict_proba", n_jobs=1)[:, 1]
        thr = threshold_for_specificity(y.to_numpy(), prob, TARGET_SPECIFICITY)
        sens, spec, _ = evaluate_threshold(y.to_numpy(), prob, thr)
        rows.append({
            "algorithm": alg,
            "proposed_deployment_threshold": thr,
            "full_data_oof_sensitivity": sens,
            "full_data_oof_specificity": spec,
            "full_data_oof_brier": brier_score_loss(y, prob),
            "best_params_full_data": json.dumps(search.best_params_, sort_keys=True),
            "best_inner_roc_auc_full_data": search.best_score_,
        })
    return pd.DataFrame(rows)


def plot_scaling_ablation(main_results, ablation_results):
    rows = []
    for alg in ["k-NN", "RBF SVM"]:
        scaled = main_results[main_results.algorithm == alg]
        unscaled = ablation_results[ablation_results.algorithm == alg]
        rows.append({
            "algorithm": alg,
            "scaled_balanced_accuracy_mean": scaled.balanced_accuracy.mean(),
            "scaled_balanced_accuracy_std": scaled.balanced_accuracy.std(ddof=1),
            "unscaled_balanced_accuracy_mean": unscaled.balanced_accuracy.mean(),
            "unscaled_balanced_accuracy_std": unscaled.balanced_accuracy.std(ddof=1),
            "delta_scaled_minus_unscaled": scaled.balanced_accuracy.mean()-unscaled.balanced_accuracy.mean(),
            "scaled_roc_auc_mean": scaled.roc_auc.mean(),
            "unscaled_roc_auc_mean": unscaled.roc_auc.mean(),
        })
    table = pd.DataFrame(rows)
    table.to_csv(RESULTS_DIR / "scaling_ablation_summary.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.2), constrained_layout=True)
    x = np.arange(len(table))
    w = 0.34
    ax.bar(x-w/2, table.scaled_balanced_accuracy_mean, width=w, yerr=table.scaled_balanced_accuracy_std, capsize=4, label="Scaled inside fold")
    ax.bar(x+w/2, table.unscaled_balanced_accuracy_mean, width=w, yerr=table.unscaled_balanced_accuracy_std, capsize=4, label="No scaling")
    ax.set_xticks(x, table.algorithm)
    ax.set_ylim(0.5, 1.0)
    ax.set_ylabel("Balanced accuracy")
    ax.set_title("Scaling ablation for distance/kernel methods (mean ± SD, n=15)")
    ax.legend()
    ax.grid(axis="y", alpha=0.25)
    fig.savefig(FIG_DIR / "figure4_scaling_ablation.png", dpi=220, bbox_inches="tight")
    plt.close(fig)
    return table


def environment_info():
    info = {
        "python": sys.version,
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "scikit_learn": sklearn_version,
        "numpy": np.__version__,
        "pandas": pandas.__version__ if pandas else None,
        "scipy": scipy.__version__ if scipy else None,
        "matplotlib": matplotlib.__version__ if matplotlib else None,
    }
    # best-effort CPU model and memory on Linux
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.lower().startswith("model name"):
                    info["cpu_model"] = line.split(":", 1)[1].strip()
                    break
    except Exception:
        pass
    try:
        import psutil
        info["ram_gb"] = round(psutil.virtual_memory().total / 1024**3, 2)
    except Exception:
        pass
    (RESULTS_DIR / "environment.json").write_text(json.dumps(info, indent=2))
    return info


def main():
    np.random.seed(SEED)
    X, y = load_data()
    audit, high_pairs = audit_data(X, y)
    env = environment_info()
    print("Audit:", audit)
    print("Environment:", env)

    models = build_models(scale=True)
    main_results, main_preds, best_params = run_nested_cv(X, y, models, label="main_scaled")
    main_results.to_csv(RESULTS_DIR / "fold_results_main.csv", index=False)
    main_preds.to_csv(RESULTS_DIR / "oof_predictions_main.csv", index=False)
    best_params.to_csv(RESULTS_DIR / "best_hyperparameters_by_fold.csv", index=False)

    summary = summarize_results(main_results)
    summary.to_csv(RESULTS_DIR / "main_results_summary.csv", index=False)
    omnibus, pairwise = statistical_tests(main_results, "balanced_accuracy")
    omnibus.to_csv(RESULTS_DIR / "friedman_balanced_accuracy.csv", index=False)
    pairwise.to_csv(RESULTS_DIR / "wilcoxon_holm_balanced_accuracy.csv", index=False)

    top_algs = summary.sort_values("rank_balanced_accuracy").algorithm.head(2).tolist()
    print("Top algorithms:", top_algs)
    thresholds = final_thresholds(X, y, top_algs, models)
    thresholds.to_csv(RESULTS_DIR / "deployment_thresholds_top2.csv", index=False)

    plot_dispersion(main_results)
    plot_confusion(main_preds, top_algs)
    plot_calibration(main_preds, top_algs)

    # Secondary ablation: remove scaling only for the two algorithms most dependent on it.
    unscaled_all = build_models(scale=False)
    unscaled_subset = {k: unscaled_all[k] for k in ["k-NN", "RBF SVM"]}
    ablation_results, ablation_preds, ablation_params = run_nested_cv(X, y, unscaled_subset, label="no_scaling_ablation")
    ablation_results.to_csv(RESULTS_DIR / "fold_results_no_scaling_ablation.csv", index=False)
    ablation_params.to_csv(RESULTS_DIR / "best_hyperparameters_no_scaling_ablation.csv", index=False)
    ablation_table = plot_scaling_ablation(main_results, ablation_results)

    # Dataset summary compact CSV for report generation.
    ds = pd.DataFrame([{
        "source": "UCI Machine Learning Repository, Breast Cancer Wisconsin (Diagnostic), dataset 17",
        "doi": "10.24432/C5DW2B",
        "license": "CC BY 4.0",
        "file": "wdbc.data",
        "rows": audit["rows"],
        "predictors": audit["features"],
        "types": "30 real-valued numeric features",
        "target": "Diagnosis: malignant (positive) vs benign",
        "class_distribution": f"Malignant {audit['malignant']} ({audit['malignant_percent']:.1f}%); Benign {audit['benign']} ({100-audit['malignant_percent']:.1f}%)",
        "missing": audit["missing_values"],
        "duplicate_feature_rows": audit["duplicate_feature_rows"],
        "identifier_handling": "ID column excluded before modelling",
    }])
    ds.to_csv(RESULTS_DIR / "dataset_summary.csv", index=False)

    print("\nMAIN SUMMARY\n", summary.to_string(index=False))
    print("\nFRIEDMAN\n", omnibus.to_string(index=False))
    print("\nPAIRWISE\n", pairwise.to_string(index=False))
    print("\nTHRESHOLDS\n", thresholds.to_string(index=False))
    print("\nABLATION\n", ablation_table.to_string(index=False))

if __name__ == "__main__":
    main()
