"""Training module for groundedness classifier.

Per SPEC section 7:
- Computes features with trustrag.features.extract_features
- Compares 4 models: majority-class baseline, LogisticRegression, RandomForest, GradientBoosting
- Strict group splitting by question_id (GroupShuffleSplit 80/20 & 5-fold GroupKFold)
- Logs runs to MLflow under experiment 'trustrag'
- Saves best model to CONFIG.model_path
- Writes results/classifier.json and results/feature_importance.json
"""
from __future__ import annotations

import csv
import json
import logging
from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from trustrag.config import CONFIG
from trustrag.features import FEATURE_NAMES, extract_features
from trustrag.retrieve import get_embedding_model

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_dataset_features(
    labeled_csv: Path = CONFIG.labeled_path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Load labeled dataset, extract features, and return X, y, groups, records."""
    labeled_csv = Path(labeled_csv)
    if not labeled_csv.exists():
        raise FileNotFoundError(f"Labeled data not found: {labeled_csv}")

    records: list[dict[str, Any]] = []
    with open(labeled_csv, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if r.get("label", "").strip() != "":
                records.append(r)

    logger.info(f"Loaded {len(records)} labeled rows from {labeled_csv}")

    embed_model = get_embedding_model(CONFIG.embed_model)

    answers = [r["answer"] for r in records]
    contexts = [r["context"] for r in records]

    logger.info("Computing embeddings for answer-context similarity...")
    ans_embs = embed_model.encode(answers, batch_size=32, normalize_embeddings=True)
    ctx_embs = embed_model.encode(contexts, batch_size=32, normalize_embeddings=True)
    # Cosine similarities
    embed_sims = np.sum(ans_embs * ctx_embs, axis=1)

    X_list: list[list[float]] = []
    y_list: list[int] = []
    groups_list: list[str] = []

    for i, r in enumerate(records):
        f = extract_features(r["answer"], r["context"], embed_sim=float(embed_sims[i]))
        X_list.append(f.to_vector())
        y_list.append(int(r["label"]))
        groups_list.append(r["question_id"])

    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    groups = np.array(groups_list)

    return X, y, groups, records


def train_and_evaluate(
    labeled_csv: Path = CONFIG.labeled_path,
    model_save_path: Path = CONFIG.model_path,
    results_dir: Path = CONFIG.results_dir,
) -> dict[str, Any]:
    """Train models, evaluate on holdout and CV, save best model and metrics."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    model_save_path = Path(model_save_path)
    model_save_path.parent.mkdir(parents=True, exist_ok=True)

    X, y, groups, _ = compute_dataset_features(labeled_csv)

    # 1. GroupShuffleSplit 80/20
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=CONFIG.random_seed)
    train_idx, test_idx = next(gss.split(X, y, groups))

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    groups_train = groups[train_idx]

    # Verify no question overlap
    train_qids = set(groups[train_idx])
    test_qids = set(groups[test_idx])
    assert len(train_qids & test_qids) == 0, "Data leakage detected: train and test share question_ids!"

    logger.info(f"Split: Train {len(X_train)} rows ({len(train_qids)} questions), Test {len(X_test)} rows ({len(test_qids)} questions)")

    # Define model candidates
    models: dict[str, Any] = {
        "Baseline": DummyClassifier(strategy="most_frequent"),
        "LogisticRegression": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(random_state=CONFIG.random_seed, max_iter=1000)),
        ]),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=5,
            random_state=CONFIG.random_seed,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=3,
            random_state=CONFIG.random_seed,
        ),
    }

    # Setup MLflow
    try:
        mlflow.set_experiment("trustrag")
    except Exception as exc:
        logger.warning(f"Could not set MLflow experiment: {exc}")

    results_report: dict[str, Any] = {
        "dataset": {
            "total_rows": int(len(y)),
            "train_rows": int(len(y_train)),
            "test_rows": int(len(y_test)),
            "train_questions": len(train_qids),
            "test_questions": len(test_qids),
        },
        "models": {},
    }

    best_model_name = ""
    best_f1 = -1.0
    best_fitted_model = None

    # 5-fold GroupKFold for CV
    gkf = GroupKFold(n_splits=5)

    for name, model in models.items():
        logger.info(f"Evaluating {name}...")

        # 5-Fold Cross Validation
        cv_f1s: list[float] = []
        cv_rocs: list[float] = []

        for cv_train_idx, cv_val_idx in gkf.split(X, y, groups):
            X_cv_tr, X_cv_val = X[cv_train_idx], X[cv_val_idx]
            y_cv_tr, y_cv_val = y[cv_train_idx], y[cv_val_idx]

            m_clone = joblib.load(joblib.dump(model, Path("scratch_model.joblib"))[0]) if Path("scratch_model.joblib").exists() else None
            # Fit on fold
            model.fit(X_cv_tr, y_cv_tr)
            preds_val = model.predict(X_cv_val)

            # Score probabilities if supported
            if hasattr(model, "predict_proba"):
                probs_val = model.predict_proba(X_cv_val)[:, 1]
                try:
                    cv_rocs.append(float(roc_auc_score(y_cv_val, probs_val)))
                except ValueError:
                    pass

            cv_f1s.append(float(f1_score(y_cv_val, preds_val, zero_division=0)))

        if Path("scratch_model.joblib").exists():
            Path("scratch_model.joblib").unlink(missing_ok=True)

        # Fit on 80% train split and evaluate on 20% holdout test split
        model.fit(X_train, y_train)
        test_preds = model.predict(X_test)

        if hasattr(model, "predict_proba"):
            test_probs = model.predict_proba(X_test)[:, 1]
            try:
                test_auc = float(roc_auc_score(y_test, test_probs))
            except ValueError:
                test_auc = 0.5
        else:
            test_probs = None
            test_auc = 0.5

        prec = float(precision_score(y_test, test_preds, zero_division=0))
        rec = float(recall_score(y_test, test_preds, zero_division=0))
        f1 = float(f1_score(y_test, test_preds, zero_division=0))
        cm = confusion_matrix(y_test, test_preds).tolist()

        model_metrics = {
            "test_precision": round(prec, 4),
            "test_recall": round(rec, 4),
            "test_f1": round(f1, 4),
            "test_roc_auc": round(test_auc, 4),
            "confusion_matrix": cm,
            "cv_5fold_f1_mean": round(float(np.mean(cv_f1s)), 4),
            "cv_5fold_f1_std": round(float(np.std(cv_f1s)), 4),
            "cv_5fold_roc_mean": round(float(np.mean(cv_rocs)), 4) if cv_rocs else 0.5,
            "cv_5fold_roc_std": round(float(np.std(cv_rocs)), 4) if cv_rocs else 0.0,
        }
        results_report["models"][name] = model_metrics

        logger.info(f"{name} -> Test F1: {prec:.4f} P, {rec:.4f} R, {f1:.4f} F1, {test_auc:.4f} AUC | CV F1: {np.mean(cv_f1s):.4f}")

        # Log to MLflow
        try:
            with mlflow.start_run(run_name=name):
                mlflow.log_params({"model_type": name, "seed": CONFIG.random_seed})
                mlflow.log_metrics({
                    "test_precision": prec,
                    "test_recall": rec,
                    "test_f1": f1,
                    "test_roc_auc": test_auc,
                    "cv_f1_mean": float(np.mean(cv_f1s)),
                })
        except Exception as exc:
            logger.debug(f"MLflow logging omitted: {exc}")

        # Selection criterion: highest test F1
        if f1 > best_f1:
            best_f1 = f1
            best_model_name = name
            best_fitted_model = model

    results_report["best_model"] = best_model_name
    logger.info(f"Best model: {best_model_name} (F1 = {best_f1:.4f})")

    # Save best model
    if best_fitted_model is not None:
        joblib.dump(best_fitted_model, model_save_path)
        logger.info(f"Saved winning model to {model_save_path}")

    # Save results/classifier.json
    classifier_json_path = results_dir / "classifier.json"
    with open(classifier_json_path, "w", encoding="utf-8") as f:
        json.dump(results_report, f, indent=2)
    logger.info(f"Saved classifier metrics to {classifier_json_path}")

    # Extract feature importance for the winner
    feature_importance: dict[str, float] = {}
    if best_model_name == "LogisticRegression":
        clf_step = best_fitted_model.named_steps["clf"]
        coefs = clf_step.coef_[0]
        for fname, c in zip(FEATURE_NAMES, coefs):
            feature_importance[fname] = round(float(c), 4)
    elif hasattr(best_fitted_model, "feature_importances_"):
        importances = best_fitted_model.feature_importances_
        for fname, imp in zip(FEATURE_NAMES, importances):
            feature_importance[fname] = round(float(imp), 4)
    else:
        # Fallback to RandomForest if best was baseline
        rf = models["RandomForest"]
        rf.fit(X, y)
        for fname, imp in zip(FEATURE_NAMES, rf.feature_importances_):
            feature_importance[fname] = round(float(imp), 4)

    fi_path = results_dir / "feature_importance.json"
    with open(fi_path, "w", encoding="utf-8") as f:
        json.dump(feature_importance, f, indent=2)
    logger.info(f"Saved feature importances to {fi_path}")

    return results_report


if __name__ == "__main__":
    train_and_evaluate()
