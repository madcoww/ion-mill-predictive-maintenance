"""Audit saved scores, calibration thresholds, and reloaded model predictions."""
from pathlib import Path
import hashlib
import json

import joblib
import numpy as np
import pandas as pd
import torch
from xgboost import XGBClassifier

import run_clone as experiment


def transform_saved(frame, features, prep):
    numeric_columns = prep["numeric_columns"]
    transformed = prep["global_scaler"].transform(features[numeric_columns])
    for tool, scaler in prep["per_tool_scalers"].items():
        positions = frame.Tool.eq(tool)
        transformed[positions] = scaler.transform(features.loc[positions, numeric_columns])
    numeric = prep["imputer"].transform(transformed).clip(-30, 30).astype("float32")
    categories = prep["encoder"].transform(features[prep["categorical_columns"]]).astype("float32")
    return np.column_stack([numeric, categories]), numeric


def predict_gru(frame, numeric, folder):
    checkpoint = torch.load(folder / "clone_gru.pt", map_location="cpu", weights_only=True)
    model = experiment.CloneGRU(checkpoint["feature_count"])
    model.load_state_dict(checkpoint["state_dict"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    features = torch.as_tensor(numeric, device=device)
    first = torch.as_tensor(np.arange(len(frame))-frame.groupby("Tool").cumcount().to_numpy(), device=device)
    offsets = torch.arange(14, -1, -1, device=device)
    scores = np.empty(len(frame), dtype="float32")
    with torch.no_grad():
        for start in range(0, len(frame), 2048):
            positions = torch.arange(start, min(start+2048, len(frame)), device=device)
            indices = torch.maximum(positions[:, None]-offsets, first[positions, None])
            scores[start:start+len(positions)] = model(features[indices]).sigmoid().cpu().numpy()
    return scores


def main():
    torch.set_num_threads(6)
    frame, spans, paths = experiment.load_data()
    manifest = json.loads((experiment.OUT / "manifest.json").read_text(encoding="utf-8"))
    for path, recorded in zip(paths, manifest["input_files"]):
        with path.open("rb") as stream:
            assert hashlib.file_digest(stream, "sha256").hexdigest() == recorded["sha256"]
    fit, calibration, test = frame.part.eq("fit"), frame.part.eq("calib"), frame.part.eq("test")
    assert (frame.time_end[fit] + experiment.DAY <= frame.Tool[fit].map(spans.fit_end)).all()
    assert (frame.time_start[test] >= frame.Tool[test].map(spans.cal_end) + experiment.DAY).all()
    assert (frame.time_end[test] + experiment.DAY <= frame.Tool[test].map(spans["last"])).all()
    # An adversarial future change must leave earlier features unchanged.
    sample = frame.iloc[:100].copy()
    original_features = experiment.reference_features(sample)
    changed = sample.copy()
    for column in experiment.NUMERIC_RAW:
        changed.loc[75:, column + "_mean"] += 10000
    changed_features = experiment.reference_features(changed)
    np.testing.assert_allclose(original_features.iloc[:75], changed_features.iloc[:75], equal_nan=True)
    features = experiment.reference_features(frame)
    checks = []
    for folder in sorted(experiment.ARTIFACTS.iterdir()):
        if not folder.is_dir():
            continue
        prep = joblib.load(folder / "preprocessing.joblib")
        training_tools = sorted(prep["per_tool_scalers"])
        evaluation_tools = sorted(frame.Tool.unique()) if folder.name == "same_tools" else sorted(set(frame.Tool)-set(training_tools))
        if folder.name != "same_tools":
            assert not set(training_tools).intersection(evaluation_tools)
        tree_values, numeric = transform_saved(frame, features, prep)
        baseline_inputs = experiment.baseline_features(frame, training_tools)
        saved_baseline = joblib.load(folder / "baseline_lr13.joblib")
        assert list(baseline_inputs) == saved_baseline["columns"]
        xgb = XGBClassifier()
        xgb.load_model(folder / "clone_xgb_smote.ubj")
        xgb.set_params(device="cpu")
        predictions = {
            "baseline_lr13": saved_baseline["model"].predict_proba(baseline_inputs)[:, 1],
            "clone_hgbm": joblib.load(folder / "clone_hgbm.joblib").predict_proba(tree_values)[:, 1],
            "clone_xgb_smote": xgb.predict_proba(tree_values)[:, 1],
            "clone_gru": predict_gru(frame, numeric, folder),
        }
        predictions["clone_hgbm_gru"] = .5*predictions["clone_hgbm"] + .5*predictions["clone_gru"]
        for name, scores in predictions.items():
            selected = frame.part.eq("test") & frame.Tool.isin(evaluation_tools)
            actual = experiment.daily_scores(frame, scores, selected)
            expected = pd.read_csv(folder / f"{name}_test_days.csv", float_precision="round_trip")
            pd.testing.assert_frame_equal(actual[["Tool", "day", "label"]], expected[["Tool", "day", "label"]], check_dtype=False)
            np.testing.assert_allclose(actual.score, expected.score, rtol=3e-5, atol=3e-6)
            recorded_cal = pd.read_csv(folder / f"{name}_calibration_days.csv", float_precision="round_trip")
            recorded_metrics = pd.read_csv(folder / "metrics.csv", float_precision="round_trip")
            # pandas writes float32's short round-trip representation to CSV.
            # Restore that dtype before threshold comparisons at exact ties.
            expected["score"] = expected.score.astype(scores.dtype)
            recorded_cal["score"] = recorded_cal.score.astype(scores.dtype)
            for lift in [2, 4]:
                threshold, _ = experiment.choose_threshold(recorded_cal, lift)
                metric = recorded_metrics.loc[recorded_metrics.model.eq(name) & recorded_metrics.lift_target.eq(lift)].iloc[0]
                if np.isinf(threshold):
                    assert np.isinf(metric.threshold)
                else:
                    assert abs(threshold-metric.threshold) < 1e-10
                # Original saved scores avoid insignificant CPU/GPU boundary differences.
                recomputed = experiment.score_metrics(expected.label, expected.score, metric.threshold)
                for key in ["tn", "fp", "fn", "tp", "days", "alarm_days"]:
                    assert recomputed[key] == metric[key], (folder.name, name, key)
            checks.append(dict(fold=folder.name, model=name, reload_predictions_match=True,
                               score_dtype=str(scores.dtype),
                               calibration_threshold_verified=True, confusion_matrix_verified=True))
        experiment.log(f"Verified reloaded artifacts: {folder.name}")
    daily = pd.read_csv(experiment.OUT / "test_day_predictions.csv")
    for setup in ["same_tools", "unseen_tools"]:
        selected = daily.loc[daily.fold.eq("same_tools")] if setup == "same_tools" else daily.loc[daily.fold.str.startswith("unseen_")]
        reference_days = None
        for name, group in selected.groupby("model"):
            keys = group[["Tool", "day", "label"]].sort_values(["Tool", "day"]).reset_index(drop=True)
            assert not keys.duplicated(["Tool", "day"]).any()
            if reference_days is not None:
                pd.testing.assert_frame_equal(keys, reference_days)
            reference_days = keys
    experiment.save_json(experiment.OUT / "verification.json", dict(
        status="passed", checks=checks, input_hashes_verified=True, split_boundaries_verified=True,
        same_evaluation_days_verified=True, future_perturbation_features_unchanged=True,
        note="Reload comparisons allow CPU/GPU floating point tolerance; no retraining or test tuning."))
    experiment.log("ALL AUDITS PASSED")


if __name__ == "__main__":
    main()
