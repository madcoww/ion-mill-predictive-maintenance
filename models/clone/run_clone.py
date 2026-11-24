"""GitHub clone adaptation: PHM2018 24-hour pressure-drop warning.

Reference: ki-seoultech/PHM2018-Ion-Mill-Etch-3Stage-Predictive-Maintenance.
This is an adaptation, not a replication of its conditional RUL benchmark.
Run from any directory: python -X utf8 models/clone/run_clone.py
"""
from __future__ import annotations

import argparse
import ast
import hashlib
import json
import os
import platform
import random
import subprocess
import time
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, average_precision_score, confusion_matrix,
                             f1_score, precision_recall_curve, precision_score,
                             recall_score, roc_auc_score)
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from torch import nn
from xgboost import XGBClassifier

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
UPSTREAM = HERE / "upstream"
REFERENCE = HERE / "reference"
OUT = HERE / "results"
ARTIFACTS = HERE / "artifacts"
DAY = 86400
SEED = 42
DROP_FAULT = "FlowCool Pressure Dropped Below Limit"
CORE = ["FLOWCOOLPRESSURE", "IONGAUGEPRESSURE", "ETCHBEAMCURRENT", "FLOWCOOLFLOWRATE"]
NUMERIC_RAW = ["IONGAUGEPRESSURE", "ETCHBEAMVOLTAGE", "ETCHBEAMCURRENT",
               "ETCHSUPPRESSORVOLTAGE", "ETCHSUPPRESSORCURRENT", "FLOWCOOLFLOWRATE",
               "FLOWCOOLPRESSURE", "ETCHGASCHANNEL1READBACK", "ETCHPBNGASREADBACK",
               "FIXTURETILTANGLE", "ACTUALROTATIONANGLE"]


def log(message):
    print(time.strftime("%H:%M:%S"), message, flush=True)


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def load_data():
    paths = sorted((ROOT / "data/processed/step_s1").glob("*.parquet"))
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame = frame.sort_values(["Tool", "time_end", "time_start"], kind="stable").reset_index(drop=True)
    assert frame.shutter.eq(1).all()
    faults = pd.read_csv(ROOT / "data/processed/faults.csv")
    faults["Tool"] = faults.Tool.str[2:] + "_" + faults.Tool.str[:2]
    faults["fault_type"] = faults.fault_name.str.replace(" [NoWaferID]", "", regex=False)
    faults = faults.drop_duplicates(["Tool", "time", "fault_type"]).sort_values(["Tool", "time"])
    spans = frame.groupby("Tool").agg(first=("time_start", "min"), last=("time_end", "max"))
    spans["fit_end"] = spans["first"] + .60 * (spans["last"] - spans["first"])
    spans["cal_end"] = spans["first"] + .75 * (spans["last"] - spans["first"])
    spans["inner_end"] = spans["first"] + .50 * (spans["last"] - spans["first"])
    frame["part"] = np.select([
        frame.time_end + DAY <= frame.Tool.map(spans.fit_end),
        (frame.time_start >= frame.Tool.map(spans.fit_end)) & (frame.time_end <= frame.Tool.map(spans.cal_end)),
        (frame.time_start >= frame.Tool.map(spans.cal_end) + DAY) & (frame.time_end + DAY <= frame.Tool.map(spans["last"]))
    ], ["fit", "calib", "test"], default="gap")
    frame["inner"] = np.select([
        frame.time_end + DAY <= frame.Tool.map(spans.inner_end),
        (frame.time_start >= frame.Tool.map(spans.inner_end)) & frame.part.eq("fit")
    ], ["train", "valid"], default="gap")
    frame["label"] = 0
    frame["normal"] = True
    frame["since_any"] = np.log1p(8760.)
    frame["since_drop"] = np.log1p(8760.)
    frame["n7d_any"] = 0
    for tool, positions in frame.groupby("Tool", sort=False).indices.items():
        ends = frame.time_end.to_numpy()[positions]
        starts = frame.time_start.to_numpy()[positions]
        tool_faults = faults.loc[faults.Tool == tool]
        drop_times = tool_faults.loc[tool_faults.fault_type == DROP_FAULT, "time"].to_numpy()
        all_times = tool_faults.time.to_numpy()
        if len(drop_times):
            next_positions = np.searchsorted(drop_times, ends, side="right")
            next_times = np.where(next_positions < len(drop_times), drop_times[np.minimum(next_positions, len(drop_times)-1)], np.inf)
            frame.loc[positions, "label"] = ((next_times - ends) <= DAY).astype(int)
        if len(all_times):
            previous_positions = np.searchsorted(all_times, starts, side="left")
            previous_times = np.where(previous_positions > 0, all_times[np.maximum(previous_positions-1, 0)], -np.inf)
            next_positions = np.searchsorted(all_times, ends, side="right")
            next_times = np.where(next_positions < len(all_times), all_times[np.minimum(next_positions, len(all_times)-1)], np.inf)
            # Labels of calibration/test faults never define a training normal row.
            normal = (starts - previous_times > DAY) & (next_times - ends > DAY)
            inside = np.searchsorted(all_times, ends, side="right") > np.searchsorted(all_times, starts, side="left")
            frame.loc[positions, "normal"] = normal & ~inside
        for column, event_times in [("since_any", all_times), ("since_drop", drop_times)]:
            if not len(event_times):
                continue
            past_positions = np.searchsorted(event_times, ends, side="left")
            past_times = np.where(past_positions > 0, event_times[np.maximum(past_positions-1, 0)], -np.inf)
            hours = np.minimum((ends-past_times)/3600, 8760.)
            frame.loc[positions, column] = np.log1p(hours)
        frame.loc[positions, "n7d_any"] = np.searchsorted(all_times, ends, side="left") - np.searchsorted(all_times, ends - 7*DAY, side="left")
    # Every value and alert is available at segment END, never at segment start.
    frame["day"] = np.floor((frame.time_end-frame.Tool.map(spans["first"]))/DAY).astype(int)
    log(f"Loaded {len(frame):,} segments / {len(spans)} tools; labels from next fault after segment end")
    return frame, spans, paths


def reference_features(frame):
    """Execute only the upstream pure feature function; no training/Drive side effects."""
    source = REFERENCE / "v8_multi_fault_sweep_f1_f2.py"
    tree = ast.parse(source.read_text(encoding="utf-8-sig"))
    function = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "add_reference_features")
    namespace = {"np": np, "pd": pd}
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    inputs = frame[["Tool", "time_end", "stage", "recipe", "recipe_step"]].copy()
    for sensor in NUMERIC_RAW:
        inputs[sensor] = frame[sensor + "_mean"].to_numpy()
    # A fault/TTF-derived episode boundary is unknown in a live warning stream.
    inputs["episode_id"] = pd.factorize(frame.Tool, sort=False)[0]
    engineered = namespace["add_reference_features"](inputs, CORE, w=10)
    engineered = engineered.drop(columns=["Tool", "time_end", "episode_id"])
    engineered["log_gap_hours"] = np.log1p(frame.groupby("Tool").time_end.diff().fillna(0).clip(lower=0)/3600)
    assert not any("TTF" in column or "label" in column for column in engineered)
    return engineered.replace([np.inf, -np.inf], np.nan).astype("float32")


def baseline_features(frame, training_tools):
    normal_rows = frame.part.eq("fit") & frame.normal & frame.Tool.isin(training_tools)
    context = frame.recipe.astype(str) + "|" + frame.recipe_step.fillna(-1).astype(str)
    own_context = frame.Tool + "|" + context
    values = pd.DataFrame(index=frame.index)
    for sensor, prefix in [("FLOWCOOLPRESSURE", "p"), ("FLOWCOOLFLOWRATE", "f")]:
        normal = pd.DataFrame({"value": frame[sensor + "_mean"], "own": own_context, "pooled": context}).loc[normal_rows]
        own = normal.groupby("own").value.agg(["mean", "std", "size"])
        pooled = normal.groupby("pooled").value.agg(["mean", "std", "size"])
        own = own.loc[(own["size"] >= 10) & (own["std"] > 1e-6)]
        pooled = pooled.loc[(pooled["size"] >= 10) & (pooled["std"] > 1e-6)]
        baseline_mean = own_context.map(own["mean"])
        baseline_std = own_context.map(own["std"])
        unseen = ~frame.Tool.isin(training_tools)
        baseline_mean.loc[unseen] = context.loc[unseen].map(pooled["mean"])
        baseline_std.loc[unseen] = context.loc[unseen].map(pooled["std"])
        values[prefix + "_z"] = ((frame[sensor + "_mean"] - baseline_mean)/baseline_std).clip(-10, 10)
        for tool, positions in frame.groupby("Tool", sort=False).indices.items():
            series = pd.Series(values[prefix + "_z"].to_numpy()[positions], index=pd.to_datetime(frame.time_end.to_numpy()[positions], unit="s"))
            for window in ["6h", "24h", "7d"]:
                values.loc[positions, prefix + "_r" + window] = series.rolling(window.replace("d", "D"), min_periods=1).mean().to_numpy()
        values[prefix + "_chg24"] = values[prefix + "_r24h"] - values[prefix + "_r7d"]
    for column in ["since_any", "since_drop", "n7d_any"]:
        values[column] = frame[column]
    assert values.shape[1] == 13
    return values.fillna(0).astype("float32")


def prepare_reference(frame, features, training_tools):
    fit_mask = frame.part.eq("fit") & frame.Tool.isin(training_tools)
    numeric_columns = [column for column in features if column not in ["stage", "recipe", "recipe_step"]]
    categorical_columns = ["stage", "recipe", "recipe_step"]
    scaler = StandardScaler().fit(features.loc[fit_mask, numeric_columns])
    transformed = scaler.transform(features[numeric_columns])
    per_tool = {}
    for tool in training_tools:
        own_mask = fit_mask & frame.Tool.eq(tool)
        if not own_mask.any():
            continue
        own_scaler = StandardScaler().fit(features.loc[own_mask, numeric_columns])
        rows = frame.Tool.eq(tool)
        transformed[rows] = own_scaler.transform(features.loc[rows, numeric_columns])
        per_tool[tool] = own_scaler
    imputer = SimpleImputer(strategy="median", keep_empty_features=True).fit(transformed[fit_mask])
    numeric_values = imputer.transform(transformed).clip(-30, 30).astype("float32")
    encoder = OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1, encoded_missing_value=-1).fit(features.loc[fit_mask, categorical_columns])
    categorical_values = encoder.transform(features[categorical_columns]).astype("float32")
    tree_values = np.column_stack([numeric_values, categorical_values]).astype("float32")
    preprocessing = dict(global_scaler=scaler, per_tool_scalers=per_tool, imputer=imputer,
                         encoder=encoder, numeric_columns=numeric_columns, categorical_columns=categorical_columns)
    return tree_values, numeric_values, preprocessing


def daily_scores(frame, scores, selected):
    chosen = frame.loc[selected, ["Tool", "day", "label"]].copy()
    chosen["score"] = scores[np.asarray(selected)]
    return chosen.groupby(["Tool", "day"], sort=True).agg(label=("label", "max"), score=("score", "max")).reset_index()


def choose_threshold(daily, lift):
    target = float(daily.label.mean() * lift)
    if not daily.label.any() or target > 1:
        return float("inf"), target
    precision, recall, thresholds = precision_recall_curve(daily.label, daily.score)
    ordered = np.sort(daily.score.to_numpy())
    alarm_counts = len(ordered)-np.searchsorted(ordered, thresholds, side="left")
    valid = np.flatnonzero((precision[:-1] >= target) & (alarm_counts >= 10))
    if not len(valid):
        return float("inf"), target
    # Maximize calibration recall subject to precision target, without test labels.
    winner = valid[np.argmax(recall[valid])]
    return float(thresholds[winner]), target


def score_metrics(labels, scores, threshold):
    labels, scores = np.asarray(labels), np.asarray(scores)
    predicted = scores >= threshold
    matrix = confusion_matrix(labels, predicted, labels=[0, 1])
    return dict(precision=float(precision_score(labels, predicted, zero_division=0)),
                recall=float(recall_score(labels, predicted, zero_division=0)),
                f1=float(f1_score(labels, predicted, zero_division=0)),
                accuracy=float(accuracy_score(labels, predicted)),
                auc=float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else None,
                pr_auc=float(average_precision_score(labels, scores)),
                days=len(labels), positive_days=int(labels.sum()), alarm_days=int(predicted.sum()),
                tn=int(matrix[0, 0]), fp=int(matrix[0, 1]), fn=int(matrix[1, 0]), tp=int(matrix[1, 1]))


class CloneGRU(nn.Module):
    """PyTorch port of upstream GRU(64)->GRU(32)->Dropout(.2)->Dense(16)."""
    def __init__(self, feature_count):
        super().__init__()
        self.first = nn.GRU(feature_count, 64, batch_first=True)
        self.second = nn.GRU(64, 32, batch_first=True)
        self.head = nn.Sequential(nn.Dropout(.2), nn.Linear(32, 16), nn.ReLU(), nn.Linear(16, 1))

    def forward(self, sequence):
        first_output, _ = self.first(sequence)
        second_output, _ = self.second(first_output)
        return self.head(second_output[:, -1]).squeeze(-1)


def train_gru(frame, numeric, training_tools, folder, epochs):
    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.set_num_threads(6)
    values = torch.as_tensor(numeric, device=device)
    labels = torch.as_tensor(frame.label.to_numpy(dtype="float32"), device=device)
    starts = frame.groupby("Tool", sort=False).cumcount().to_numpy()
    first_positions = np.arange(len(frame))-starts
    # Left padding lets every test row be scored; no cross-tool windows.
    offsets = torch.arange(14, -1, -1, device=device)
    minimum = torch.as_tensor(first_positions, device=device)

    def windows(positions):
        positions = torch.as_tensor(positions, dtype=torch.long, device=device)
        sequence_indices = torch.maximum(positions[:, None]-offsets, minimum[positions, None])
        return values[sequence_indices]

    eligible = frame.Tool.isin(training_tools) & frame.part.eq("fit")
    train_positions = np.flatnonzero(eligible & frame.inner.eq("train"))
    valid_positions = np.flatnonzero(eligible & frame.inner.eq("valid"))
    assert len(train_positions) and len(valid_positions)
    model = CloneGRU(numeric.shape[1]).to(device)
    positive_weight = float((1-frame.label.iloc[train_positions]).sum()/max(frame.label.iloc[train_positions].sum(), 1))
    loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(positive_weight, device=device))
    optimizer = torch.optim.Adam(model.parameters(), lr=.001)
    best_loss, bad_epochs, best_state = float("inf"), 0, None
    history = []
    generator = np.random.default_rng(SEED)
    for epoch in range(epochs):
        model.train()
        running_loss = 0.
        order = generator.permutation(train_positions)
        for start in range(0, len(order), 512):
            positions = order[start:start+512]
            optimizer.zero_grad(set_to_none=True)
            loss = loss_fn(model(windows(positions)), labels[positions])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.)
            optimizer.step()
            running_loss += float(loss.detach())*len(positions)
        model.eval()
        valid_loss = 0.
        with torch.no_grad():
            for start in range(0, len(valid_positions), 2048):
                positions = valid_positions[start:start+2048]
                valid_loss += float(loss_fn(model(windows(positions)), labels[positions]))*len(positions)
        valid_loss /= len(valid_positions)
        history.append(dict(epoch=epoch+1, train_loss=running_loss/len(order), validation_loss=valid_loss))
        log(f"{folder.name}: GRU epoch {epoch+1}/{epochs} train={running_loss/len(order):.4f} valid={valid_loss:.4f}")
        if valid_loss < best_loss-1e-4:
            best_loss, bad_epochs = valid_loss, 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            bad_epochs += 1
        if bad_epochs >= 4:
            break
    model.load_state_dict(best_state)
    scores = np.zeros(len(frame), dtype="float32")
    with torch.no_grad():
        for start in range(0, len(frame), 2048):
            positions = np.arange(start, min(start+2048, len(frame)))
            scores[positions] = model(windows(positions)).sigmoid().cpu().numpy()
    torch.save(dict(state_dict=best_state, feature_count=numeric.shape[1], sequence_length=15,
                    selected_epoch=int(np.argmin([row['validation_loss'] for row in history])+1),
                    input_scaling="see preprocessing.joblib", target="pressure drop within 24h"), folder / "clone_gru.pt")
    pd.DataFrame(history).to_csv(folder / "gru_history.csv", index=False)
    del model, optimizer, values, labels
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return scores


def run_fold(frame, features, training_tools, evaluation_tools, fold_name, epochs):
    folder = ARTIFACTS / fold_name
    folder.mkdir(parents=True, exist_ok=True)
    fit = frame.part.eq("fit") & frame.Tool.isin(training_tools)
    calib = frame.part.eq("calib") & frame.Tool.isin(training_tools)
    test = frame.part.eq("test") & frame.Tool.isin(evaluation_tools)
    log(f"Starting {fold_name}: fit={fit.sum():,}, calib={calib.sum():,}, test={test.sum():,}")
    baseline = baseline_features(frame, training_tools)
    baseline_model = make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=2000, random_state=SEED))
    baseline_model.fit(baseline.loc[fit], frame.label.loc[fit])
    predictions = {"baseline_lr13": baseline_model.predict_proba(baseline)[:, 1]}
    joblib.dump(dict(model=baseline_model, columns=list(baseline)), folder / "baseline_lr13.joblib")
    tree_values, numeric, preprocessing = prepare_reference(frame, features, training_tools)
    joblib.dump(preprocessing, folder / "preprocessing.joblib")
    tree = HistGradientBoostingClassifier(learning_rate=.05, max_iter=250, class_weight="balanced", early_stopping=False, random_state=SEED)
    tree.fit(tree_values[fit], frame.label.loc[fit])
    predictions["clone_hgbm"] = tree.predict_proba(tree_values)[:, 1]
    joblib.dump(tree, folder / "clone_hgbm.joblib")
    log(f"{fold_name}: HGBM complete")
    smote = SMOTE(random_state=SEED)
    oversampled_inputs, oversampled_labels = smote.fit_resample(tree_values[fit], frame.label.loc[fit])
    xgb = XGBClassifier(objective="binary:logistic", eval_metric="logloss", max_depth=6,
                        learning_rate=.1, n_estimators=150, random_state=SEED,
                        n_jobs=6, tree_method="hist", device="cuda" if torch.cuda.is_available() else "cpu")
    xgb.fit(oversampled_inputs, oversampled_labels)
    predictions["clone_xgb_smote"] = xgb.predict_proba(tree_values)[:, 1]
    xgb.save_model(folder / "clone_xgb_smote.ubj")
    del oversampled_inputs, oversampled_labels
    log(f"{fold_name}: XGBoost + SMOTE complete")
    predictions["clone_gru"] = train_gru(frame, numeric, training_tools, folder, epochs)
    # Original RUL-dependent ensemble weights have no probability interpretation.
    # Fixed equal weighting is prespecified; no weights are tuned on the test set.
    predictions["clone_hgbm_gru"] = .5*predictions["clone_hgbm"] + .5*predictions["clone_gru"]
    rows, daily_outputs = [], []
    for name, scores in predictions.items():
        calibration_daily = daily_scores(frame, scores, calib)
        test_daily = daily_scores(frame, scores, test)
        test_daily["model"] = name
        test_daily["fold"] = fold_name
        thresholds = {}
        for lift in [2., 4.]:
            threshold, target = choose_threshold(calibration_daily, lift)
            thresholds[str(int(lift))] = threshold
            metrics = score_metrics(test_daily.label, test_daily.score, threshold)
            rows.append(dict(fold=fold_name, model=name, lift_target=lift, threshold=threshold,
                             calibration_positive_rate=float(calibration_daily.label.mean()),
                             calibration_precision_target=target, **metrics))
            test_daily[f"alarm_lift{int(lift)}"] = test_daily.score >= threshold
        calibration_daily.to_csv(folder / f"{name}_calibration_days.csv", index=False)
        test_daily.to_csv(folder / f"{name}_test_days.csv", index=False)
        daily_outputs.append(test_daily)
        thresholds["score_dtype"] = str(scores.dtype)
        save_json(folder / f"{name}_thresholds.json", thresholds)
        selected = rows[-1]
        log(f"{fold_name} {name}: P={selected['precision']:.3f} R={selected['recall']:.3f} F1={selected['f1']:.3f} AUC={selected['auc']:.3f}")
    pd.DataFrame(rows).to_csv(folder / "metrics.csv", index=False)
    return rows, pd.concat(daily_outputs, ignore_index=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--folds", choices=["all", "same", "unseen"], default="all")
    args = parser.parse_args()
    random.seed(SEED)
    np.random.seed(SEED)
    OUT.mkdir(parents=True, exist_ok=True)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    frame, spans, paths = load_data()
    spans.to_csv(OUT / "split_boundaries.csv")
    log("Reusing upstream rolling feature function (causal history by tool)")
    features = reference_features(frame)
    all_tools = sorted(frame.Tool.unique())
    permutation = np.random.default_rng(SEED).permutation(all_tools)
    folds = [list(permutation[index::4]) for index in range(4)]
    revision = "6ba6b026340cd21a95f8846344c1996a02bcc97d"
    if (UPSTREAM / ".git").exists():
        actual_revision = subprocess.check_output(["git", "-C", str(UPSTREAM), "rev-parse", "HEAD"], text=True).strip()
        if actual_revision != revision:
            raise ValueError("Upstream revision differs from the frozen reference files")
    manifest = dict(source="https://github.com/ki-seoultech/PHM2018-Ion-Mill-Etch-3Stage-Predictive-Maintenance",
                    source_commit=revision, experiment="clone_24h_adaptation", seed=SEED,
                    python=platform.python_version(), sklearn=sklearn.__version__, torch=torch.__version__,
                    device=torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
                    arguments=vars(args), rows=len(frame), reference_features=list(features), unseen_tool_folds=folds,
                    input_files=[dict(path=str(path.relative_to(ROOT)), size=path.stat().st_size,
                                      sha256=hashlib.file_digest(path.open('rb'), 'sha256').hexdigest()) for path in paths],
                    assumptions=["Labels from first fault strictly AFTER segment end, horizon 86400 seconds.",
                                 "Day=max row label/max row score, relative to each tool's first observation.",
                                 "Fit ends at 60%, calibration 60-75%, test begins 75%+1 day.",
                                 "Fit labels purged 1 day before calibration; final unobservable day excluded.",
                                 "Unseen-tool preprocessing and thresholds fitted on other tools only.",
                                 "Historical faults before prediction time are assumed available.",
                                 "GRU early stopping uses 50-60% within fit tools, never calibration/test.",
                                 "All models use same eligible test days. No future fill/isotonic/target episode resets.",
                                 "13-feature LR is rerun; user-reported historical numbers are not asserted identical."])
    save_json(OUT / "manifest.json", manifest)
    rows, day_tables = [], []
    if args.folds in ["same", "all"]:
        fold_rows, days = run_fold(frame, features, all_tools, all_tools, "same_tools", args.epochs)
        rows.extend(fold_rows)
        day_tables.append(days)
        pd.DataFrame(rows).to_csv(OUT / "fold_metrics.csv", index=False)
    if args.folds in ["unseen", "all"]:
        for fold_number, heldout in enumerate(folds, 1):
            training = [tool for tool in all_tools if tool not in heldout]
            fold_rows, days = run_fold(frame, features, training, heldout, f"unseen_{fold_number}", args.epochs)
            rows.extend(fold_rows)
            day_tables.append(days)
            pd.DataFrame(rows).to_csv(OUT / "fold_metrics.csv", index=False)
    daily = pd.concat(day_tables, ignore_index=True)
    daily.to_csv(OUT / "test_day_predictions.csv", index=False)
    summary = []
    for setup, selector in [("same_tools", daily.fold.eq("same_tools")), ("unseen_tools", daily.fold.str.startswith("unseen_"))]:
        for name, group in daily.loc[selector].groupby("model", sort=False):
            for lift in [2, 4]:
                predicted = group[f"alarm_lift{lift}"]
                metrics = score_metrics(group.label, group.score, float("inf"))
                # Each unseen fold has its own training-only calibration threshold.
                matrix = confusion_matrix(group.label, predicted, labels=[0, 1])
                metrics.update(precision=float(precision_score(group.label, predicted, zero_division=0)),
                               recall=float(recall_score(group.label, predicted, zero_division=0)),
                               f1=float(f1_score(group.label, predicted, zero_division=0)),
                               accuracy=float(accuracy_score(group.label, predicted)), alarm_days=int(predicted.sum()),
                               tn=int(matrix[0,0]), fp=int(matrix[0,1]), fn=int(matrix[1,0]), tp=int(matrix[1,1]))
                summary.append(dict(validation=setup, model=name, lift_target=lift, **metrics))
    pd.DataFrame(summary).to_csv(OUT / "summary.csv", index=False)
    log("COMPLETE: " + str(OUT / "summary.csv"))
    print(pd.DataFrame(summary).query("lift_target == 4").to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
