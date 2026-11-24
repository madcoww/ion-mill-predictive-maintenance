"""Clone of the GitHub diagnosis + capped-RUL task using PHM2018 M01 raw data.

Source revision: 6ba6b026340cd21a95f8846344c1996a02bcc97d (ki-seoultech).
No 24-hour binary target is used. Default action runs preparation and all models.
"""
from __future__ import annotations

import argparse
import ast
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "6")
os.environ.setdefault("MKL_NUM_THREADS", "6")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             f1_score, mean_absolute_error, mean_squared_error, r2_score)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
RAW = ROOT / "data/raw/phm_data_challenge_2018/train"
UPSTREAM = HERE.parent / "upstream"
REFERENCE = HERE / "reference"
CACHE = HERE / "cache"
OUT = HERE / "results"
ARTIFACTS = HERE / "artifacts"
SEED = 42
RATE = 140
SEQ_LEN = 15
MODES = {
    "F1": dict(target="TTF_Flowcool Pressure Too High Check Flowcool Pump", cap=14000, kind="GRU", name="압력 과다"),
    "F2": dict(target="TTF_Flowcool leak", cap=14000, kind="LSTM", name="누설"),
    "F3": dict(target="TTF_FlowCool Pressure Dropped Below Limit", cap=15000, kind="GRU", name="압력 저하"),
}
TARGETS = [entry["target"] for entry in MODES.values()]
CORE = ["FLOWCOOLPRESSURE", "IONGAUGEPRESSURE", "ETCHBEAMCURRENT", "FLOWCOOLFLOWRATE"]
SENSORS = ["IONGAUGEPRESSURE", "ETCHBEAMVOLTAGE", "ETCHBEAMCURRENT", "ETCHSUPPRESSORVOLTAGE",
           "ETCHSUPPRESSORCURRENT", "FLOWCOOLFLOWRATE", "FLOWCOOLPRESSURE", "ETCHGASCHANNEL1READBACK",
           "ETCHPBNGASREADBACK", "FIXTURETILTANGLE", "ROTATIONSPEED", "ACTUALROTATIONANGLE",
           "ETCHSOURCEUSAGE", "ETCHAUXSOURCETIMER", "ETCHAUX2SOURCETIMER", "ACTUALSTEPDURATION"]
REG_DROP = ["ROTATIONSPEED", "Tool", "Lot", "runnum", "ETCHAUX2SOURCETIMER",
            "ETCHSOURCEUSAGE", "ETCHAUXSOURCETIMER", "ACTUALSTEPDURATION"]
SOURCE_COMMIT = "6ba6b026340cd21a95f8846344c1996a02bcc97d"


def log(message):
    line = time.strftime("%H:%M:%S") + " " + str(message)
    print(line, flush=True)
    with (HERE / "run.log").open("a", encoding="utf-8") as stream:
        stream.write(line + "\n")


def save_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def source_function(filename, name):
    path = REFERENCE / filename
    tree = ast.parse(path.read_text(encoding="utf-8-sig"))
    selected = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    namespace = {"pd": pd, "np": np}
    exec(compile(ast.Module(body=[selected], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


def aggregate(frame, tool, mode):
    frame = frame.copy()
    if mode == "classifier":
        frame["Unit"] = tool + "_Lot" + frame.Lot.astype(str)
        # np.argmin on NaN-containing rows in the original can choose a missing target.
        # Inf masking preserves the intended nearest-known-fault label.
        target_array = frame[TARGETS].fillna(np.inf).to_numpy()
        frame["min_TTF"] = target_array.min(axis=1)
        frame["Fault_Type"] = target_array.argmin(axis=1)
        frame = frame.loc[np.isfinite(frame.min_TTF)].copy()
        target = "min_TTF"
    else:
        target = MODES[mode]["target"]
        frame = frame.dropna(subset=[target]).copy()
        frame["Unit"] = tool
    if frame.empty:
        return frame
    frame = frame.sort_values(["Unit", "time"], kind="stable").reset_index(drop=True)
    new_episode = frame.Unit.ne(frame.Unit.shift()) | frame[target].diff().gt(0)
    frame["episode_id"] = new_episode.cumsum().astype("int64") + int(tool[:2])*1000000
    frame["time_bin"] = frame.time // RATE
    excluded = {"Tool", "Lot", "Unit", "episode_id", "time", "time_bin", "FIXTURESHUTTERPOSITION", *TARGETS}
    if mode != "classifier":
        excluded.update(REG_DROP)
    numerical = [column for column in frame if column not in excluded and column not in ["recipe", "stage", "min_TTF", "Fault_Type"]]
    aggregation = {"time": "last", "recipe": "last", "stage": "last"}
    aggregation.update({column: "mean" for column in numerical})
    if mode == "classifier":
        aggregation.update({column: "min" for column in TARGETS})
        aggregation.update(min_TTF="min", Fault_Type="last")
    else:
        aggregation[target] = "min"
    binned = frame.groupby(["Unit", "episode_id", "time_bin"], sort=True).agg(aggregation).reset_index()
    binned["PhysicalTool"] = tool
    return binned


def prepare():
    sensor_paths = sorted(RAW.glob("*_M01_DC_train.csv"))
    assert len(sensor_paths) == 10, f"Expected 10 M01 tools, got {len(sensor_paths)}"
    records = []
    for sensor_path in sensor_paths:
        tool = sensor_path.name.split("_DC")[0]
        paths = {mode: CACHE / mode / f"{tool}.parquet" for mode in [*MODES, "classifier"]}
        if all(path.exists() for path in paths.values()):
            log(f"Cached {tool}")
            continue
        started = time.perf_counter()
        sensor = pd.read_csv(sensor_path, engine="pyarrow")
        ttf_path = RAW / "train_ttf" / sensor_path.name
        targets = pd.read_csv(ttf_path, engine="pyarrow")
        assert len(sensor) == len(targets)
        np.testing.assert_array_equal(sensor.time.to_numpy(), targets.time.to_numpy())
        raw_rows = len(sensor)
        # Source does not remove duplicates: preserve original rows here as well.
        selected = sensor.FIXTURESHUTTERPOSITION.eq(1)
        frame = pd.concat([sensor.loc[selected].reset_index(drop=True), targets.loc[selected].drop(columns="time").reset_index(drop=True)], axis=1)
        del sensor, targets
        gc.collect()
        record = dict(tool=tool, raw_rows=raw_rows, shutter1_rows=len(frame),
                      sensor_file=str(sensor_path.relative_to(ROOT)), sensor_bytes=sensor_path.stat().st_size,
                      target_file=str(ttf_path.relative_to(ROOT)), target_bytes=ttf_path.stat().st_size)
        for mode, path in paths.items():
            path.parent.mkdir(parents=True, exist_ok=True)
            binned = aggregate(frame, tool, mode)
            binned.to_parquet(path, index=False)
            record[mode + "_bins"] = len(binned)
            del binned
        del frame
        gc.collect()
        try:
            import pyarrow as pa
            pa.default_memory_pool().release_unused()
        except Exception:
            pass
        records.append(record)
        pd.DataFrame(records).to_csv(OUT / "preparation.csv", index=False)
        log(f"Prepared {tool}: {raw_rows:,} raw rows -> 140s bins, {time.perf_counter()-started:.1f}s")


def load_binned(mode):
    paths = sorted((CACHE / mode).glob("*.parquet"))
    assert len(paths) == 10
    frames = [pd.read_parquet(path) for path in paths]
    frames = [frame for frame in frames if not frame.empty]
    frame = pd.concat(frames, ignore_index=True).sort_values(["Unit", "time"], kind="stable").reset_index(drop=True)
    source = ("src/02_fault_classifier/train_save_classifier.py" if mode == "classifier" else
              "src/03_inference_pipeline/train_save_regressor_f3.py")
    frame = source_function(source, "add_reference_features")(frame, CORE, w=10)
    # Reproduce the source's OFFLINE preprocessing, including backward fill.
    frame = frame.bfill()
    episode_lists = frame.groupby("Unit", sort=True).episode_id.unique()
    validation_episodes = [episodes[-1] for episodes in episode_lists if len(episodes) > 1]
    is_validation = frame.episode_id.isin(validation_episodes)
    assert is_validation.any() and (~is_validation).any()
    assert not set(frame.loc[is_validation, "episode_id"]).intersection(frame.loc[~is_validation, "episode_id"])
    folder = ARTIFACTS / mode
    folder.mkdir(parents=True, exist_ok=True)
    split = frame.groupby(["PhysicalTool", "Unit", "episode_id"], sort=False).agg(first=("time", "min"), last=("time", "max"), rows=("time", "size")).reset_index()
    split["part"] = np.where(split.episode_id.isin(validation_episodes), "validation", "train")
    split.to_csv(folder / "episodes.csv", index=False)
    return frame, is_validation, folder


def classifier():
    from imblearn.over_sampling import SMOTE
    from xgboost import XGBClassifier
    frame, validation, folder = load_binned("classifier")
    excluded = {"PhysicalTool", "Unit", "time", "time_bin", "episode_id", "min_TTF", "Fault_Type", *TARGETS}
    feature_columns = [column for column in frame if column not in excluded]
    categorical = [column for column in feature_columns if column in ["recipe", "stage"]]
    numeric = [column for column in feature_columns if column not in categorical]
    eligible = frame.min_TTF.le(14000)
    train_rows, valid_rows = ~validation & eligible, validation & eligible
    label = frame.Fault_Type.astype(int)
    assert not any(column.startswith("TTF_") for column in feature_columns)
    log(f"Classifier H<=14000s: train={train_rows.sum():,}, validation={valid_rows.sum():,}, label counts={label[valid_rows].value_counts().to_dict()}")
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), categorical)])
    train_values = preprocessor.fit_transform(frame.loc[train_rows, feature_columns])
    valid_values = preprocessor.transform(frame.loc[valid_rows, feature_columns])
    oversampler = SMOTE(random_state=SEED)
    balanced_values, balanced_labels = oversampler.fit_resample(train_values, label[train_rows])
    model = XGBClassifier(objective="multi:softprob", num_class=3, eval_metric="mlogloss", max_depth=6,
                          learning_rate=.1, n_estimators=150, random_state=SEED, n_jobs=6, tree_method="hist")
    model.fit(balanced_values, balanced_labels)
    predicted = model.predict(valid_values)
    probabilities = model.predict_proba(valid_values)
    names = [MODES[mode]["name"] for mode in MODES]
    metrics = dict(task="fault_cause_within_14000s", model="XGBoost+SMOTE", accuracy=float(accuracy_score(label[valid_rows], predicted)),
                   macro_f1=float(f1_score(label[valid_rows], predicted, labels=[0,1,2], average="macro", zero_division=0)),
                   train_rows=int(train_rows.sum()), validation_rows=int(valid_rows.sum()),
                   physical_tools=int(frame.PhysicalTool.nunique()), validation_units=int(frame.loc[valid_rows, "Unit"].nunique()),
                   classification_report=classification_report(label[valid_rows], predicted, labels=[0,1,2], target_names=names, output_dict=True, zero_division=0),
                   confusion_matrix=confusion_matrix(label[valid_rows], predicted, labels=[0,1,2]).tolist(),
                   scope="conditional fault-cause classification, not healthy/fault detection")
    predictions = frame.loc[valid_rows, ["PhysicalTool", "Unit", "time", "episode_id", "min_TTF", "Fault_Type"]].copy()
    predictions["predicted"] = predicted
    for position, mode in enumerate(MODES):
        predictions["probability_" + mode] = probabilities[:, position]
    predictions.to_csv(folder / "predictions.csv", index=False)
    joblib.dump(preprocessor, folder / "preprocessor.joblib")
    model.save_model(folder / "clone_fault_classifier.ubj")
    # Reload verifies actual serialized artifacts before reporting the metric.
    restored = XGBClassifier()
    restored.load_model(folder / "clone_fault_classifier.ubj")
    np.testing.assert_array_equal(restored.predict(valid_values), predicted)
    save_json(OUT / "classifier.json", metrics)
    log(f"CLASSIFIER COMPLETE: Macro F1={metrics['macro_f1']:.4f}, accuracy={metrics['accuracy']:.4f}")


def reg_metrics(truth, predicted):
    return dict(rmse_seconds=float(np.sqrt(mean_squared_error(truth, predicted))),
                mae_seconds=float(mean_absolute_error(truth, predicted)), r2=float(r2_score(truth, predicted)),
                source_score=float(np.sum(np.exp(-.001*np.asarray(truth))*np.abs(np.asarray(truth)-predicted))),
                rows=len(truth))


def sequence_ends(episodes):
    positions = pd.Series(np.arange(len(episodes)))
    within = positions.groupby(np.asarray(episodes), sort=False).cumcount().to_numpy()
    return np.flatnonzero(within >= SEQ_LEN-1)


def regression(mode, epochs):
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(6)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.keras.utils.set_random_seed(SEED)
    specification = MODES[mode]
    target, cap, kind = specification["target"], specification["cap"], specification["kind"]
    frame, validation, folder = load_binned(mode)
    # Defense against the other-fault TTF leakage present in some source variants.
    excluded = {"PhysicalTool", "Unit", "time", "time_bin", "episode_id", *TARGETS, *REG_DROP}
    feature_columns = [column for column in frame if column not in excluded]
    categorical = [column for column in feature_columns if column in ["recipe", "stage"]]
    numeric = [column for column in feature_columns if column not in categorical]
    assert not any(column.startswith("TTF_") for column in feature_columns)
    train = frame.loc[~validation].copy().reset_index(drop=True)
    valid = frame.loc[validation].copy().reset_index(drop=True)
    train[numeric] = train[numeric].astype("float64")
    valid[numeric] = valid[numeric].astype("float64")
    scalers = {}
    for unit in frame.Unit.unique():
        own_train, own_valid = train.Unit.eq(unit), valid.Unit.eq(unit)
        if not own_train.any():
            continue
        scaler = StandardScaler().fit(train.loc[own_train, numeric])
        train.loc[own_train, numeric] = scaler.transform(train.loc[own_train, numeric])
        if own_valid.any():
            valid.loc[own_valid, numeric] = scaler.transform(valid.loc[own_valid, numeric])
        scalers[unit] = scaler
    train_target = np.log1p(train[target].clip(0, cap).to_numpy())
    valid_target = valid[target].clip(0, cap).to_numpy()
    preprocessor = ColumnTransformer([
        ("num", SimpleImputer(strategy="median", keep_empty_features=True), numeric),
        ("cat", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")),
                          ("encoder", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1))]), categorical)])
    tree = Pipeline([("prep", preprocessor),
                     ("model", HistGradientBoostingRegressor(loss="squared_error", learning_rate=.05, max_iter=250, random_state=SEED))])
    log(f"{mode} {kind}: HGBM fitting train={len(train):,}, validation={len(valid):,}, cap={cap}s")
    tree.fit(train[feature_columns], train_target)
    tree_predictions = np.expm1(tree.predict(valid[feature_columns])).clip(0, cap)
    imputer = SimpleImputer(strategy="median", keep_empty_features=True)
    numeric_train = imputer.fit_transform(train[numeric]).astype("float32")
    numeric_valid = imputer.transform(valid[numeric]).astype("float32")
    train_ends, valid_ends = sequence_ends(train.episode_id), sequence_ends(valid.episode_id)
    assert len(train_ends) and len(valid_ends)
    split_point = int(len(train_ends)*.8)
    fit_ends, internal_valid_ends = train_ends[:split_point], train_ends[split_point:]
    offsets = np.arange(SEQ_LEN-1, -1, -1)

    class WindowBatches(tf.keras.utils.PyDataset):
        def __init__(self, inputs, labels, ends, shuffle):
            super().__init__(workers=1, use_multiprocessing=False, max_queue_size=2)
            self.inputs, self.labels, self.ends, self.shuffle = inputs, labels, ends.copy(), shuffle
            self.generator = np.random.default_rng(SEED)
            if self.shuffle:
                self.generator.shuffle(self.ends)

        def __len__(self):
            return (len(self.ends)+511)//512

        def __getitem__(self, batch_index):
            ends = self.ends[batch_index*512:(batch_index+1)*512]
            return self.inputs[ends[:, None]-offsets], self.labels[ends].astype("float32")

        def on_epoch_end(self):
            if self.shuffle:
                self.generator.shuffle(self.ends)

    inputs = tf.keras.Input(shape=(SEQ_LEN, numeric_train.shape[1]))
    cell = tf.keras.layers.GRU if kind == "GRU" else tf.keras.layers.LSTM
    hidden = cell(64, return_sequences=True)(inputs)
    hidden = cell(32)(hidden)
    hidden = tf.keras.layers.Dropout(.2)(hidden)
    hidden = tf.keras.layers.Dense(16, activation="relu")(hidden)
    outputs = tf.keras.layers.Dense(1)(hidden)
    neural = tf.keras.Model(inputs, outputs)
    neural.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=.001), loss="mse")

    class Progress(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            log(f"{mode} {kind} epoch {epoch+1}: train_loss={logs['loss']:.5f}, validation_loss={logs['val_loss']:.5f}")

    log(f"{mode}: {kind} training on CPU (same TensorFlow/Keras backend as original), windows={len(train_ends):,}")
    history = neural.fit(WindowBatches(numeric_train, train_target, fit_ends, True),
                         validation_data=WindowBatches(numeric_train, train_target, internal_valid_ends, False),
                         epochs=epochs, shuffle=False, verbose=0,
                         callbacks=[Progress(), tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=4, restore_best_weights=True)])
    pd.DataFrame(history.history).assign(epoch=lambda table: np.arange(1, len(table)+1)).to_csv(folder / "training_history.csv", index=False)
    neural_predictions = []
    for start in range(0, len(valid_ends), 2048):
        ends = valid_ends[start:start+2048]
        batch = numeric_valid[ends[:, None]-offsets]
        neural_predictions.append(neural(batch, training=False).numpy().reshape(-1))
    neural_predictions = np.expm1(np.concatenate(neural_predictions)).clip(0, cap)
    aligned_tree = tree_predictions[valid_ends]
    truth = valid_target[valid_ends]
    alpha = np.clip((cap-neural_predictions)/(cap*.8), .1, .9)
    ensemble = alpha*neural_predictions + (1-alpha)*aligned_tree
    offline_isotonic = np.zeros_like(ensemble)
    isotonic_models = {}
    aligned_episodes = valid.episode_id.to_numpy()[valid_ends]
    for episode in np.unique(aligned_episodes):
        positions = np.flatnonzero(aligned_episodes == episode)
        isotonic = IsotonicRegression(increasing=False, out_of_bounds="clip")
        offline_isotonic[positions] = isotonic.fit_transform(np.arange(len(positions)), ensemble[positions])
        isotonic_models[int(episode)] = isotonic
    predictions = valid.iloc[valid_ends][["PhysicalTool", "Unit", "time", "episode_id", target]].copy().reset_index(drop=True)
    predictions["target_capped"] = truth
    methods = {"always_cap": np.full(len(truth), cap),
               "training_mean": np.full(len(truth), np.expm1(train_target).mean()),
               "HGBM": aligned_tree, kind: neural_predictions,
               "dynamic_ensemble": ensemble, "offline_isotonic_ensemble": offline_isotonic}
    rows = []
    for name, prediction in methods.items():
        predictions[name] = prediction
        rows.append(dict(fault=mode, fault_name=specification["name"], model=name, scope="all_capped", **reg_metrics(truth, prediction)))
        near = valid[target].to_numpy()[valid_ends] < cap
        if near.sum() >= 2:
            rows.append(dict(fault=mode, fault_name=specification["name"], model=name, scope="near_failure_only", **reg_metrics(truth[near], prediction[near])))
    pd.DataFrame(rows).to_csv(OUT / f"{mode}_metrics.csv", index=False)
    predictions.to_csv(folder / "predictions.csv", index=False)
    joblib.dump(tree, folder / "clone_hgbm.joblib")
    joblib.dump(dict(scalers=scalers, imputer=imputer, numeric_columns=numeric, feature_columns=feature_columns), folder / "preprocessing.joblib")
    joblib.dump(isotonic_models, folder / "offline_isotonic.joblib")
    neural.save(folder / f"clone_{kind.lower()}.keras")
    # Round-trip model and metadata validation on actual held-out data.
    restored_tree = joblib.load(folder / "clone_hgbm.joblib")
    np.testing.assert_allclose(restored_tree.predict(valid[feature_columns].iloc[:100]), tree.predict(valid[feature_columns].iloc[:100]))
    restored_neural = tf.keras.models.load_model(folder / f"clone_{kind.lower()}.keras", compile=False)
    check_ends = valid_ends[:min(len(valid_ends), 128)]
    check_batch = numeric_valid[check_ends[:, None]-offsets]
    np.testing.assert_allclose(restored_neural(check_batch, training=False).numpy(), neural(check_batch, training=False).numpy(), rtol=1e-5, atol=1e-5)
    metadata = dict(fault=mode, target=target, cap_seconds=cap, neural_kind=kind, train_bins=len(train), validation_bins=len(valid),
                    validation_windows=len(truth), near_failure_windows=int(near.sum()), validation_episodes=int(len(np.unique(aligned_episodes))),
                    training_epochs=len(history.history["loss"]), best_epoch=int(np.argmin(history.history["val_loss"])+1),
                    tensorflow=tf.__version__, saved_models_reload_verified=True,
                    benchmark="offline last failure cycle per M01 tool; target-conditioned cycles and capped target",
                    future_information="source bfill and full-episode isotonic retained for offline reproduction; not an online metric")
    save_json(folder / "metadata.json", metadata)
    best = next(row for row in rows if row["model"] == "offline_isotonic_ensemble" and row["scope"] == "all_capped")
    log(f"{mode} COMPLETE: offline MAE={best['mae_seconds']:.1f}s, RMSE={best['rmse_seconds']:.1f}s, R2={best['r2']:.3f}")


def finish():
    tables = [pd.read_csv(OUT / f"{mode}_metrics.csv") for mode in MODES]
    pd.concat(tables, ignore_index=True).to_csv(OUT / "rul_metrics.csv", index=False)
    source_files = ["src/03_inference_pipeline/train_save_regressor_f3.py",
                    "src/03_inference_pipeline/train_save_regressors_f1_f2.py",
                    "src/02_fault_classifier/train_save_classifier.py", "src/02_fault_classifier/build_v2_dataset.py"]
    save_json(OUT / "manifest.json", dict(source="https://github.com/ki-seoultech/PHM2018-Ion-Mill-Etch-3Stage-Predictive-Maintenance",
        source_commit=SOURCE_COMMIT, task="fault cause + capped RUL", tools="01_M01 through 10_M01", bin_seconds=RATE,
        environment=dict(python=sys.version, numpy=np.__version__, pandas=pd.__version__, sklearn=sklearn.__version__),
        sequence_length=SEQ_LEN, seed=SEED, source_files={name:hashlib.sha256((REFERENCE/name).read_bytes()).hexdigest() for name in source_files},
        changes=["Raw PHM CSV + official aligned TTF used because author's labeled master CSVs are not published.",
                 "Masked missing TTF to infinity before nearest-fault argmin (prevents choosing a missing target).",
                 "Chunk-free per-tool 140s aggregation replaces a single in-memory master CSV.",
                 "Memory-efficient Keras PyDataset replaces materialized overlapping windows; same layers/loss/hyperparameters.",
                 "Seed 42 and six CPU threads fixed; unit scalers also saved for reproducibility.",
                 "No 24h warning label, no unseen-tool score, no operational safety-gate performance claim.",
                 "Source bfill and full-episode isotonic are retained as OFFLINE benchmark only; pre-isotonic scores also reported."]));
    log("ALL TASKS COMPLETE: diagnosis + F1/F2/F3 RUL")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["all", "prepare", "classifier", *MODES, "finish"], default="all")
    parser.add_argument("--epochs", type=int, default=20)
    args = parser.parse_args()
    for directory in [CACHE, OUT, ARTIFACTS]:
        directory.mkdir(parents=True, exist_ok=True)
    if args.task == "all":
        for task in ["prepare", "classifier", *MODES, "finish"]:
            subprocess.run([sys.executable, "-X", "utf8", "-u", str(Path(__file__).resolve()), "--task", task, "--epochs", str(args.epochs)], check=True, cwd=ROOT)
    elif args.task == "prepare":
        prepare()
    elif args.task == "classifier":
        classifier()
    elif args.task == "finish":
        finish()
    else:
        regression(args.task, args.epochs)


if __name__ == "__main__":
    main()
