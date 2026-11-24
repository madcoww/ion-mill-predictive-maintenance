"""Independent audit of recorded diagnosis/RUL scores and held-out cycles."""
from pathlib import Path
import json

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, mean_squared_error, r2_score

HERE = Path(__file__).resolve().parent
OUT = HERE / "results"


def check_split(folder):
    episodes = pd.read_csv(folder / "episodes.csv")
    training = episodes.loc[episodes.part.eq("train")]
    validation = episodes.loc[episodes.part.eq("validation")]
    assert not set(training.episode_id).intersection(validation.episode_id)
    # Source reserves the last episode only for Units with >1 known episode.
    for unit, own in episodes.groupby("Unit"):
        reserved = own.loc[own.part.eq("validation")]
        if len(own) > 1:
            assert len(reserved) == 1
            assert reserved["last"].iloc[0] == own["last"].max()
        else:
            assert len(reserved) == 0
    return episodes, validation


def main():
    checks = []
    classifier_folder = HERE / "artifacts/classifier"
    _, classifier_validation = check_split(classifier_folder)
    predictions = pd.read_csv(classifier_folder / "predictions.csv", float_precision="round_trip")
    scores = json.loads((OUT / "classifier.json").read_text(encoding="utf-8"))
    assert predictions.min_TTF.le(14000).all()
    assert set(predictions.episode_id).issubset(classifier_validation.episode_id)
    np.testing.assert_allclose(f1_score(predictions.Fault_Type, predictions.predicted, labels=[0,1,2], average="macro"), scores["macro_f1"])
    np.testing.assert_allclose(accuracy_score(predictions.Fault_Type, predictions.predicted), scores["accuracy"])
    probabilities = predictions[["probability_F1", "probability_F2", "probability_F3"]].to_numpy()
    np.testing.assert_allclose(probabilities.sum(axis=1), 1, atol=1e-6)
    np.testing.assert_array_equal(probabilities.argmax(axis=1), predictions.predicted)
    # Verify labels against the binned official-TTF reconstruction, not model output.
    raw_label_tables = []
    for path in sorted((HERE / "cache/classifier").glob("*.parquet")):
        table = pd.read_parquet(path)
        if not table.empty:
            raw_label_tables.append(table[["Unit", "time", "episode_id", "min_TTF", "Fault_Type"]])
    labels = pd.concat(raw_label_tables, ignore_index=True)
    merged = predictions.merge(labels, on=["Unit", "time", "episode_id"], validate="one_to_one", suffixes=("_saved", "_cache"))
    assert len(merged) == len(predictions)
    np.testing.assert_array_equal(merged.Fault_Type_saved, merged.Fault_Type_cache)
    np.testing.assert_allclose(merged.min_TTF_saved, merged.min_TTF_cache)
    checks.append(dict(task="classifier", metrics_verified=True, cached_labels_verified=True, split_verified=True))
    for mode in ["F1", "F2", "F3"]:
        folder = HERE / "artifacts" / mode
        _, validation = check_split(folder)
        metadata = json.loads((folder / "metadata.json").read_text(encoding="utf-8"))
        assert metadata["saved_models_reload_verified"]
        predictions = pd.read_csv(folder / "predictions.csv", float_precision="round_trip")
        metrics = pd.read_csv(OUT / f"{mode}_metrics.csv", float_precision="round_trip")
        assert len(predictions) == sum(max(int(count)-14, 0) for count in validation.rows)
        assert set(predictions.episode_id).issubset(validation.episode_id)
        cap = metadata["cap_seconds"]
        np.testing.assert_allclose(predictions[metadata["target"]].clip(0, cap), predictions.target_capped)
        neural_name = metadata["neural_kind"]
        predictions[neural_name] = predictions[neural_name].astype("float32")
        alpha = np.clip((cap-predictions[neural_name].to_numpy())/(cap*.8), .1, .9)
        ensemble = alpha*predictions[neural_name] + (1-alpha)*predictions.HGBM
        np.testing.assert_allclose(ensemble, predictions.dynamic_ensemble, rtol=1e-5, atol=1e-3)
        for episode, group in predictions.groupby("episode_id", sort=False):
            fitted = IsotonicRegression(increasing=False).fit_transform(np.arange(len(group)), group.dynamic_ensemble)
            np.testing.assert_allclose(fitted, group.offline_isotonic_ensemble, rtol=1e-7, atol=1e-3)
            assert (group.offline_isotonic_ensemble.diff().dropna() <= 1e-6).all()
        for row in metrics.itertuples():
            selected = predictions if row.scope == "all_capped" else predictions.loc[predictions[metadata["target"]] < cap]
            assert len(selected) == row.rows
            actual, predicted = selected.target_capped, selected[row.model]
            calculated = [mean_absolute_error(actual, predicted), np.sqrt(mean_squared_error(actual, predicted)), r2_score(actual, predicted)]
            np.testing.assert_allclose(calculated, [row.mae_seconds, row.rmse_seconds, row.r2], rtol=1e-5, atol=1e-3)
            score = np.sum(np.exp(-.001*actual)*np.abs(actual-predicted))
            np.testing.assert_allclose(score, row.source_score, rtol=1e-5, atol=.01)
        checks.append(dict(task=mode, metrics_verified=True, split_verified=True, target_cap_verified=True,
                           sequence_window_counts_verified=True, ensemble_verified=True, offline_isotonic_verified=True,
                           saved_models_reload_verified=True))
    (OUT / "verification.json").write_text(json.dumps(dict(status="passed", checks=checks,
        scope="offline source-style evaluation; does not certify online validity"), ensure_ascii=False, indent=2), encoding="utf-8")
    print("ALL AUDITS PASSED: classifier + F1/F2/F3 RUL")


if __name__ == "__main__":
    main()
