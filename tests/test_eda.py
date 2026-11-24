"""Verify event and observation boundaries that materially change EDA results."""
import unittest

import numpy as np
import pandas as pd

from src.eda.run import CONDITIONS, FAULTS, coverage, fault_events, segment_ids


class EdaBoundaryTests(unittest.TestCase):
    def test_separate_recipe_runs_shutter_and_missing_observation_periods(self):
        frame = pd.DataFrame({column: [1] * 7 for column in CONDITIONS})
        frame["time"] = [0, 4, 8, 12, 16, 20, 1000]
        frame["recipe"] = [1, 1, 2, 2, 2, 2, 2]
        frame["runnum"] = [1, 1, 1, 2, 2, 2, 2]
        frame["FIXTURESHUTTERPOSITION"] = [1, 1, 1, 1, 0, 1, 1]
        self.assertEqual(segment_ids(frame, 60).tolist(), [1, 1, 2, 3, 4, 5, 6])

    def test_missing_steps_do_not_collapse_into_known_steps(self):
        frame = pd.DataFrame({column: [1] * 4 for column in CONDITIONS})
        frame["time"] = [0, 4, 8, 12]
        frame["recipe_step"] = [1, np.nan, np.nan, 1]
        self.assertEqual(segment_ids(frame, 60).tolist(), [1, 2, 2, 3])

    def test_fault_clusters_are_per_equipment_and_type_with_chain_rule(self):
        low, high = list(FAULTS)[:2]
        faults = pd.DataFrame({"Tool": ["01M01"] * 6 + ["02M01"],
                               "time": [0, 0, 20 * 3600, 40 * 3600, 70 * 3600, 0, 0],
                               "fault_name": [low] * 5 + [high, low]})
        events = fault_events(faults, 24)
        self.assertEqual(len(events), 4)
        self.assertEqual(sorted(events["record_count"]), [1, 1, 1, 3])

    def test_coverage_excludes_event_time_and_does_not_count_duplicates_twice(self):
        frame = pd.DataFrame({"time": [0, 0, 59, 60, 3599, 3600]})
        etching = frame.iloc[[0, 1, 4]]
        events = pd.DataFrame({"event_time": [3600], "fault_type": ["pressure_low"]})
        result = coverage(frame, etching, events, 1).iloc[0]
        self.assertEqual(result["unique_timestamps"], 4)
        self.assertEqual(result["observed_minutes"], 3)
        self.assertEqual(result["minute_coverage"], 3 / 60)
        self.assertTrue(result["full_pre_window_within_period"])

    def test_no_faults_returns_valid_empty_tables(self):
        events = fault_events(pd.DataFrame(columns=["Tool", "time", "fault_name"]), 24)
        result = coverage(pd.DataFrame({"time": [0, 4]}), pd.DataFrame({"time": [0]}), events, 24)
        self.assertEqual(len(result), 0)
        self.assertIn("minute_coverage", result.columns)


if __name__ == "__main__":
    unittest.main()
