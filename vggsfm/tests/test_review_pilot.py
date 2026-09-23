"""Evidence parsing must not turn a failed inference into a successful review."""

import importlib.util
import json
import tempfile
import unittest
from pathlib import Path


spec = importlib.util.spec_from_file_location(
    "review_pilot", Path(__file__).resolve().parents[1] / "scripts" / "review_pilot.py"
)
review = importlib.util.module_from_spec(spec)
spec.loader.exec_module(review)


class PilotReviewTests(unittest.TestCase):
    def test_exact_success_marker_and_final_failure_rejected(self):
        marker = "===== official VGGSfM inference exit=0 elapsed=122.063s @ 2026-09-19T17:20:28+00:00 ====="
        self.assertEqual(review.successful_inference(marker)["elapsed_as_logged"], "122.063")
        with self.assertRaisesRegex(ValueError, "not exit=0"):
            review.successful_inference(marker + "\n" + marker.replace("exit=0", "exit=1"))

    def test_success_claim_without_executor_completion_rejected(self):
        with self.assertRaises(ValueError):
            review.successful_inference("Demo Finished Successfully")

    def test_name_map_must_cover_all_requested_images(self):
        request = {"input_manifest": {"images": [{"path": "images/a.jpg"}]},
                   "official_image_name_map": [{"official": "a.jpg", "source": "images/a.jpg"}]}
        self.assertEqual(review.requested_names(request)[1], {"a.jpg": "images/a.jpg"})
        request["official_image_name_map"][0]["source"] = "images/wrong.jpg"
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            review.requested_names(request)

    def test_resource_summary_preserves_sampled_scope(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "resources-42.jsonl"
            rows = [{"time_unix": 1.0, "pids": [42], "rss_kib": 20, "gpu_mib": 30},
                    {"time_unix": 3.0, "pids": [42, 43], "rss_kib": 40, "gpu_mib": None}]
            path.write_text("\n".join(json.dumps(row) for row in rows))
            actual = review.summarize_samples(path)
            self.assertEqual(actual["samples"], 2)
            self.assertEqual(actual["peak_process_tree_rss_kib"], 40)
            self.assertEqual(actual["peak_process_tree_gpu_mib"], 30)
            self.assertEqual(actual["gpu_samples_available"], 1)
            rows[-1]["pids"] = [43]
            path.write_text("\n".join(json.dumps(row) for row in rows))
            with self.assertRaisesRegex(ValueError, "attributed"):
                review.summarize_samples(path)

    def test_frozen_record_detects_content_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "receipt.json"
            path.write_text("first")
            first = review.frozen_record(path)
            self.assertEqual(review.frozen_record(path), first)
            path.write_text("other")
            self.assertNotEqual(review.frozen_record(path), first)


if __name__ == "__main__":
    unittest.main()
