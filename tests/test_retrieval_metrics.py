import unittest

from retrieval_metrics import score_retrieval


class RetrievalMetricsTests(unittest.TestCase):
    def test_scores_first_hit_recall_cutoff_and_mrr(self):
        cases = [
            {"question": "first", "expected_section": "target"},
            {"question": "third", "expected_section": "target"},
            {"question": "miss", "expected_section": "target"},
        ]
        result = score_retrieval(
            cases,
            [
                ["target", "other", "third"],
                ["other", "another", "target"],
                ["other", "another", "third", "target"],
            ],
        )
        self.assertAlmostEqual(result["metrics"]["recall_at_1"], 1 / 3)
        self.assertAlmostEqual(result["metrics"]["recall_at_3"], 2 / 3)
        self.assertAlmostEqual(
            result["metrics"]["mean_reciprocal_rank_at_3"],
            (1 + 1 / 3) / 3,
        )

    def test_requires_one_result_per_question(self):
        with self.assertRaisesRegex(ValueError, "one retrieval result"):
            score_retrieval(
                [{"question": "q", "expected_section": "s"}],
                [],
            )

    def test_rejects_an_empty_evaluation_set(self):
        with self.assertRaisesRegex(ValueError, "empty"):
            score_retrieval([], [])


if __name__ == "__main__":
    unittest.main()
