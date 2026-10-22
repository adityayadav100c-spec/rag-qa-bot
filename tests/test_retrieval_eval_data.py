import json
import re
import unittest
from pathlib import Path

ROOT = Path(__file__).parents[1]


class RetrievalEvaluationDataTests(unittest.TestCase):
    def test_questions_point_to_real_handbook_sections(self):
        handbook = (ROOT / "data" / "knowledge_base.md").read_text(encoding="utf-8")
        section_names = {
            heading.lstrip("# ").strip()
            for heading in re.findall(r"(?m)^## .+$", handbook)
        }
        cases = [
            json.loads(line)
            for line in (ROOT / "data" / "retrieval_eval.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        self.assertGreaterEqual(len(cases), 10)
        self.assertEqual(len({case["question"] for case in cases}), len(cases))
        self.assertTrue(
            all(case["expected_section"] in section_names for case in cases)
        )


if __name__ == "__main__":
    unittest.main()
