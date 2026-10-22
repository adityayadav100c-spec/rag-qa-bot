def score_retrieval(cases: list[dict], retrieved: list[list[str]]) -> dict:
    if len(cases) != len(retrieved):
        raise ValueError("Each evaluation question must have one retrieval result.")
    if not cases:
        raise ValueError("Retrieval evaluation set is empty.")

    per_question = []
    for case, sections in zip(cases, retrieved):
        expected = case["expected_section"]
        rank = next(
            (i + 1 for i, section in enumerate(sections) if section == expected),
            0,
        )
        per_question.append(
            {
                "question": case["question"],
                "expected_section": expected,
                "retrieved_sections": " | ".join(sections),
                "first_relevant_rank": rank or "",
                "hit_at_1": rank == 1,
                "hit_at_3": 0 < rank <= 3,
                "reciprocal_rank_at_3": (1 / rank if 0 < rank <= 3 else 0),
            }
        )

    count = len(per_question)
    metrics = {
        "question_count": count,
        "recall_at_1": sum(row["hit_at_1"] for row in per_question) / count,
        "recall_at_3": sum(row["hit_at_3"] for row in per_question) / count,
        "mean_reciprocal_rank_at_3": (
            sum(row["reciprocal_rank_at_3"] for row in per_question) / count
        ),
    }
    return {"metrics": metrics, "per_question": per_question}
