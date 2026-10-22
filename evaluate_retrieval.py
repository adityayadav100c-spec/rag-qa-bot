import json
import argparse
import csv
from pathlib import Path

from dotenv import load_dotenv

from rag import build_retriever
from retrieval_metrics import score_retrieval

ROOT = Path(__file__).parent


def main():
    parser = argparse.ArgumentParser(description="Score labeled retrieval questions.")
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Retrieve at least 3 sections so Recall@1 and Recall@3 remain comparable.",
    )
    args = parser.parse_args()
    if args.top_k < 3 or args.top_k > 20:
        raise ValueError("top-k must be between 3 and 20.")
    load_dotenv(ROOT / ".env")
    cases = [
        json.loads(line)
        for line in (ROOT / "data" / "retrieval_eval.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    ]
    retriever = build_retriever(top_k=args.top_k)
    retrieved_sections = []
    for case in cases:
        results = retriever.invoke(case["question"])
        sections = [doc.metadata["section"] for doc in results]
        retrieved_sections.append(sections)

    scored = score_retrieval(cases, retrieved_sections)
    metrics = scored["metrics"]
    per_question = scored["per_question"]
    for row in per_question:
        rank = row["first_relevant_rank"]
        print(
            f"{'HIT' if rank else 'MISS'}@3 | {row['question']} "
            f"| rank={rank or '-'}"
        )
    print(f"\nQuestions: {metrics['question_count']}")
    print(f"Recall@1: {metrics['recall_at_1']:.3f}")
    print(f"Recall@3: {metrics['recall_at_3']:.3f}")
    print(f"MRR@3: {metrics['mean_reciprocal_rank_at_3']:.3f}")
    print("Metric scope: labeled section retrieval only; no answer generation is scored.")
    output = ROOT / "outputs"
    output.mkdir(exist_ok=True)
    with (output / "retrieval_question_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=per_question[0].keys())
        writer.writeheader()
        writer.writerows(per_question)
    (output / "retrieval_metrics.json").write_text(
        json.dumps(
            {
                **metrics,
                "top_k": args.top_k,
                "scope": "Section retrieval only; generated answers are not evaluated.",
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"Saved per-question results and metric summary under {output}")


if __name__ == "__main__":
    main()
