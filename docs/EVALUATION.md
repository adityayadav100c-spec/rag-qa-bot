# Retrieval Evaluation

`data/retrieval_eval.jsonl` contains 12 questions with a human-assigned target
handbook section. Run `python evaluate_retrieval.py` to compute:

- **Recall@1**: fraction of queries whose target section appears first.
- **Recall@3**: fraction of queries whose target section appears in the top three
  chunks.
- **MRR@3**: mean reciprocal rank of the first chunk from the target section;
  misses contribute zero.

The target is a section, while search returns text chunks. A match therefore
shows that at least one chunk from the target section was retrieved, not that
the returned passage contains every detail required to answer the question.
The set is small and authored alongside the demo handbook; it is not an
independent benchmark. It does not grade generated answers, citation precision,
faithfulness, latency, or cost.

For a stronger evaluation, create a larger independently reviewed query set,
include hard negatives and unanswerable questions, track passage-level
relevance judgments, compare chunking and embedding configurations on the same
queries, and separately grade answer correctness and grounded citations.
