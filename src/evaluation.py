"""
evaluate.py
-----------
Measures retrieval quality on a hand-labelled question set.

Run from the PROJECT ROOT (the folder that contains Data/, src/, vector_store/):

    python src/evaluate.py                       # all questions, 4 setups
    python src/evaluate.py --limit 5 --verbose   # quick smoke test
    python src/evaluate.py --cutoff -1.0         # try a different "not found" cutoff

It only READS your existing Chroma index. It changes nothing.

Question file format (eval/questions.json):

    [
      {"question": "What is Java?", "file": "JAVA NOTES.pdf", "pages": [1], "type": "exact"},
      {"question": "Who won the 2018 football world cup?", "type": "out_of_scope"}
    ]

  - "pages" are the page numbers you see in your PDF viewer (the same p.N your
    citations show). List more than one if the answer spans pages.
  - A question with no "pages" is treated as out-of-scope (the answer is NOT in
    your documents), and is used to check the "not found" cutoff.
  - "type" is optional; it is only used to break results down (for example
    "exact" vs "paraphrase").
"""

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = PROJECT_ROOT / "eval" / "questions.json"
DEFAULT_DB = PROJECT_ROOT / "vector_store" / "chroma_db"

TOP_K = 5                      # number of chunks the real system passes to the LLM
RERANK = "Hybrid + rerank"     # the setup that produces comparable scores for the cutoff


# --------------------------------------------------------------------------
# Scoring helpers (no API calls, easy to test)
# --------------------------------------------------------------------------
def file_and_page(doc):
    """(file name, 1-based page) of a chunk, matching the p.N shown in citations."""
    meta = doc.metadata or {}
    name = Path(str(meta.get("source", "")).replace("\\", "/")).name
    page = meta.get("page")
    return name, (page + 1 if isinstance(page, int) else None)


def first_hit_rank(results, gold_file, gold_pages):
    """1-based rank of the first retrieved chunk that sits on a gold page, else None."""
    for rank, (_, doc, _) in enumerate(results, start=1):
        name, page = file_and_page(doc)
        if name == gold_file and page in gold_pages:
            return rank
    return None


def summarize(ranks):
    """(hit rate, MRR) from a list of ranks, where None means a miss."""
    if not ranks:
        return 0.0, 0.0
    hits = sum(1 for r in ranks if r is not None)
    mrr = sum(1.0 / r for r in ranks if r is not None) / len(ranks)
    return hits / len(ranks), mrr


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------
def load_questions(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    for i, q in enumerate(data, start=1):
        if "question" not in q:
            raise ValueError(f"Question #{i} has no 'question' field")
        if q.get("pages") and not q.get("file"):
            raise ValueError(f"Question #{i} has 'pages' but no 'file'")
    return data


def build_retriever(db_path):
    # Heavy imports live here so the scoring helpers above can be tested without them.
    from dotenv import load_dotenv
    from langchain_openai import OpenAIEmbeddings
    from Retrieval.hybrid_retriever import HybridRetriever

    load_dotenv()
    return HybridRetriever(
        persist_directory=str(db_path),
        embedding_model=OpenAIEmbeddings(model="text-embedding-3-small"),
    )


def build_setups(hr):
    """The four retrieval strategies to compare. Each takes a query, returns [(id, doc, score)]."""

    def bm25(q):
        return hr.bm25_retriever.search(q, k=TOP_K)

    def vector(q):
        return hr.vector_retriever.search(q, k=TOP_K)

    def hybrid(q):
        fused = hr.fusion.fuse(
            [hr.bm25_retriever.search(q, k=20), hr.vector_retriever.search(q, k=20)]
        )
        return fused[:TOP_K]

    def hybrid_rerank(q):
        return hr.retrieve(q, final_top_n=TOP_K)

    return {
        "BM25 only": bm25,
        "Vector only": vector,
        "Hybrid (RRF)": hybrid,
        RERANK: hybrid_rerank,
    }


# --------------------------------------------------------------------------
# Running and reporting
# --------------------------------------------------------------------------
def run_setups(setups, questions):
    all_results = {name: [] for name in setups}
    for n, q in enumerate(questions, start=1):
        print(f"  [{n}/{len(questions)}] {q['question'][:70]}")
        for name, fn in setups.items():
            all_results[name].append(fn(q["question"]))
    return all_results


def report(setups, questions, all_results, cutoff, verbose):
    in_idx = [i for i, q in enumerate(questions) if q.get("pages")]
    out_idx = [i for i, q in enumerate(questions) if not q.get("pages")]

    ranks = {
        name: {
            i: first_hit_rank(
                all_results[name][i], questions[i]["file"], set(questions[i]["pages"])
            )
            for i in in_idx
        }
        for name in setups
    }

    # ---- headline table
    print(f"\n### Retrieval quality ({len(in_idx)} answerable questions, top {TOP_K} chunks)\n")
    print(f"| Setup | Hit@{TOP_K} | MRR |")
    print("|---|---|---|")
    for name in setups:
        hit, mrr = summarize(list(ranks[name].values()))
        print(f"| {name} | {hit:.0%} | {mrr:.2f} |")

    # ---- breakdown by question type
    types = Counter(questions[i].get("type", "other") for i in in_idx)
    if len(types) > 1:
        print(f"\n### Hit@{TOP_K} by question type\n")
        header = " | ".join(f"{t} ({c})" for t, c in sorted(types.items()))
        print(f"| Setup | {header} |")
        print("|---|" + "---|" * len(types))
        for name in setups:
            cells = []
            for t in sorted(types):
                subset = [ranks[name][i] for i in in_idx if questions[i].get("type", "other") == t]
                cells.append(f"{summarize(subset)[0]:.0%}")
            print(f"| {name} | " + " | ".join(cells) + " |")

    # ---- "not found" cutoff check (uses reranker scores, which are comparable)
    def top_score(i):
        res = all_results[RERANK][i]
        return res[0][2] if res else float("-inf")

    if in_idx or out_idx:
        print(f"\n### 'Not found' cutoff check (cutoff = {cutoff})\n")
        if out_idx:
            rejected = sum(1 for i in out_idx if top_score(i) < cutoff)
            print(f"- Out-of-scope questions correctly rejected: {rejected}/{len(out_idx)}")
        else:
            print("- No out-of-scope questions in the file, so rejection can't be checked.")
        if in_idx:
            wrongly = sum(1 for i in in_idx if top_score(i) < cutoff)
            print(f"- Answerable questions wrongly rejected: {wrongly}/{len(in_idx)}")
            scores = [top_score(i) for i in in_idx]
            print(f"- Top reranker score, answerable: min {min(scores):.2f}, "
                  f"median {statistics.median(scores):.2f}")
        if out_idx:
            print(f"- Top reranker score, out-of-scope: max {max(top_score(i) for i in out_idx):.2f}")
        print("- A good cutoff sits above the out-of-scope max and below the answerable min.")

    # ---- per-question detail
    if verbose:
        print("\n### Per-question ranks (MISS = no gold page in the top chunks)\n")
        for i in in_idx:
            parts = []
            for name in setups:
                r = ranks[name][i]
                parts.append(f"{name}: {'MISS' if r is None else r}")
            print(f"- {questions[i]['question'][:60]}\n    " + " | ".join(parts))
            if ranks[RERANK][i] is None:
                got = [file_and_page(doc) for _, doc, _ in all_results[RERANK][i][:3]]
                want = f"{questions[i]['file']} p.{questions[i]['pages']}"
                print(f"    expected {want}; got " + ", ".join(f"{f} p.{p}" for f, p in got))


def main():
    parser = argparse.ArgumentParser(description="Evaluate retrieval quality.")
    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--cutoff", type=float, default=0.0,
                        help="relevance cutoff to test for the 'not found' check")
    parser.add_argument("--limit", type=int, default=None, help="only run the first N questions")
    parser.add_argument("--verbose", action="store_true", help="print per-question ranks and misses")
    args = parser.parse_args()

    questions = load_questions(args.questions)
    if args.limit:
        questions = questions[: args.limit]

    print(f"Loading retriever from {args.db} ...")
    setups = build_setups(build_retriever(args.db))

    print(f"Running {len(questions)} questions through {len(setups)} setups ...")
    all_results = run_setups(setups, questions)
    report(setups, questions, all_results, args.cutoff, args.verbose)


if __name__ == "__main__":
    main()