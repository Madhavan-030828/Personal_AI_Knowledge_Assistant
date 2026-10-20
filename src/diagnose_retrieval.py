
import argparse
import csv
import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_QUESTIONS = PROJECT_ROOT / "eval" / "questions.json"
DEFAULT_DB = PROJECT_ROOT / "vector_store" / "chroma_db"

TOP_K = 5
CUTOFF = 0.0


def load_questions(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_retriever(db_path):
    from dotenv import load_dotenv
    from langchain_openai import OpenAIEmbeddings
    from Retrieval.hybrid_retriever import HybridRetriever

    load_dotenv()

    return HybridRetriever(
        persist_directory=str(db_path),
        embedding_model=OpenAIEmbeddings(
            model="text-embedding-3-small"
        ),
    )


def file_and_page(doc):
    metadata = doc.metadata or {}

    filename = Path(
        str(metadata.get("source", "")).replace("\\", "/")
    ).name

    page = metadata.get("page")

    if isinstance(page, int):
        page += 1

    return filename, page


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--questions",
        default=str(DEFAULT_QUESTIONS)
    )

    parser.add_argument(
        "--db",
        default=str(DEFAULT_DB)
    )

    parser.add_argument(
        "--cutoff",
        type=float,
        default=CUTOFF
    )

    args = parser.parse_args()

    questions = load_questions(args.questions)

    # Only answerable questions
    questions = [
        q for q in questions
        if q.get("pages") and q.get("file")
    ]

    print("Loading HybridRetriever...")

    retriever = build_retriever(args.db)

    failed_questions = []
    csv_rows = []

    for index, q in enumerate(questions, start=1):

        question = q["question"]
        expected_file = q["file"]
        expected_pages = set(q["pages"])

        results = retriever.retrieve(
            question,
            final_top_n=TOP_K
        )

        top_score = results[0][2] if results else float("-inf")

        # Identify false rejection
        if top_score >= args.cutoff:
            continue

        failed_questions.append(q)

        print("\n" + "=" * 80)
        print(f"QUESTION #{index}: {question}")

        print(f"Expected file: {expected_file}")
        print(f"Expected pages: {sorted(expected_pages)}")
        print(f"Top reranker score: {top_score:.4f}")

        gold_page_found = False

        for rank, (doc_id, doc, score) in enumerate(
            results, start=1
        ):

            filename, page = file_and_page(doc)

            is_relevant = (
                filename == expected_file
                and page in expected_pages
            )

            if is_relevant:
                gold_page_found = True

            print(f"\nRank: {rank}")
            print(f"Score: {score:.4f}")
            print(f"File: {filename}")
            print(f"Page: {page}")
            print(f"Expected page match: {is_relevant}")
            print(f"Content:\n{doc.page_content[:600]}")

            csv_rows.append({
                "question": question,
                "expected_file": expected_file,
                "expected_pages": str(sorted(expected_pages)),
                "top_score": top_score,
                "rank": rank,
                "retrieved_file": filename,
                "retrieved_page": page,
                "reranker_score": score,
                "expected_page_match": is_relevant,
                "content": doc.page_content[:600]
            })

        print("\nExpected page found in top 5:", gold_page_found)

    output_path = PROJECT_ROOT / "eval" / "false_rejections.csv"

    with output_path.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as file:

        if csv_rows:
            writer = csv.DictWriter(
                file,
                fieldnames=csv_rows[0].keys()
            )
            writer.writeheader()
            writer.writerows(csv_rows)

    print("\n" + "=" * 80)
    print(f"False rejections found: {len(failed_questions)}")
    print(f"Diagnostic CSV saved to: {output_path}")


if __name__ == "__main__":
    main()
