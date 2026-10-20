
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from dotenv import load_dotenv
from openai import AsyncOpenAI

from LLM.llm import LLMClient
from LLM.answer_generator import AnswerGenerator
from Retrieval.hybrid_retriever import HybridRetriever

import ragas_compat  # noqa: F401

from langchain_openai import OpenAIEmbeddings
from ragas.llms import llm_factory
from ragas.embeddings.base import embedding_factory
from ragas.metrics.collections import Faithfulness, AnswerCorrectness


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_QUESTIONS = PROJECT_ROOT / "eval" / "questions.json"
DEFAULT_DB = PROJECT_ROOT / "vector_store" / "chroma_db"

# Separate file: does not overwrite your previous 45-question experiment.
DEFAULT_OUTPUT = PROJECT_ROOT / "eval" / "generation_results_90.csv"

TOP_K = 5
MAX_RETRIES = 3

FIELDS = [
    "question",
    "generated_answer",
    "ground_truth",
    "faithfulness",
    "answer_correctness",
    "retrieved_contexts",
]


def load_questions(path):
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    return [
        q for q in data
        if q.get("pages") and q.get("ground_truth")
    ]


def build_retriever(db_path):
    return HybridRetriever(
        persist_directory=str(db_path),
        embedding_model=OpenAIEmbeddings(
            model="text-embedding-3-small"
        ),
    )


def build_evaluator():
    client = AsyncOpenAI()

    evaluator_llm = llm_factory(
        "gpt-4o-mini",
        client=client,
    )

    evaluator_embeddings = embedding_factory(
        "openai",
        model="text-embedding-3-small",
        client=client,
    )

    faithfulness = Faithfulness(llm=evaluator_llm)

    answer_correctness = AnswerCorrectness(
        llm=evaluator_llm,
        embeddings=evaluator_embeddings,
    )

    return faithfulness, answer_correctness


def load_existing(output_path):
    existing = {}

    if not output_path.exists():
        return existing

    with output_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as file:

        reader = csv.DictReader(file)

        for row in reader:
            existing[row["question"]] = row

    return existing


def save_checkpoint(output_path, rows):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    temp_path = output_path.with_suffix(".tmp")

    with temp_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=FIELDS,
        )

        writer.writeheader()

        for row in rows.values():
            writer.writerow({
                field: row.get(field, "")
                for field in FIELDS
            })

    temp_path.replace(output_path)


def has_score(value):
    if value is None or str(value).strip() == "":
        return False

    try:
        return float(value) >= 0
    except (ValueError, TypeError):
        return False


def score_with_retry(metric, kwargs, metric_name):
    for attempt in range(1, MAX_RETRIES + 1):

        try:
            result = metric.score(**kwargs)
            return result.value

        except Exception as error:
            print(
                f"\n{metric_name} failed "
                f"(attempt {attempt}/{MAX_RETRIES})"
            )
            print("Error:", repr(error))

            completion = getattr(error, "last_completion", None)

            if completion:
                print("Last completion:", completion)

            if attempt < MAX_RETRIES:
                wait = attempt * 5
                print(f"Retrying in {wait} seconds...")
                time.sleep(wait)

    return None


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate RAG generation quality."
    )

    parser.add_argument("--questions", default=str(DEFAULT_QUESTIONS))
    parser.add_argument("--db", default=str(DEFAULT_DB))
    parser.add_argument("--output", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--limit", type=int, default=None)

    args = parser.parse_args()

    load_dotenv()

    questions = load_questions(args.questions)

    if args.limit:
        questions = questions[:args.limit]

    output_path = Path(args.output)

    rows = load_existing(output_path)

    print(f"Loaded {len(questions)} answerable questions.")
    print(f"Existing checkpoint records: {len(rows)}")

    pending = [
        q for q in questions
        if not (
            q["question"] in rows
            and has_score(rows[q["question"]].get("faithfulness"))
            and has_score(rows[q["question"]].get("answer_correctness"))
        )
    ]

    completed = len(questions) - len(pending)

    print(f"Already completed: {completed}")
    print(f"Remaining to evaluate: {len(pending)}")

    if not pending:
        print("All requested questions are already evaluated.")
        return

    retriever = build_retriever(args.db)

    generator = AnswerGenerator(
        llm_client=LLMClient()
    )

    faithfulness_metric, correctness_metric = build_evaluator()

    for index, q in enumerate(questions, start=1):

        question = q["question"]
        ground_truth = q["ground_truth"]

        row = rows.get(question, {})

        faith_done = has_score(row.get("faithfulness"))
        correctness_done = has_score(row.get("answer_correctness"))

        if faith_done and correctness_done:
            print(f"\n[{index}/{len(questions)}] Already completed. Skipping.")
            continue

        print(f"\n[{index}/{len(questions)}] {question[:70]}")

        # Reuse saved answer and contexts if this question
        # was partially evaluated during a previous run.
        answer = row.get("generated_answer", "")
        contexts_raw = row.get("retrieved_contexts", "")

        if answer and contexts_raw:

            contexts = json.loads(contexts_raw)
            print("Reusing saved answer and retrieved contexts.")

        else:
            print("Retrieving documents...")

            results = retriever.retrieve(
                question,
                final_top_n=TOP_K,
            )

            contexts = [
                doc.page_content
                for _, doc, _ in results
            ]

            if not contexts:
                print("No contexts retrieved. Skipping.")
                continue

            print("Generating answer...")

            generated = generator.generate_from_retriever_results(
                user_query=question,
                retriever_results=results,
                use_history=False,
            )

            answer = generated["answer"]

        # Create or update checkpoint before metric evaluation.
        row.update({
            "question": question,
            "generated_answer": answer,
            "ground_truth": ground_truth,
            "retrieved_contexts": json.dumps(contexts),
            "faithfulness": row.get("faithfulness", ""),
            "answer_correctness": row.get("answer_correctness", ""),
        })

        rows[question] = row
        save_checkpoint(output_path, rows)

        # Faithfulness
        if not has_score(row.get("faithfulness")):

            faithfulness_score = score_with_retry(
                faithfulness_metric,
                {
                    "user_input": question,
                    "response": answer,
                    "retrieved_contexts": contexts,
                },
                "Faithfulness",
            )

            if faithfulness_score is not None:
                row["faithfulness"] = faithfulness_score
                print(f"Faithfulness: {faithfulness_score:.3f}")

            else:
                print("Faithfulness could not be evaluated.")

            save_checkpoint(output_path, rows)

        # Answer Correctness
        if not has_score(row.get("answer_correctness")):

            correctness_score = score_with_retry(
                correctness_metric,
                {
                    "user_input": question,
                    "response": answer,
                    "reference": ground_truth,
                },
                "Answer Correctness",
            )

            if correctness_score is not None:
                row["answer_correctness"] = correctness_score
                print(f"Answer Correctness: {correctness_score:.3f}")

            else:
                print("Answer Correctness could not be evaluated.")

            save_checkpoint(output_path, rows)

    # Final summary for the requested dataset.
    evaluated_rows = [
        rows[q["question"]]
        for q in questions
        if q["question"] in rows
    ]

    faithfulness_scores = [
        float(r["faithfulness"])
        for r in evaluated_rows
        if has_score(r.get("faithfulness"))
    ]

    correctness_scores = [
        float(r["answer_correctness"])
        for r in evaluated_rows
        if has_score(r.get("answer_correctness"))
    ]

    fully_completed = sum(
        1 for r in evaluated_rows
        if has_score(r.get("faithfulness"))
        and has_score(r.get("answer_correctness"))
    )

    print("\n========== GENERATION EVALUATION ==========")
    print(f"Questions in dataset: {len(questions)}")
    print(f"Fully evaluated: {fully_completed}")
    print(f"Faithfulness scores: {len(faithfulness_scores)}")
    print(f"Answer Correctness scores: {len(correctness_scores)}")

    if faithfulness_scores:
        print(
            "Average Faithfulness:",
            f"{sum(faithfulness_scores) / len(faithfulness_scores):.3f}",
        )

    if correctness_scores:
        print(
            "Average Answer Correctness:",
            f"{sum(correctness_scores) / len(correctness_scores):.3f}",
        )

    print(f"\nCheckpoint saved to: {output_path}")


if __name__ == "__main__":
    main()
