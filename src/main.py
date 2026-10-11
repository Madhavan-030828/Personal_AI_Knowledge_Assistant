from Retrieval.retrieve import retrieve
from LLM.llm import LLMClient
from LLM.answer_generator import AnswerGenerator

MIN_RERANK_SCORE = 0.0   # placeholder: calibrate by printing results[0][2] for a few questions

llm = LLMClient(model="gpt-4o-mini")
generator = AnswerGenerator(llm_client=llm)

while True:
    try:
        query = input("\nYou: ").strip()
        if not query:
            continue
        if query.lower() == "exit":
            break

        # Resolve follow-ups ("what about its pricing?") into a standalone search query
        search_query = generator.chain.rewrite_query(query)
        results = retrieve(search_query)

        print("search_query:", search_query)
        print("num results:", len(results))
        for doc_id, doc, score in results:
            print(round(score, 3), "|", doc.metadata.get("source"), "| page", doc.metadata.get("page"))
            print("   ", doc.page_content[:100].replace("\n", " "))

        if not results or results[0][2] < MIN_RERANK_SCORE:
            print("\nAssistant:\nI couldn't find this in your documents.")
            continue

        output = generator.generate_from_retriever_results(query, results)

        print("\nAssistant:\n")
        print(output["answer"])

        print("\nSources:")
        for c in output["citations"]:
            print(f"[{c['index']}] {c['title']}")

    except KeyboardInterrupt:
        break
    except Exception as e:
        print(f"\nSomething went wrong: {e}")