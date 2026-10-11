"""
answer_generator.py
--------------------
Top-level entry point for the generation stage of the RAG pipeline.
Given a user question + retrieved chunks (+ optional conversation
memory), it builds the prompt, calls the LLM, and returns an answer
with attached citations.

Typical usage:

    llm = LLMClient(model="gpt-4o-mini")
    chain = ConversationalChain(llm_client=llm)
    generator = AnswerGenerator(llm_client=llm, chain=chain)

    result = generator.generate(
        user_query="How does the retriever rank chunks?",
        retrieved_chunks=retrieved_chunks,  # from your retrieval module
    )
    print(result["answer"])
    print(result["citations"])
"""

from __future__ import annotations

from pathlib import Path

from LLM.llm import LLMClient
from LLM.citations import (
    Source,
    build_sources,
    build_sources_from_retriever_results,
    format_context_block,
    attach_citations,
)
from LLM.conversational_chain import ConversationalChain

PROMPTS_DIR = Path(__file__).resolve().parent.parent / "prompts"

_DEFAULT_SYSTEM_PROMPT = """You are a precise, helpful assistant answering questions using ONLY the \
provided context sources. Follow these rules strictly:

1. Base your answer only on the information in the numbered sources below.
2. Cite every factual claim with the matching [n] marker, e.g. "Revenue grew 12% [1]."
3. If multiple sources support a claim, cite all of them, e.g. [1][3].
4. If the sources don't contain enough information to answer, say so plainly \
instead of guessing.
5. Keep the answer focused and avoid repeating the question back."""


def _load_prompt(filename: str, default: str) -> str:
    """Load a prompt from /prompts if present, otherwise fall back to default."""
    path = PROMPTS_DIR / filename
    if path.exists():
        return path.read_text(encoding="utf-8").strip()
    return default


class AnswerGenerator:
    """
    Coordinates system prompt, context formatting, conversation history,
    and citation post-processing into one generation call.
    """

    def __init__(
        self,
        llm_client: LLMClient,
        chain: ConversationalChain | None = None,
        system_prompt: str | None = None,
    ):
        self.llm = llm_client
        self.chain = chain or ConversationalChain(llm_client=llm_client)
        self.system_prompt = system_prompt or _load_prompt(
            "system_prompt_v1.txt", _DEFAULT_SYSTEM_PROMPT
        )

    def generate(
        self,
        user_query: str,
        retrieved_chunks: list[dict],
        use_history: bool = True,
    ) -> dict:
        """
        Produce a grounded, cited answer.

        Parameters
        ----------
        user_query: the raw question as typed by the user this turn.
        retrieved_chunks: output of the retrieval module for this query
            (already rewritten/expanded upstream if needed).
        use_history: whether to include prior conversation turns in the
            prompt. Set False for stateless/single-shot calls.

        Returns
        -------
        {
            "answer": str,
            "citations": [{"index": int, "doc_id": str, "title": str|None}, ...],
            "usage": {"prompt_tokens": int, "completion_tokens": int},
        }
        """
        sources = build_sources(retrieved_chunks)
        return self._generate_from_sources(user_query, sources, use_history)

    def generate_from_retriever_results(
        self,
        user_query: str,
        retriever_results: list[tuple],
        use_history: bool = True,
    ) -> dict:
        """
        Same as `generate()`, but takes the raw output of
        HybridRetriever.retrieve() directly:
            [(doc_id, langchain_document, score), ...]

        This is the method to call from a script that does:
            results = retrieve(query)
            answer = generator.generate_from_retriever_results(query, results)
        """
        sources = build_sources_from_retriever_results(retriever_results)
        return self._generate_from_sources(user_query, sources, use_history)

    def _generate_from_sources(
        self,
        user_query: str,
        sources: list[Source],
        use_history: bool,
    ) -> dict:
        context_block = format_context_block(sources)

        messages = [{"role": "system", "content": self.system_prompt}]

        if use_history:
            messages.extend(self.chain.history_messages())

        messages.append(
            {
                "role": "user",
                "content": (
                    f"Context sources:\n{context_block}\n\n"
                    f"Question: {user_query}\n\n"
                    "Answer using the rules above, citing sources as [n]."
                ),
            }
        )

        response = self.llm.generate(messages)
        result = attach_citations(response.text, sources)
        result["usage"] = {
            "prompt_tokens": response.prompt_tokens,
            "completion_tokens": response.completion_tokens,
        }

        if use_history:
            self.chain.record_turn(user_query, response.text)

        return result

