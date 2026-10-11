"""
conversational_chain.py
------------------------
Manages multi-turn conversation state for the RAG chat: history storage,
trimming to fit context window, and turning history + the new user
question into a standalone query for retrieval (handles follow-ups like
"what about its pricing?").
"""

from __future__ import annotations

from dataclasses import dataclass, field

from LLM.llm import LLMClient


@dataclass
class Turn:
    role: str  # "user" | "assistant"
    content: str


@dataclass
class ConversationMemory:
    """
    Simple in-memory turn buffer per session. Swap `store` for Redis /
    a DB-backed dict in production — the interface stays the same.
    """
    max_turns: int = 12  # keep last N turns (user+assistant pairs count separately)
    turns: list[Turn] = field(default_factory=list)

    def add(self, role: str, content: str) -> None:
        self.turns.append(Turn(role=role, content=content))
        if len(self.turns) > self.max_turns:
            self.turns = self.turns[-self.max_turns:]

    def as_messages(self) -> list[dict[str, str]]:
        return [{"role": t.role, "content": t.content} for t in self.turns]

    def clear(self) -> None:
        self.turns = []


_REWRITE_SYSTEM_PROMPT = (
    "You rewrite a user's latest message into a standalone search query, "
    "resolving any pronouns or implicit references using the chat history. "
    "Output ONLY the rewritten query, no explanation, no quotes."
)


class ConversationalChain:
    """
    Ties conversation memory to query rewriting, so the retriever always
    gets a self-contained query even when the user asks a follow-up.
    """

    def __init__(self, llm_client: LLMClient, memory: ConversationMemory | None = None):
        self.llm = llm_client
        self.memory = memory or ConversationMemory()

    def rewrite_query(self, user_message: str) -> str:
        """
        Condense chat history + latest message into a standalone query
        for the retriever. Skips the LLM call entirely on the first turn.
        """
        if not self.memory.turns:
            return user_message

        history_text = "\n".join(f"{t.role}: {t.content}" for t in self.memory.turns[-6:])
        messages = [
            {"role": "system", "content": _REWRITE_SYSTEM_PROMPT},
            {
                "role": "user",
                "content": f"Chat history:\n{history_text}\n\nLatest message: {user_message}",
            },
        ]
        response = self.llm.generate(messages, temperature=0.0, max_tokens=100)
        rewritten = response.text.strip()
        return rewritten or user_message

    def record_turn(self, user_message: str, assistant_message: str) -> None:
        self.memory.add("user", user_message)
        self.memory.add("assistant", assistant_message)

    def history_messages(self) -> list[dict[str, str]]:
        return self.memory.as_messages()
