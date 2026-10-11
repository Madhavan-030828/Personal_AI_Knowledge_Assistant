"""
llm.py
------
Thin wrapper around the OpenAI Chat Completions API.

Keeps provider-specific details (client setup, retries, token limits)
isolated from the rest of the RAG pipeline so the rest of the codebase
only ever talks to `LLMClient`.
"""

from __future__ import annotations

import os
import time
import logging
from dataclasses import dataclass
from typing import Any

from openai import OpenAI, APIError, RateLimitError, APITimeoutError

logger = logging.getLogger(__name__)


@dataclass
class LLMResponse:
    """Normalized response returned by LLMClient.generate()."""
    text: str
    model: str
    prompt_tokens: int
    completion_tokens: int
    finish_reason: str
    raw: Any = None  # original SDK response, kept for debugging


class LLMClient:
    """
    Wraps the OpenAI API for chat-style generation.

    Parameters
    ----------
    model: str
        Model name, e.g. "gpt-4o" or "gpt-4o-mini".
    api_key: str | None
        Falls back to OPENAI_API_KEY env var if not provided.
    temperature: float
        Sampling temperature. Keep low (0-0.3) for grounded RAG answers.
    max_retries: int
        Number of retries on transient errors (rate limit / timeout).
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        api_key: str | None = None,
        temperature: float = 0.2,
        max_tokens: int = 800,
        max_retries: int = 3,
        timeout: float = 30.0,
    ) -> None:
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.client = OpenAI(
            api_key=api_key or os.environ.get("OPENAI_API_KEY"),
            timeout=timeout,
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
        response_format: dict | None = None,
    ) -> LLMResponse:
        """
        Send a chat completion request with exponential-backoff retries.

        `messages` follows the standard OpenAI format:
            [{"role": "system"|"user"|"assistant", "content": "..."}]
        """
        last_err: Exception | None = None

        for attempt in range(1, self.max_retries + 1):
            try:
                kwargs: dict[str, Any] = dict(
                    model=self.model,
                    messages=messages,
                    temperature=temperature if temperature is not None else self.temperature,
                    max_tokens=max_tokens or self.max_tokens,
                )
                if response_format:
                    kwargs["response_format"] = response_format

                resp = self.client.chat.completions.create(**kwargs)
                choice = resp.choices[0]

                return LLMResponse(
                    text=choice.message.content or "",
                    model=resp.model,
                    prompt_tokens=resp.usage.prompt_tokens if resp.usage else 0,
                    completion_tokens=resp.usage.completion_tokens if resp.usage else 0,
                    finish_reason=choice.finish_reason,
                    raw=resp,
                )

            except (RateLimitError, APITimeoutError) as e:
                last_err = e
                wait = 2 ** attempt
                logger.warning(
                    "LLM call failed (attempt %d/%d): %s — retrying in %ds",
                    attempt, self.max_retries, e, wait,
                )
                time.sleep(wait)
            except APIError as e:
                # Non-transient API error — don't retry.
                logger.error("Non-retryable LLM API error: %s", e)
                raise

        raise RuntimeError(f"LLM call failed after {self.max_retries} retries") from last_err
