"""
ragas_compat.py
---------------
Import this BEFORE importing ragas:

    import ragas_compat  # noqa: F401   (must come before any "from ragas ..." line)

Why it exists
-------------
ragas 0.4.3 (src/ragas/llms/base.py) has two plain top-level imports:

    from langchain_community.chat_models.vertexai import ChatVertexAI
    from langchain_community.llms import VertexAI

It only uses them in a list of LLM classes that support multiple completions.
langchain-community 0.4.x no longer provides the first one (and may not provide
the second), so `import ragas` crashes with ModuleNotFoundError.

This project doesn't use Vertex AI, so empty stand-in classes are enough. If the
real classes are available, nothing is changed.

Delete this file once a ragas release stops importing them.
"""

import sys
import types


def _ensure_vertexai_stand_ins():
    # 1) langchain_community.chat_models.vertexai.ChatVertexAI
    try:
        import langchain_community.chat_models.vertexai  # noqa: F401
    except ImportError:
        module = types.ModuleType("langchain_community.chat_models.vertexai")

        class ChatVertexAI:  # stand-in; never used for real calls
            pass

        module.ChatVertexAI = ChatVertexAI
        sys.modules["langchain_community.chat_models.vertexai"] = module

    # 2) langchain_community.llms.VertexAI
    import langchain_community.llms as llms

    try:
        getattr(llms, "VertexAI")
    except (ImportError, AttributeError):

        class VertexAI:  # stand-in; never used for real calls
            pass

        llms.VertexAI = VertexAI


_ensure_vertexai_stand_ins()