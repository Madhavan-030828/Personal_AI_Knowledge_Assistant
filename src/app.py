"""
app.py
------
Streamlit chat UI for the personal AI knowledge assistant.

Run from the PROJECT ROOT (the folder that contains Data/ and vector_store/):

    streamlit run src/app.py

It reuses the existing pipeline unchanged:
    retrieve()  ->  AnswerGenerator.generate_from_retriever_results()
"""

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Notes Assistant",
    page_icon="📓",
    layout="centered",
)

NOT_FOUND_MESSAGE = (
    "I couldn't find this in your documents. "
    "Try rephrasing, or ask about a topic your PDFs cover."
)

EXAMPLE_QUESTIONS = [
    "What is Java?",
    "Explain abstraction in Java",
    "What are the features of Java?",
    "Formula for profit percentage",
]


# --------------------------------------------------------------------------
# Loading (heavy objects are created once)
# --------------------------------------------------------------------------
@st.cache_resource(show_spinner="Loading your knowledge base...")
def load_retriever():
    """Importing retrieve.py builds the hybrid retriever (Chroma + BM25 + reranker) once."""
    from Retrieval.retrieve import retrieve
    return retrieve


def new_generator():
    """One generator per browser session, so each session has its own chat memory."""
    from LLM.llm import LLMClient
    from LLM.answer_generator import AnswerGenerator

    return AnswerGenerator(llm_client=LLMClient(model="gpt-4o-mini"))


def init_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "generator" not in st.session_state:
        st.session_state.generator = new_generator()


def clear_chat():
    st.session_state.messages = []
    st.session_state.generator.chain.memory.clear()


# --------------------------------------------------------------------------
# Question answering
# --------------------------------------------------------------------------
def answer_question(query: str, min_score: float):
    """Returns (answer_text, sources) where sources is a list of dicts for display."""
    retrieve = load_retriever()
    generator = st.session_state.generator

    # Turn follow-ups ("what about its types?") into a standalone search query
    search_query = generator.chain.rewrite_query(query)
    results = retrieve(search_query)

    if not results or results[0][2] < min_score:
        return NOT_FOUND_MESSAGE, []

    output = generator.generate_from_retriever_results(query, results)

    sources = []
    for c in output["citations"]:
        _, doc, score = results[c["index"] - 1]   # sources are numbered in retrieval order
        sources.append(
            {
                "index": c["index"],
                "title": c["title"],
                "score": score,
                "snippet": doc.page_content.strip(),
            }
        )
    return output["answer"], sources


def render_sources(sources: list[dict]):
    if not sources:
        return
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**[{s['index']}] {s['title']}**")
            snippet = s["snippet"]
            st.caption(snippet[:500] + ("..." if len(snippet) > 500 else ""))


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
init_state()

with st.sidebar:
    st.header("Notes Assistant")
    st.write(
        "Ask questions about your PDF notes. Answers come only from your "
        "documents, with the source page cited."
    )
    st.divider()
    min_score = st.slider(
        "Relevance cutoff",
        min_value=-10.0,
        max_value=10.0,
        value=0.0,
        step=0.5,
        help=(
            "If the best match scores below this, the assistant says it couldn't "
            "find the answer instead of guessing. Raise it to be stricter."
        ),
    )
    st.button("Clear chat", on_click=clear_chat, use_container_width=True)

st.title("Ask your notes")

# Past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        render_sources(message.get("sources", []))

# Empty state: offer starting points
pending = None
if not st.session_state.messages:
    st.write("Pick a question to start, or type your own below.")
    cols = st.columns(2)
    for i, question in enumerate(EXAMPLE_QUESTIONS):
        if cols[i % 2].button(question, use_container_width=True, key=f"example_{i}"):
            pending = question

typed = st.chat_input("Ask a question about your PDFs")
prompt = typed or pending

if prompt:
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Searching your notes..."):
            try:
                answer, sources = answer_question(prompt, min_score)
            except Exception as e:
                answer = (
                    f"Something went wrong while answering: {e}\n\n"
                    "Check that your OpenAI key is set in `.env` and that "
                    "`vector_store/chroma_db` exists (run ingestion first)."
                )
                sources = []
        st.markdown(answer)
        render_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )