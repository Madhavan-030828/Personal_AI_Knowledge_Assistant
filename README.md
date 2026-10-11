# Personal AI Knowledge Assistant

A retrieval-augmented generation (RAG) system that answers questions from your own PDF notes, **including scanned and handwritten pages**, and cites the source file and page for every answer. When the answer isn't in your documents, it says so instead of guessing.

> _Add a screenshot or short GIF of the app here: `docs/demo.png`_

## Highlights

- **Scanned and handwritten PDF support.** Pages with no text layer are transcribed by a vision model (`gpt-4o-mini`) and cached on disk, so each page is only paid for once.
- **Hybrid retrieval.** BM25 keyword search and vector search are merged with Reciprocal Rank Fusion, then a cross-encoder reranker picks the final chunks.
- **Semantic chunking.** Chunks split where the topic changes instead of at fixed sizes.
- **Cited answers.** Every claim carries a `[n]` marker that maps back to a file name, page number, and the original text.
- **Follow-up questions.** Chat history plus query rewriting turns "what about its types?" into a standalone search.
- **"Not found" handling.** A relevance cutoff on the reranker score stops the model from answering when nothing relevant was retrieved.
- **Streamlit chat UI** and a command-line chat.

## Architecture

```
INGESTION (run once per document set)

  PDF pages
     |
     +-- has text layer? --yes--> extract text
     |
     +-- no (scanned / handwritten) --> render page image --> vision model transcribes --> cache (JSON)
                                                                 |
                                                                 v
                      semantic chunking --> OpenAI embeddings --> ChromaDB (with source + page metadata)


QUERY

  question
     |
     v
  query rewrite (uses chat history)
     |
     +--> BM25 keyword search (top 20) ---+
     |                                    +--> Reciprocal Rank Fusion --> top 15
     +--> vector search (top 20) ---------+                                  |
                                                                             v
                                                            cross-encoder rerank --> top 5 chunks
                                                                             |
                                                relevance cutoff? --below--> "not found in your documents"
                                                                             |
                                                                           above
                                                                             v
                                                       LLM answer with [n] citations --> answer + sources
```

## Design decisions

| Decision | Why |
|---|---|
| Hybrid search (BM25 + vectors) | Vector search handles paraphrases but can miss exact terms, names, and formulas. BM25 does the opposite. Together they cover each other's gaps. |
| Reciprocal Rank Fusion | BM25 scores and cosine distances are on different scales. RRF uses only rank positions, so no score normalization is needed. |
| Cross-encoder reranking | Reads the query and chunk together, which is more accurate than comparing embeddings. It is too slow for the whole corpus, so it only scores a shortlist of 15. |
| Vision-model OCR with caching | Scanned and handwritten pages have no text layer, and traditional OCR struggles with handwriting. Caching avoids paying again on re-ingestion or crashes. |
| Semantic chunking | Fixed-size chunks cut ideas in half. Splitting at topic changes keeps each chunk about one thing. |
| Relevance cutoff | The assistant returns "not found" when the best reranker score is too low, which reduces hallucination. |

## Tech stack

Python, OpenAI API (`gpt-4o-mini`, `text-embedding-3-small`), LangChain, ChromaDB, rank-bm25, sentence-transformers (`cross-encoder/ms-marco-MiniLM-L-6-v2`), PyMuPDF, Streamlit

## Project structure

```
src/
  Ingestion/     PDF loading with OCR fallback, semantic chunking, embedding, Chroma indexing
  Retrieval/     BM25, vector search, rank fusion, reranker, hybrid retriever
  LLM/           OpenAI client, answer generation, citations, conversation memory
  app.py         Streamlit UI
  main.py        Command-line chat
Data/
  Raw/           Your PDFs (not committed)
  ocr_cache/     Cached OCR text per PDF (not committed)
vector_store/    ChromaDB index (not committed)
```

## Getting started

Requires Python 3.10+ and an OpenAI API key.

```bash
git clone <your-repo-url>
cd <repo-folder>

python -m venv .venv
.venv\Scripts\activate          # Windows
# source .venv/bin/activate     # macOS / Linux

pip install -r requirements.txt
```

Create a `.env` file in the project root:

```
OPENAI_API_KEY=your-key-here
```

### 1. Add your documents

Put your PDFs in `Data/Raw/`.

### 2. Build the index

```bash
python src/Ingestion/Ingest.py
```

- The first run OCRs any scanned pages, which takes a few minutes per large PDF and uses the OpenAI API. Pages are cached in `Data/ocr_cache/`, so re-runs are fast.
- Ingestion prints progress per page and ends with `Indexed N chunks successfully.`
- To rebuild from scratch, delete `vector_store/chroma_db` first. Otherwise new chunks are appended to the existing index and you get duplicates.

### 3. Ask questions

Web UI:

```bash
python -m streamlit run src/app.py
```

Command line:

```bash
python src/main.py
```

Run both commands from the project root, because the paths are relative.

## Using the UI

- Ask a question, or click a starter question.
- Expand **Sources** under an answer to see the file, page, and the retrieved text behind each citation.
- The **Relevance cutoff** slider in the sidebar controls how strict the "not found" behavior is. Raise it to make the assistant more cautious.
- **Clear chat** resets the conversation and its memory.

## Troubleshooting

| Problem | Fix |
|---|---|
| `ZeroDivisionError` when starting the app | The vector store is empty. Run ingestion to completion first. |
| OCR stops with `429` rate-limit errors | You hit the OpenAI tokens-per-minute limit. The loader retries and paces requests; if it still fails, run ingestion again. Cached pages are skipped. |
| `getaddrinfo failed` for `huggingface.co` on startup | The reranker model is checked against the Hugging Face Hub at startup. Connect to the internet, or set `HF_HUB_OFFLINE=1` after the model has been downloaded once. |
| Every answer says "couldn't find this in your documents" | Lower the relevance cutoff, and check that chunks aren't blank (empty text from a scanned PDF means OCR didn't run). |
| Duplicate sources in answers | The index was built more than once. Delete `vector_store/chroma_db` and re-ingest. |

## Limitations

- Handwriting transcription depends on legibility. Spot-check a few pages in `Data/ocr_cache/` after ingesting.
- Chunks are built per page, so a topic that spans two pages can be split.
- Citation page numbers are PDF viewer page numbers, which may differ from page numbers written on the pages.
- Retrieval quality has been tested manually but not yet measured on a labeled evaluation set.
- The app needs internet access. Embeddings, OCR, and answer generation use the OpenAI API, and the reranker model is fetched from Hugging Face on first use.
- Page images and text are sent to the OpenAI API. Don't index documents you aren't comfortable sending there.

## Roadmap

- [ ] Evaluation set (hit rate and MRR) comparing vector-only, hybrid, and hybrid + rerank
- [ ] Overlap between pages when chunking, and metadata filters (search within one PDF)
- [ ] Incremental indexing with deterministic chunk IDs, so re-ingesting never duplicates
- [ ] Docker image and a hosted demo with public-domain sample PDFs
- [ ] Local mode with open-source models for fully offline use

## Author

**Madhavan K** · [LinkedIn](https://www.linkedin.com/in/madhavan-k-6a4468330)
