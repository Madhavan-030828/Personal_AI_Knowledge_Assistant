# Personal AI Knowledge Assistant

A retrieval-augmented generation (RAG) system that answers questions from your own PDF notes, **including scanned and handwritten pages**, and cites the source file and page for every answer. When the answer isn't in your documents, it says so instead of guessing.

<img width="1920" height="1080" alt="demo" src="https://github.com/user-attachments/assets/b34c67df-e55a-4aa6-8c65-15ced8f9449a" />

## Highlights

- **Scanned and handwritten PDF support.** Pages with no text layer are transcribed by a vision model (`gpt-4o-mini`) and cached on disk, so each page is only paid for once.
- **Multiple retrieval strategies.** BM25 keyword search, vector search, and hybrid retrieval using Reciprocal Rank Fusion (RRF).
- **Cross-encoder reranking.** Retrieved candidates are re-scored using query-document relevance to improve ranking.
- **Semantic chunking.** Chunks split where the topic changes instead of at fixed sizes.
- **Cited answers.** Every claim carries a `[n]` marker that maps back to a file name, page number, and the original text.
- **Follow-up questions.** Chat history plus query rewriting turns "what about its types?" into a standalone search.
- **"Not found" handling.** A relevance cutoff on the reranker score stops the model from answering when nothing relevant was retrieved.
- **Retrieval evaluation.** A labeled evaluation dataset is used to compare BM25, vector search, hybrid retrieval, and hybrid retrieval with reranking.
- **Streamlit chat UI** and a command-line chat.

## Architecture

```
INGESTION (run once per document set)

  PDF pages
     |
     +-- has text layer? --yes--> extract text
     |
     +-- no (scanned / handwritten) --> render page image
                                         |
                                      vision model
                                         |
                                    transcribe --> cache (JSON)
                                         |
                                         v
                      semantic chunking --> OpenAI embeddings
                                         |
                                         v
                                ChromaDB (source + page metadata)


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
                                                            cross-encoder rerank --> top 5
                                                                             |
                                                relevance cutoff? --below--> "not found"
                                                                             |
                                                                           above
                                                                             v
                                                       LLM answer with [n] citations
                                                                             |
                                                                             v
                                                                  answer + sources
```

## Design decisions

| Decision | Why |
|---|---|
| Hybrid search (BM25 + vectors) | Vector search handles semantic similarity, while BM25 captures exact keywords, names, and technical terms. |
| Reciprocal Rank Fusion | BM25 scores and vector similarity scores are on different scales. RRF combines rankings without requiring score normalization. |
| Cross-encoder reranking | Evaluates the query and retrieved chunk together, providing a more detailed relevance score than embedding similarity alone. |
| Vision-model OCR with caching | Scanned and handwritten pages have no reliable text layer. Caching avoids repeating OCR on re-ingestion or crashes. |
| Semantic chunking | Fixed-size chunks can split related ideas. Semantic splitting helps preserve contextual meaning. |
| Relevance cutoff | The assistant returns "not found" when the best reranker score is below a configured threshold, reducing unsupported answers. |
| Retrieval evaluation | Comparing multiple retrieval strategies on a labeled dataset helps identify which approach works best for the actual document collection. |

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
  evaluation.py  Retrieval evaluation pipeline

Data/
  Raw/           Your PDFs (not committed)
  ocr_cache/     Cached OCR text per PDF (not committed)

eval/
  questions.json Labeled evaluation dataset

vector_store/    ChromaDB index (not committed)
```

## Evaluation

A labeled evaluation dataset containing **50 questions** was used to assess retrieval quality and out-of-scope detection.

The dataset includes:
- 45 answerable questions covering different question types.
- 5 out-of-scope questions to test the system's ability to reject unsupported queries.

### Retrieval quality

Retrieval strategies were compared using Hit Rate@5 and Mean Reciprocal Rank (MRR).

| Retrieval strategy | Hit@5 | MRR |
|---|---:|---:|
| BM25 only | 62% | 0.48 |
| Vector only | **96%** | **0.72** |
| Hybrid (RRF) | 80% | 0.54 |
| Hybrid + Reranking | 91% | 0.65 |

**Key observations:**

- Vector-only retrieval achieved the best overall results, with 96% Hit@5 and 0.72 MRR.
- BM25 alone achieved 62% Hit@5, showing limitations when relying exclusively on keyword matching.
- Hybrid retrieval using RRF did not outperform vector-only retrieval on this dataset.
- Adding reranking improved hybrid retrieval from 80% to 91% Hit@5.
- Combining multiple retrieval techniques does not automatically guarantee better results; performance depends on the dataset, candidate selection, fusion strategy, and ranking configuration.

### Out-of-scope detection

The relevance cutoff was evaluated using 5 out-of-scope and 45 answerable questions.

| Measurement | Result |
|---|---:|
| Out-of-scope questions correctly rejected | 5/5 |
| Answerable questions incorrectly rejected | 3/45 |
| Cutoff used | 0.0 |

The experiment correctly rejected all five out-of-scope questions, but also incorrectly rejected three answerable questions.

This highlights the trade-off between rejecting unsupported questions and avoiding false rejections. The cutoff should be validated on additional data before being considered reliable for general use.

### Evaluation methodology

- **Hit Rate@5:** Measures whether at least one relevant chunk appears in the top five retrieved results.
- **MRR:** Measures how highly the first relevant chunk is ranked. Higher values indicate better ranking performance.
- **Out-of-scope rejection:** Measures whether unsupported questions are correctly rejected instead of receiving an unsupported answer.

The evaluation script reads the existing ChromaDB index and compares retrieval configurations without modifying the index.

## Getting started

Requires Python 3.10+ and an OpenAI API key.

```bash
git clone https://github.com/Madhavan-030828/Personal_AI_Knowledge_Assistant.git
cd Personal_AI_Knowledge_Assistant

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

### 4. Run retrieval evaluation

```bash
python src/evaluation.py --verbose
```

For a quick test:

```bash
python src/evaluation.py --limit 5 --verbose
```

The evaluation script compares retrieval strategies against the labeled question set.

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
- Retrieval evaluation currently covers 45 answerable questions and 5 out-of-scope questions. Results may vary with larger and more diverse datasets.
- The best-performing retrieval strategy on the current evaluation set is vector-only retrieval, although hybrid retrieval and reranking remain available.
- The relevance cutoff incorrectly rejected 3 of the 45 answerable questions in the current experiment.
- The app needs internet access. Embeddings, OCR, and answer generation use the OpenAI API, and the reranker model is fetched from Hugging Face on first use.
- Page images and text are sent to the OpenAI API. Don't index documents you aren't comfortable sending there.

## Roadmap

- [ ] Overlap between pages when chunking, and metadata filters (search within one PDF)
- [ ] Incremental indexing with deterministic chunk IDs, so re-ingesting never duplicates
- [ ] Docker image and a hosted demo with public-domain sample PDFs
- [ ] Local mode with open-source models for fully offline use

## Author

**Madhavan K** · [LinkedIn](https://www.linkedin.com/in/madhavan-k-6a4468330)

