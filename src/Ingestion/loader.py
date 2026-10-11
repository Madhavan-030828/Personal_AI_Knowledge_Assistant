import base64
import json
import re
import time
from pathlib import Path

import pymupdf
from langchain_core.documents import Document
from openai import OpenAI

OCR_PROMPT = (
    "Transcribe this handwritten notes page exactly as written, in Markdown. "
    "Keep headings, bullets, code and formulas. Do not add explanations or anything not on the page. "
    "Ignore scanner watermarks such as 'Scanned with CamScanner' and the circled page number. "
    "If a word is unreadable, write [illegible]."
)


class DocumentLoader:
    def __init__(self, docs_path: str, cache_dir: str = "Data/ocr_cache", model: str = "gpt-4o-mini"):
        self.docs_path = Path(docs_path)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.client = OpenAI(max_retries=8)
        self.model = model

    def _ocr_page(self, page) -> str:
        pix = page.get_pixmap(dpi=150)
        b64 = base64.b64encode(pix.tobytes("png")).decode()
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=1200,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "text", "text": OCR_PROMPT},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}},
                ],
            }],
        )
        return resp.choices[0].message.content or ""

    def load_documents(self, limit_pages: int | None = None):
        documents = []
        for pdf_path in sorted(self.docs_path.glob("**/*.pdf")):
            cache_file = self.cache_dir / f"{pdf_path.stem}.json"
            cache = json.loads(cache_file.read_text(encoding="utf-8")) if cache_file.exists() else {}

            pdf = pymupdf.open(pdf_path)
            for i, page in enumerate(pdf):
                if limit_pages and i >= limit_pages:
                    break

                text = page.get_text().strip()
                if len(text) < 50:                      # scanned/handwritten page -> OCR (cached)
                    if str(i) not in cache:
                        try:
                            cache[str(i)] = self._ocr_page(page).strip()
                            cache_file.write_text(json.dumps(cache), encoding="utf-8")
                            time.sleep(4)               # stay under the tokens-per-minute limit
                            print(f"  OCR done: {pdf_path.name} page {i + 1}/{len(pdf)}")
                        except Exception as e:
                            print(f"OCR failed for {pdf_path.name} page {i}: {e}")
                            continue
                    text = cache[str(i)]

                text = re.sub(r"^```(?:markdown)?\s*|\s*```$", "", text.strip())   # strip markdown fences

                if len(text) > 50:
                    documents.append(Document(
                        page_content=text,
                        metadata={"source": str(pdf_path), "page": i},
                    ))
            print(f"{pdf_path.name}: {len(pdf)} pages processed")
        return documents