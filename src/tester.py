import sys
sys.path.insert(0, "src/Ingestion")        # so "from loader import ..." finds your file

from dotenv import load_dotenv
load_dotenv()                               # loads OPENAI_API_KEY

from loader import DocumentLoader

import inspect
print("LOADED FROM:", inspect.getsourcefile(DocumentLoader))

loader = DocumentLoader("Data/Raw")
docs = loader.load_documents(limit_pages=10)   # first 10 pages only (includes the "Features of Java" page)

print("\nPages loaded:", len(docs))
for d in docs:
    print("\n" + "=" * 60)
    print("page index:", d.metadata["page"], "| chars:", len(d.page_content))
    print(d.page_content[:500])