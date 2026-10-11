from dotenv import load_dotenv
from langchain_chroma import Chroma
from loader import DocumentLoader
from semantic_chunker import SemanticDocumentChunker
from embedding import get_embedding_model

load_dotenv()

DOCUMENT_PATH = "Data/Raw"            
VECTOR_DB_PATH = "vector_store/chroma_db"

def ingest():

    embedding_model = get_embedding_model()

    loader = DocumentLoader(DOCUMENT_PATH)
    documents = loader.load_documents()
    print(f"Loaded {len(documents)} pages")

    chunker = SemanticDocumentChunker(embedding_model)
    chunks = chunker.split_documents(documents)

    # Drop blank/tiny chunks so they never reach the vector store
    chunks = [c for c in chunks if len(c.page_content.strip()) > 50]
    if not chunks:
        raise ValueError("No non-empty chunks. Check the PDFs/OCR.")
    print(f"Kept {len(chunks)} non-empty chunks")
    print(chunks[0].page_content[:200])      # should be readable Java/math text

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding_model,
        persist_directory=str(VECTOR_DB_PATH),
        collection_metadata={
            "hnsw:space": "cosine"
        }
    )

    print(f"Indexed {len(chunks)} chunks successfully.")


if __name__ == "__main__":
    ingest()