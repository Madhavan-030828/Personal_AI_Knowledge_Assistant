from dotenv import load_dotenv
from Retrieval.hybrid_retriever import HybridRetriever
from langchain_openai import OpenAIEmbeddings
load_dotenv()

VECTOR_DB_PATH = "vector_store/chroma_db"


embedding_model = OpenAIEmbeddings(
        model = "text-embedding-3-small"
    )
retriever = HybridRetriever(
        persist_directory=VECTOR_DB_PATH,
        embedding_model=embedding_model,
    )

def retrieve(query: str):
        return retriever.retrieve(query, final_top_n=5)

if __name__ == "__main__":
    retrieve("Name of the company")