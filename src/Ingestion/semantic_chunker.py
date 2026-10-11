from langchain_experimental.text_splitter import SemanticChunker

class SemanticDocumentChunker:
    
    def __init__(self, embedding_model):
        self.chunker = SemanticChunker(
            embeddings=embedding_model
        )

    def split_documents(self, documents):
        return self.chunker.split_documents(documents)
    
#Semantic chunking
    #splits our paragraph into sentences
    #Check similarity score for each sentence and group them into chunks.