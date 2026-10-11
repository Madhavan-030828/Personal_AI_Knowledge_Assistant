from langchain_openai import OpenAIEmbeddings

def get_embedding_model():
    return OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

#How chunks are getting converted into embeddings
    #1.Tokenization
        #-The chunk is broken down into smaller pieces and then those pieces are converted into numbers.
    #2.Contextualization
        #Ai puts/reads together all those tokens based on its grammer rule training.
    #3.Pooling
        #Those tokens are then combined by Ai and gets converted into one vector.(one chunk one vector)
    #4.Normalization
        #The model then makes the length of vector to 1 unit.(To make the cosine similarity search easy).