from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import OpenAIEmbeddings

def create_vector_store(chunks):
    embeddings = OpenAIEmbeddings()
    db = FAISS.from_documents(chunks, embeddings)
    return db