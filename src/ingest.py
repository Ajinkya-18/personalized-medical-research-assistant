import os
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


DATA_DIR = 'knowledge-base'
FAISS_DB_PATH = 'faiss_index'


loader = PyPDFDirectoryLoader(DATA_DIR)
documents = loader.load()

print(f"Loaded {len(documents)} documents.")


text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=25)
docs = text_splitter.split_documents(documents)

print(f"Split into {len(docs)} chunks.")


model_name = "sentence-transformers/all-MiniLM-L6-v2"
embeddings = HuggingFaceEmbeddings(
    model_name=model_name, 
    model_kwargs = {'device': 'cpu'}
)

print("Embedding model loaded.")


db = FAISS.from_documents(docs, embeddings)
db.save_local(FAISS_DB_PATH)

print(f"Vector store saved to {FAISS_DB_PATH}")



