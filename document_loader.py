# In your document_loader.py file

import os
from uuid import uuid4
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
# ⬇️ 1. Import the filter utility
from langchain_community.vectorstores.utils import filter_complex_metadata

# ... (your existing setup code)
CHROMA_PATH = "chroma_db_structured"
pdf_file = "./s1.pdf"



load_dotenv() #loads api key
# --- Load documents ---
loader = UnstructuredPDFLoader(pdf_file, mode="elements")
docs = loader.load()



embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
# ⬇️ 2. Apply the filter to clean the metadata
filtered_docs = filter_complex_metadata(docs)

# ... (your existing ChromaDB setup code)
vector_store = Chroma(
    collection_name="structured_paper_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# ⬇️ 3. Add the CLEANED documents to the vector store
vector_store.add_documents(documents=filtered_docs, ids=[str(uuid4()) for _ in range(len(filtered_docs))])

print("✅ Ingestion complete with filtered metadata.")