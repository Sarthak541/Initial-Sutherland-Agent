import os
from uuid import uuid4
from dotenv import load_dotenv

from langchain_community.document_loaders import UnstructuredPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load API Key
load_dotenv()

CHROMA_PATH = "chroma_db_structured"
pdf_file = "./s1.pdf"


loader = UnstructuredPDFLoader(pdf_file, mode="elements")
docs = loader.load()

embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = Chroma(
    collection_name="structured_paper_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

vector_store.add_documents(documents=docs, ids=[str(uuid4()) for _ in range(len(docs))])

