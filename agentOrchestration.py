import os
from uuid import uuid4
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

# Load API Key
load_dotenv()

CHROMA_PATH = "chroma_db"

# --- Load and Split PDFs (multiple files) ---
pdf_file = "./s1.pdf"

all_chunks = []
title_chunks = []  # ⬅️ store first chunks here

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)


loader = PyPDFLoader(pdf_file)
docs = loader.load()

chunks = text_splitter.split_documents(docs)

if chunks:
    title_chunks.append(chunks[0])  # ⬅️ Save first chunk of each file

all_chunks.extend(chunks)

# --- Create Embeddings and Vector Store ---
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = Chroma(
    collection_name="example_collection",
    embedding_function=embeddings_model,
    persist_directory=CHROMA_PATH,
)

# --- Index the chunks ---
uuids = [str(uuid4()) for _ in range(len(all_chunks))]
vector_store.add_documents(documents=all_chunks, ids=uuids)

# --- Set up LLM ---
llm = GoogleGenerativeAI(model="models/gemini-1.5-flash-latest")

# --- Set up retriever ---
num_results = 5
retriever = vector_store.as_retriever(search_kwargs={'k': num_results})

# --- RAG + Title-Aware Prompting ---
def stream_response(message):
    # Retrieve top-k relevant chunks
    docs = retriever.invoke(message)
    
    # Combine manual title chunks + semantic matches
    all_docs = title_chunks + docs

    knowledge = "\n\n".join(doc.page_content for doc in all_docs)

    rag_prompt = f"""
    You are an assistant which answers questions based on knowledge which is provided to you.

    The question: {message}

    The knowledge: {knowledge}
    """

    response = llm.invoke(rag_prompt)
    return response

# Test it
first_message = "What is the title of the PDF?  Only return the title/heading"
print(stream_response(first_message))