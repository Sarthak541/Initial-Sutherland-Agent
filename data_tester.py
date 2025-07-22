from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

from dotenv import load_dotenv

load_dotenv()

CHROMA_PATH = "chroma_db_structured"
COLLECTION_NAME = "structured_paper_collection"

# Connect to your existing database
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
vector_store = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings_model,
    collection_name=COLLECTION_NAME
)

print("Searching for title chunk...")

# Perform the exact query the agent should be using
try:
    results = vector_store.similarity_search(
        query="*",
        filter={"category": "Title"}
    )

    if results:
        print("\n✅ Title chunk found!")
        print("--------------------")
        for doc in results:
            print(doc.page_content)
            print(f"Metadata: {doc.metadata}")
    else:
        print("\n❌ No chunk with 'category: Title' was found in the database.")

except Exception as e:
    print(f"An error occurred during search: {e}")