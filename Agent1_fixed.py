import os
from typing import List, Dict, Any, Optional

from pydantic.v1 import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, StructuredChatAgent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Setup
aipikey = os.environ.get("GOOGLE_API_KEY")

CHROMA_PATH = "chroma_db_structured"
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings_model,
    collection_name="structured_paper_collection"
)

EXTRACTED_STRUCTURES = []


# Pydantic Data Model
class PaperStructure(BaseModel):
    """Data model for the explicitly stated structure and metadata of a paper."""
    title: str = Field(..., description="The full, complete title of the research paper.")
    authors: List[str] = Field(..., description="A list of all author names found on the title page.")
    publication_year: Optional[int] = Field(None, description="The year the paper was published, if found.")
    conference: Optional[str] = Field(None, description="The conference or journal where the paper was published, if found.")
    sections: List[Dict[str, Any]] = Field(..., description="A nested list of all sections and their subsections. Example: [{'section_title': 'Methodology', 'subsections': ['System Architecture', 'Agent Protocol']}]")


# Defining Tools
@tool
def query_vector_store(text_query: str, metadata_filter: Optional[Dict[str, str]] = None) -> List[str]:
    """
    Searches the vector store. Can be used with a text query, a metadata filter, or both.
    To find structural elements like the title, use a generic query like "*" and a specific filter.
    To find specific concepts, use a descriptive text query.
    """
    print(f"Querying with text='{text_query}' and filter={metadata_filter}")
    
    # The ChromaDB client handles the case where filter is None
    docs = vector_store.similarity_search(
        query=text_query,
        filter=metadata_filter
    )
    
    return [doc.page_content for doc in docs]

@tool(args_schema=PaperStructure)
def record_paper_structure(data: PaperStructure) -> str:
    """Records the complete, extracted structure and metadata of the paper."""
    print(f"--- 💾 TOOL: Recording entire paper structure ---")
    EXTRACTED_STRUCTURES.append(data)
    return "Successfully recorded the paper's structure."

# Creating Agent
tools = [
    query_vector_store,
    record_paper_structure,
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an Extractor Agent responsible for finding a paper's structure and metadata. You do not summarize or infer.

            You have one primary tool: `query_vector_store`. It takes two arguments: a `text_query` (for semantic searching) and an optional `metadata_filter`.

            Here is your strategy:
            - **To find structural elements (like the paper's main title or all section headers):** Use a generic `text_query` like "*" or "document" combined with a specific `metadata_filter` (e.g., `{'category': 'Title'}`).
            - **To find specific information (like authors or publication details):** Use a descriptive `text_query` (e.g., `text_query='authors and affiliations'`).

            Your step-by-step process is:
            1.  Find the main title of the paper using the metadata filter strategy.
            2.  Find the authors and publication details using the text query strategy.
            3.  Get all section and subsection titles using the metadata filter strategy.
            4.  Analyze the list of retrieved titles to determine the paper's hierarchy (e.g., '3. Methods' vs '3.1. Data Collection').
            5.  Once all information is gathered, assemble the complete `PaperStructure` object.
            6.  Make a SINGLE call to the `record_paper_structure` tool to save the final result."""
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0, convert_system_message_to_human=True)
agent = StructuredChatAgent.from_llm_and_tools(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


# Running the Agent
if __name__ == "__main__":
    print("Starting Agent 1: The Extractor (Hierarchical Mode)...")
    task = "Extract and record the full hierarchical structure (sections and subsections) and metadata for the paper."

    result = agent_executor.invoke({"input": task})

    print("\n\nAgent 1 finished its work.")
    print("FINAL EXTRACTED STRUCTURE:")
    print("------------------------------------------")

    for item in EXTRACTED_STRUCTURES:
        print(item.model_dump_json(indent=2))