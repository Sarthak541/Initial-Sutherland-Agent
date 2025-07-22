import os
from typing import Optional, Dict, List, Any

from dotenv import load_dotenv

from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, StructuredChatAgent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Setup
load_dotenv()
CHROMA_PATH = "chroma_db_structured"
embeddings_model = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

vector_store = Chroma(
    persist_directory=CHROMA_PATH,
    embedding_function=embeddings_model,
    collection_name="structured_paper_collection"
)

# Global lists to store data during the run
EXTRACTED_STRUCTURES = []
SECTIONS = []

# Pydantic Data Model
class PaperStructure(BaseModel):
    """Final data model for storing the list of section titles."""
    sections: List[str] = Field(..., description="A list of all section titles found in the paper.")

# Defining Tools
@tool
def query_vector_store(text_query: str, metadata_filter: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Searches the vector store. Use a metadata_filter to find structural elements.
    A valid filter is {'category': 'Title'}.
    """
    print(f"--- 🔎 Querying with text='{text_query}' and filter={metadata_filter} ---")
    if metadata_filter:
        docs = vector_store.similarity_search(query=text_query, filter=metadata_filter)
    else:
        docs = vector_store.similarity_search(query=text_query)
    return [doc.page_content for doc in docs]

@tool
def section_storage(section_title: str):
    """Use this tool to store a single section title that you have found."""
    print(f"--- Storing Section: {section_title} ---")
    SECTIONS.append(section_title)
    return f"Stored Section: {section_title}"

@tool
def record_final_structure() -> str:
    """
    Call this tool ONLY ONCE at the very end after all sections have been stored.
    It takes no arguments and saves the final list of collected sections.
    """
    data = PaperStructure(sections=SECTIONS)
    print(f"--- 💾 Recording Final Structure ---")
    EXTRACTED_STRUCTURES.append(data)
    return "Successfully recorded the complete paper structure."

# Creating Agent
tools = [
    query_vector_store,
    section_storage,
    record_final_structure,
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an Extractor Agent. You must follow a strict, non-negotiable procedure to get all section headers.

            **CRITICAL INSTRUCTIONS:**
            - You are **STRICTLY FORBIDDEN** from using any other filter format or operators.

            **MANDATORY WORKFLOW:**
            1.  **FIRST ACTION:** Call `query_vector_store` to get all possible section headers. Use `text_query='*'` and `metadata_filter={{'category': 'Title'}}`.
            2.  **ITERATE AND STORE:** For **each** section title found in the result from the previous step, call the `section_storage` tool one time to store it.
            3.  **FINAL STEP:** After storing all the section titles, call `record_final_structure` **ONCE** to finalize the process. This tool takes no arguments."""
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Corrected the model name to a valid one, e.g., gemini-1.5-flash
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0)
agent = StructuredChatAgent.from_llm_and_tools(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Running the Agent
if __name__ == "__main__":
    print("Starting Agent 1: Section Header Extractor...")
    task = "Extract and record every section header from the paper using the available tools. Follow your prompt directives exactly."

    result = agent_executor.invoke({"input": task})

    print("\n\n✅ Agent 1 finished its work.")
    print("📊 FINAL EXTRACTED STRUCTURE:")
    print("------------------------------------------")

    for item in EXTRACTED_STRUCTURES:
        print(item.model_dump_json(indent=2))