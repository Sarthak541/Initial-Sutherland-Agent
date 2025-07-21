import os
from typing import List, Dict, Any

# --- Pydantic for data models ---
from pydantic.v1 import BaseModel, Field

# --- LangChain Imports ---
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, StructuredChatAgent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder

# --- 1. SETUP & CONFIGURATION ---
apikey = os.environ.get("GOOGLE_API_KEY")

EXTRACTED_STRUCTURES = []

# --- 2. MOCK DATABASE (Simplified) ---
# The mock store no longer needs a paper_id key.
MOCK_VECTOR_STORE = [
    {"page_content": "A Deep Dive into Multi-Agent Systems", "metadata": {"type": "title"}},
    {"page_content": "Alex Doe", "metadata": {"type": "author"}},
    {"page_content": "Brenda Smith", "metadata": {"type": "author"}},
    {"page_content": "Published at the International Conference on AI (ICAI), 2025", "metadata": {"type": "publication_info"}},
    {"page_content": "1. Introduction", "metadata": {"type": "section_header"}},
    {"page_content": "2. Related Work", "metadata": {"type": "section_header"}},
    {"page_content": "3. Methodology", "metadata": {"type": "section_header"}},
    {"page_content": "3.1 System Architecture", "metadata": {"type": "subsection_header"}},
    {"page_content": "3.2 Agent Communication Protocol", "metadata": {"type": "subsection_header"}},
    {"page_content": "4. Results", "metadata": {"type": "section_header"}},
    {"page_content": "5. Conclusion", "metadata": {"type": "section_header"}},
]


# --- 3. PYDANTIC DATA MODEL (Simplified) ---
class PaperStructure(BaseModel):
    """Data model for the explicitly stated structure and metadata of a paper."""
    title: str = Field(..., description="The full, complete title of the research paper.")
    authors: List[str] = Field(..., description="A list of all author names found on the title page.")
    publication_year: int = Field(..., description="The year the paper was published.")
    conference: str = Field(..., description="The conference or journal where the paper was published.")
    sections: List[Dict[str, Any]] = Field(..., description="A nested list of all sections and subsections.")

# --- 4. TOOL IMPLEMENTATION (Simplified) ---
@tool
def query_vector_store(metadata_filter: Dict[str, str]) -> List[str]:
    """
    Directly queries the paper's vector store using a metadata filter.
    To get the title, use the filter {'type': 'title'}.
    To get authors, use {'type': 'author'}.
    To get section headers, use {'type': 'section_header'}.
    """
    print(f"--- 🔎 TOOL: Directly querying with filter: {metadata_filter} ---")
    filter_key = list(metadata_filter.keys())[0]
    filter_value = list(metadata_filter.values())[0]
    
    results = [
        doc["page_content"] for doc in MOCK_VECTOR_STORE
        if doc["metadata"].get(filter_key) == filter_value
    ]
    return results

@tool(args_schema=PaperStructure)
def record_paper_structure(data: PaperStructure) -> str:
    """Records the complete, extracted structure and metadata of the paper."""
    print(f"--- 💾 TOOL: Recording entire paper structure ---")
    EXTRACTED_STRUCTURES.append(data)
    return "Successfully recorded the paper's structure."

# --- 5. AGENT AND EXECUTOR CREATION ---
tools = [
    query_vector_store,
    record_paper_structure,
]

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an Extractor Agent. Your job is to find explicit facts by directly querying a vector store and then record them.

            Your process is:
            1. Use the `query_vector_store` tool multiple times to get all the pieces of information you need. You must learn to use the correct `metadata_filter` to get each piece of data.
               - To get the title, use `{'type': 'title'}`.
               - To get authors, use `{'type': 'author'}`.
               - To get publication info, use `{'type': 'publication_info'}`.
               - To get main section titles, use `{'type': 'section_header'}`.
               - To get subsection titles, use `{'type': 'subsection_header'}`.
            2. After you have retrieved all the parts, assemble the complete structure.
            3. Make a SINGLE call to the `record_paper_structure` tool to save the final, complete result.
            4. Do not summarize or infer. Your task is complete after the single record call."""
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro-latest", temperature=0, convert_system_message_to_human=True)
agent = StructuredChatAgent.from_llm_and_tools(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# --- 6. RUNNING THE AGENT ---
if __name__ == "__main__":
    print("🚀 Starting Agent 1 (Simplified)...")
    task = "Extract and record the full structure and metadata for the paper by querying the vector store directly."

    result = agent_executor.invoke({"input": task})

    print("\n\n✅ Agent 1 finished its work.")
    print("------------------------------------------")
    print("📊 FINAL EXTRACTED STRUCTURE:")
    print("------------------------------------------")

    for item in EXTRACTED_STRUCTURES:
        print(item.model_dump_json(indent=2))