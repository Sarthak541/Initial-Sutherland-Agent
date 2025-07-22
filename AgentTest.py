import os
from typing import Optional, Dict, List, Any
import json 
from dotenv import load_dotenv 
import pdfplumber
from pydantic import BaseModel, Field

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import AgentExecutor, StructuredChatAgent
from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma
from langchain_google_genai import GoogleGenerativeAIEmbeddings
import document_loader

# Setup
load_dotenv()



def load_pdf_text(pdf_path):
    all_text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            all_text += page.extract_text() or ""
    return all_text


EXTRACTED_STRUCTURES = []
TITLE = ""
SECTIONS = []
SUBJECT = ""



# Pydantic Data Model
class PaperStructure(BaseModel):
    """Data model for the explicitly stated structure and metadata of a paper."""
    title: str = Field(..., description="The full and complete titles of all the sections in the research paper")
    sections: List[str] = Field(..., description="The list of all the sections in the paper")
    subject: str = Field(..., description="A noun that tells exactly what the subject area of the field is (can be more than one word but less than a phrase)")

# Defining Tools
@tool
def query_vector_store(text_query: str, metadata_filter: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Searches the vector store. Can be used with a text query, a metadata filter, or both.
    To find structural elements like the title, use a generic query like "*" and a specific filter.
    To find specific concepts, use a descriptive text query.
    """
    print(f"Querying with text='{text_query}' and filter={metadata_filter}")
    
    # The ChromaDB client handles the case where filter is None
    if (metadata_filter):
        docs = vector_store.similarity_search(
            query=text_query,
            filter=metadata_filter
        )
    else:
        docs = vector_store.similarity_search(query=text_query)
    return [doc.page_content for doc in docs]

@tool
def document_title(title: str):
    """
    Documents the title
    """
    TITLE = title
    print(title)
    return "Title successfully recorded"

@tool
def document_section(section: str):
    """
    Appends a section to the list of sections that have been documented
    """
    SECTIONS.append(section)

    print(section)
    return "section successfully recorded"

@tool
def document_subject(subject: str):
    """
    Documents the subject area
    """
    SUBJECT = subject
    print(subject)
    return "Subject Area successfully recorded"

@tool
def record_paper_structure() -> str:
    """once everything has been documented, this will officially record everything into the paper structure"""

    data = PaperStructure(title=TITLE, sections=SECTIONS, subject_area=SUBJECT)
    print(f"--- 💾 TOOL: Recording entire paper structure ---")
    EXTRACTED_STRUCTURES.append(data)
    return "Successfully recorded the paper's title."

# Creating Agent
tools = [
    document_title,
    document_section,
    document_subject,
    record_paper_structure
]
content = ""
with open('output.txt', 'r') as file:
    content = file.read() 
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            f"""
            You are an extractor agent working with text
            
            The format may not be consistent and some information may be incorrect, extraneous. 
            Your job is to extract the proper information and document the title, sections, and subject area of the paper
            Use the document tools to document the information

            Here will be your workflow:
                1: retrieve the title, every section, and the subject area from the information and document them
                2: call record_paper_structure ONCE when you have documented all of the information and are ready to record it for future use
            """
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

llm = ChatGoogleGenerativeAI(model="gemini-2.5-pro", temperature=0, convert_system_message_to_human=True)
agent = StructuredChatAgent.from_llm_and_tools(llm=llm, tools=tools, prompt=prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)


# Running the Agent
if __name__ == "__main__":
    print("Starting Agent 1: The Extractor (Hierarchical Mode)...")
    task = f"extract the information given to you and document it using the tools provided. Once finished documenting, record the paper strucutre. Make sure you remember to use all of the information to get all of the sections, accurate title, and accurate subject area. Here is the document: {content}"

    result = agent_executor.invoke({"input": task})

    print("\n\nAgent 1 finished its work.")
    print("FINAL EXTRACTED STRUCTURE:")
    print("------------------------------------------")

    data = []

    for item in EXTRACTED_STRUCTURES:
        data.append(item.model_dump())

    with open("extracted_info.json","w") as extracted_info:
        json.dump(data,extracted_info,indent=2)
    