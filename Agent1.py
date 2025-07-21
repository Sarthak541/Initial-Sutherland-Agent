import os
import re
import uuid
from typing import List, Dict, Optional, Any
import json

# LangChain components for Gemini
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.tools import tool
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.prompts import PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field

# Config
LLM_MODEL = "gemini-1.5-pro"
LLM_TEMPERATURE = 0
CHROMA_DB_BASE_DIR = "chroma_dbs_gemini" # Base directory for ChromaDB (renamed for Gemini)


EXTRACTED_STRUCTURES = []


# Pydantic Data Models

class PaperMetadata(BaseModel):
    """Data model for the overall paper metadata."""
    # Agent 1
    title: str = Field(..., description="The full title of the research paper.")
    authors: List[str] = Field(..., description="A list of author names.")
    publication_year: Optional[int] = Field(None, description="The year the paper was published.")
    conference_journal: Optional[str] = Field(None, description="The conference or journal where the paper was published.")
    
    # Agent 2
    subject_area: str = Field(..., description="The main subject area or discipline of the paper (infer from keywords, introduction, abstract).")


class SectionDataRow(BaseModel):
    """Data model for a single extracted section."""
    # Agent 1
    section_id: str = Field(..., description="Section Number (e.g. 2)")
    subsection_id: str = Field(..., description="Subsection Number (e.g. 2.1). If it is the beginning of the section then the subsection id should be the same as section id.")
    section_title: str = Field(..., description="The exact title of the main section.")
    subsection_title: str = Field(..., description="The exact title of the subsection. If it is the beginning of the section then the subsection name should be the same as section name.")
    
    # Agent 2
    section_summary: str = Field(..., description="A concise summary of this specific section or subsection's content.")
    key_findings: str = Field(..., description="A specific, actionable key finding or contribution identified within this section")



# Tool Implementation

@tool
def query_vector_store(paper_id: str, metadata_filter: Dict[str, str]) -> List[str]:
    """
    Directly queries the paper's vector store using a metadata filter.
    To get the title, use the filter {'type': 'title'}.
    To get authors, use {'type': 'author'}.
    To get section headers, use {'type': 'section_header'}.
    """
    print(f"--- 🔎 TOOL: Directly querying paper '{paper_id}' with filter: {metadata_filter} ---")
    if paper_id not in db:
        return ["Error: Paper ID not found."]
    
    filter_key = list(metadata_filter.keys())[0]
    filter_value = list(metadata_filter.values())[0]
    
    results = [
        doc["page_content"] for doc in db[paper_id]
        if doc["metadata"].get(filter_key) == filter_value
    ]
    return results

