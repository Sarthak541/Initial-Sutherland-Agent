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

# Pydantic Data Models

class PaperMetadata(BaseModel):
    paper_id: str = Field(..., description="Unique identifier for the paper.")
    title: str = Field(..., description="The full title of the research paper.")
    authors: List[str] = Field(..., description="A list of author names.")
    publication_year: Optional[int] = Field(None, description="The year the paper was published.")
    conference_journal: Optional[str] = Field(None, description="The conference or journal where the paper was published.")
    subject_area: Optional[str] = Field(None, description="The main subject area or discipline of the paper (infer from keywords, introduction, abstract).")

