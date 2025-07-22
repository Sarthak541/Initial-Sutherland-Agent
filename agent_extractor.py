import os
from typing import Optional, Dict, List, Any
import json 
from dotenv import load_dotenv 
import pdfplumber
from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings

# --- CHANGED IMPORTS ---
# We now use the recommended agent creation function for OpenAI
from langchain.agents import AgentExecutor, create_openai_tools_agent
# -----------------------

from langchain.tools import tool
from langchain.prompts import ChatPromptTemplate, HumanMessagePromptTemplate, MessagesPlaceholder
from langchain_chroma import Chroma

# Setup
load_dotenv()

EXTRACTED_STRUCTURES = []
TITLE = ""
SECTIONS = []
SUBJECT = ""

# Pydantic Data Model
class PaperStructure(BaseModel):
    title: str = Field(..., description="The full and complete title of the research paper")
    sections: List[str] = Field(..., description="The list of all the sections in the paper")
    subject: str = Field(..., description="A noun that tells exactly what the subject area of the field is (can be more than one word but less than a phrase)")

# Defining Tools with improved descriptions
@tool
def document_title(title: str):
    """Use this tool to record the single, complete title of the research paper."""
    global TITLE
    TITLE = title
    print(title)
    return "Title successfully recorded"

@tool
def document_section(section: str):
    """Use this tool to record a single section title. You must call this tool for every section title you find."""
    SECTIONS.append(section)
    print(section)
    return "section successfully recorded"

@tool
def document_subject(subject: str):
    """Use this tool to record the main subject area of the paper.  Make the subject one or two words long"""
    global SUBJECT
    SUBJECT = subject
    print(subject)
    return "Subject Area successfully recorded"

@tool
def record_paper_structure() -> str:
    """Use this tool as the final step once the title, all sections, and the subject have been documented."""
    data = PaperStructure(title=TITLE, sections=SECTIONS, subject=SUBJECT)
    print(f"--- 💾 TOOL: Recording entire paper structure ---")
    EXTRACTED_STRUCTURES.append(data)
    return "Successfully recorded the paper's structure. The task is complete."

# Creating Agent
tools = [
    document_title,
    document_section,
    document_subject,
    record_paper_structure
]

content = ""
try:
    with open('output.txt', 'r') as file:
        content = file.read() 
except FileNotFoundError:
    print("Error: output.txt not found. Please ensure the file exists in the same directory.")
    exit()

# --- UPDATED PROMPT ---
# The prompt is simplified and more direct. The new agent type handles the complex parts.
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are an extractor agent. Your job is to extract the title, all section titles, and the subject area from the provided text. "
            "You MUST use the provided tools to document each piece of information. "
            "Follow these steps:\n"
            "1. Call the `document_title` tool once.\n"
            "2. Call the `document_section` tool for every section title you find.\n"
            "3. Call the `document_subject` tool once.\n"
            "4. As the very final step, after all other information is documented, call the `record_paper_structure` tool to finish the task."
        ),
        HumanMessagePromptTemplate.from_template("{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)
# ------------------------

llm = ChatOpenAI(model="gpt-4o", temperature=0)

# --- MODERNIZED AGENT CREATION ---
# Switched from StructuredChatAgent to the recommended create_openai_tools_agent
agent = create_openai_tools_agent(llm=llm, tools=tools, prompt=prompt)
# ---------------------------------

agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=True, handle_parsing_errors=True)

# Running the Agent
if __name__ == "__main__":
    if content:
        print("Starting Agent 1: The Extractor (Hierarchical Mode)...")
        # The task can be simplified as the main instructions are now in the system prompt
        task = f"Please extract the structure and metadata from the following document: {content}"

        result = agent_executor.invoke({"input": task})

        print("\n\nAgent 1 finished its work.")
        print("FINAL EXTRACTED STRUCTURE:")
        print("------------------------------------------")

        data = [item.model_dump() for item in EXTRACTED_STRUCTURES]

        with open("extracted_info.json", "w") as extracted_info:
            json.dump(data, extracted_info, indent=2)

        print("Extracted information saved to extracted_info.json")