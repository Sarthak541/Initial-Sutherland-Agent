import os
import pyodbc
import json
from typing import List
from dotenv import load_dotenv

from langchain_community.utilities import SQLDatabase
# --- CHANGED IMPORTS ---
# Replaced Google's Chat model with OpenAI's
from langchain_openai import ChatOpenAI
# -----------------------
from langchain_community.agent_toolkits import create_sql_agent

# Load environment variables from .env file
load_dotenv()

# --- 1. Load the Extracted Data from the JSON File ---
try:
    with open('extracted_info.json', 'r') as f:
        extracted_papers = json.load(f)
except FileNotFoundError:
    print("Error: extracted_info.json not found. Run the first script to generate it.")
    exit()
except json.JSONDecodeError:
    print("Error: extracted_info.json is empty or corrupted. Please check the file.")
    exit()

# --- 2. Setup the Database ---
# Connection parameters
server = 'SARTHAK_DESK\\SQLEXPRESS'
database = 'Initial_Sutherland_Agent'

# Build connection string for pyodbc
conn_str = (
    f'DRIVER={{ODBC Driver 17 for SQL Server}};'
    f'SERVER={server};'
    f'DATABASE={database};'
    f'Trusted_Connection=yes;'
)

# Connect to SQL Server for verification at the end
try:
    conn = pyodbc.connect(conn_str)
    cursor = conn.cursor()
except pyodbc.Error as ex:
    sqlstate = ex.args[0]
    print(f"Database connection failed: {sqlstate}")
    print("Please ensure SQL Server is running and the connection details are correct.")
    exit()


# Build the SQLAlchemy URI for the LangChain agent
db_uri = (
    f"mssql+pyodbc://@{server}/{database}"
    f"?driver=ODBC+Driver+17+for+SQL+Server&trusted_connection=yes"
)

# Create the SQLDatabase object
db = SQLDatabase.from_uri(db_uri)


# --- 3. Create the SQL Agent ---
# --- CHANGED LLM ---
# Swapped ChatGoogleGenerativeAI for ChatOpenAI and updated the model.
# "gpt-4o" is a powerful choice for SQL generation.
llm = ChatOpenAI(model="gpt-4o", temperature=0)
# -------------------

# The agent_type="openai-tools" is perfect, as it's designed for OpenAI models.
agent_executor = create_sql_agent(
    llm=llm,
    db=db,
    agent_type="openai-tools",
    verbose=True
)


# --- 4. Loop Through Each Paper and Run the Agent ---
for paper_data in extracted_papers:
    data_as_json = json.dumps(paper_data, indent=2)
    
    task = f"""

    DO NOT CREATE NEW TABLES. ONLY INSERT DATA
    
    MAKE SURE ALL OF YOUR QUERIES ARE VALID SQL QUERIES

    
    Your task is to insert the following paper data into the database, avoiding duplicates.

    Here is the data for one paper:
    {data_as_json}

    Follow these steps exactly:
    1. First, insert the title and subject into the "dbo.Titles" table. 
        The dbo.Titles table include columns in this order : 'title_id','title','subject'
    2. After ensuring the title is in the table, you MUST retrieve its 'title_id'.
    3. Then, for each section in the provided 'sections' list, insert a new row into the "dbo.Sections" table, using the 'title_id' you just found as the foreign key.
        The dbo.Sections table include columns in this order:'Section_ID','Title_ID','Section'
    """

    print(f"\n--- 📝 Sending task for '{paper_data['title']}' to SQL Agent ---")
    try:
        agent_executor.invoke({"input": task})
    except Exception as e:
        print(f"An error occurred while processing the paper: {e}")


print("\n--- ✅ All papers processed ---")


# --- 5. Verification Step ---
print("\n--- 🔍 Verifying final database contents ---")
try:
    # --- SYNTAX FIX ---
    # Changed single quotes ' ' to square brackets [ ] for table names,
    # which is the correct syntax for SQL Server.
    print("\n[dbo].[Titles] Table:")
    for row in cursor.execute("SELECT * FROM [dbo].[Titles]"):
        print(row)

    print("\n[dbo].[Sections] Table:")
    for row in cursor.execute("SELECT * FROM [dbo].[Sections]"):
        print(row)
    # ------------------

except pyodbc.Error as e:
    print(f"An error occurred during verification: {e}")
finally:
    conn.close()