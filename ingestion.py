import os
import sys  # Import the sys module to access command-line arguments
import pdfplumber
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

# --- 1. Setup ---
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found in environment variables.")

# --- CHANGED: Get PDF file from command-line argument ---
# Check if a file path was provided when running the script
if len(sys.argv) < 2:
    print("Error: No input PDF file provided.")
    # sys.argv[0] is the name of the script itself
    print(f"Usage: python {sys.argv[0]} /path/to/your/file.pdf")
    exit()

# The first argument after the script name is the file path
pdf_file = sys.argv[1]


# --- 2. Extract Full Plaintext from PDF ---
full_text = ""
try:
    with pdfplumber.open(pdf_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n\n"
except FileNotFoundError:
    print(f"Error: The file '{pdf_file}' was not found. Please make sure the path is correct.")
    exit()
except Exception as e:
    print(f"An error occurred while reading the PDF: {e}")
    exit()

print(f"📄 Successfully extracted text from '{pdf_file}'.")


# --- 3. Demonstrate the Limitation ---
char_count = len(full_text)
# A rough approximation: 1 token ~= 4 characters in English
token_count = char_count / 4


# --- 4. Attempt to Query the LLM with the Full Text ---
if char_count > 0:
    llm = ChatOpenAI(model="gpt-4o")

    # We create a prompt that stuffs the entire document text into one variable
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an expert at analyzing documents. Answer the user's question based on the provided text."),
            ("human", "Here is the full document text:\n\n---\n{document_text}\n---\n\nQuestion: {question}")
        ]
    )

    chain = prompt | llm

    print("\nSending the entire document to the AI for analysis...")
    try:
        result = chain.invoke({
            "document_text": full_text,
            "question": "What is the title, what are the sections in this document, and what is the subject area (eg. AI, Cybersecurity, Learning, etc.)"
        })
        print("\n✅ AI Response:")
        # Write the successful output to a file for the next script to use
        with open("output.txt", "w", encoding="utf-8") as f:
            f.write(result.content)
        print(result.content)
    except Exception as e:
        print("\n❌ An error occurred during the API call.")
        print("This is the expected outcome for documents larger than the model's context window.")
        print(f"Error Details: {e}")