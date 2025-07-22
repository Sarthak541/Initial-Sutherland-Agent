import os
from docling import Docling

# --- Configuration ---
# Make sure this PDF file exists in the same directory as your script.
# For best results, use a PDF that contains varied content:
# - Regular text
# - Headings
# - Lists
# - A complex table (with merged cells, multiple rows, etc.)
PDF_FILE_PATH = "sample_document.pdf"
OUTPUT_DIR = "docling_output"

# Create output directory if it doesn't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"--- Starting Docling Processing for: {PDF_FILE_PATH} ---")

# --- Step 1: Initialize Docling ---
# You can specify model size: 'default' or 'smol'
# 'default' provides higher accuracy for complex documents.
# 'smol' is faster and uses less memory, good for simpler docs or testing.
try:
    docling = Docling(model_size='default') # Or 'smol'
    print("Docling initialized successfully.")
except Exception as e:
    print(f"Error initializing Docling: {e}")
    print("Please ensure you have run 'pip install \"docling[all]\"' and have sufficient memory.")
    exit()

# --- Step 2: Load and Process the Document ---
try:
    print(f"Loading and processing {PDF_FILE_PATH}...")
    # The 'process' method performs OCR (if needed), layout analysis,
    # and extraction of elements.
    # It returns a DoclingDocument object.
    doc = docling.process(PDF_FILE_PATH)
    print("Document processed successfully.")
    print(f"Total pages: {len(doc.pages)}")
    print(f"Total elements extracted: {len(doc.elements)}")

except FileNotFoundError:
    print(f"Error: The file '{PDF_FILE_PATH}' was not found.")
    print("Please ensure the PDF file is in the correct directory.")
    exit()
except Exception as e:
    print(f"Error during document processing: {e}")
    exit()

# --- Step 3: Access and Print Extracted Elements ---
print("\n--- Extracted Elements (First 20 for brevity) ---")
for i, element in enumerate(doc.elements):
    if i >= 20: # Limit output for demonstration
        print("...")
        break

    print(f"\nElement Type: {element.type}")
    print(f"Text Content: {element.text[:100]}...") # Show first 100 chars
    print(f"Page Number: {element.page_number}")

    if element.type == 'table':
        print("\n  --- Table Details ---")
        # Tables have a 'data' attribute which is usually a list of lists (rows and cells)
        # or a pandas DataFrame (if pandas is installed and enabled in docling config).
        # We'll print the first 3 rows of the table data.
        table_data = element.data
        if table_data:
            for row_idx, row in enumerate(table_data):
                if row_idx >= 3:
                    print("    ... (more rows)")
                    break
                print(f"    Row {row_idx}: {row}")
        else:
            print("    No structured table data found.")
        print("  -------------------")

    elif element.type == 'heading':
        print(f"  Heading Level: {element.metadata.get('level')}")
    elif element.type == 'list_item':
        print(f"  List Item Indent: {element.metadata.get('indent')}")

# --- Step 4: Export the Document to Different Formats ---
print(f"\n--- Exporting Document to {OUTPUT_DIR} ---")

# Export to Markdown
md_output_path = os.path.join(OUTPUT_DIR, f"{os.path.basename(PDF_FILE_PATH).replace('.pdf', '')}.md")
doc.to_markdown(md_output_path)
print(f"Document exported to Markdown: {md_output_path}")

# Export to JSON
json_output_path = os.path.join(OUTPUT_DIR, f"{os.path.basename(PDF_FILE_PATH).replace('.pdf', '')}.json")
doc.to_json(json_output_path)
print(f"Document exported to JSON: {json_output_path}")

# --- Step 5: (Optional) Show Markdown content (first 500 chars) ---
print("\n--- Markdown Content Sample (first 500 chars) ---")
try:
    with open(md_output_path, 'r', encoding='utf-8') as f:
        md_content = f.read()
        print(md_content[:500])
        if len(md_content) > 500:
            print("...")
except Exception as e:
    print(f"Could not read Markdown file: {e}")


print("\n--- Docling Processing Complete ---")