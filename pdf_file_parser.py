import pdfplumber 

filename = "./pdf-sample_0.pdf"
with pdfplumber.open(filename) as pdf:
    first_page=pdf.pages[0]
    print(pdf.pages)

