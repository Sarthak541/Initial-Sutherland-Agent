import pdfplumber

with pdfplumber.open("./pdf-sample_0.pdf") as pdf:
    first_page=pdf.pages[0]
    print(first_page.objects)