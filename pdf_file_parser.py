import pdfplumber 

file = open("file2.txt","w",encoding="utf-8")
filename = "./s2.pdf"
with pdfplumber.open(filename) as pdf:
    ind = 1
    
    for pages in pdf.pages:
        file.write(f"Page {ind}:\n" +  str(pages.extract_words()))
        ind += 1
        break

