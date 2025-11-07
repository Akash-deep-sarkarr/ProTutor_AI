import pdfplumber

def extract_text_from_pdf(uploaded_file):
    with pdfplumber.open(uploaded_file) as pdf:
        return "\n".join([page.extract_text() for page in pdf.pages if page.extract_text()])

def chunk_text(text, length=1000):
    return [text[i:i+length] for i in range(0, len(text), length)]