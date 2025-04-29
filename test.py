from langchain.document_loaders import PyPDFLoader



# 1. Load PDF
loader = PyPDFLoader("SanjayDubey.pdf")  # Replace with your actual PDF file path
pages = loader.load()

# 2. Extract content (you can also chunk or filter this)
context = "\n\n".join([page.page_content for page in pages])
print("Extracted content from PDF:", context)  # Print first 500 characters for debugging