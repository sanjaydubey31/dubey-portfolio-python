
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI  # Replace with your model, e.g., Ollama
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
from langchain.document_loaders import PyPDFLoader

load_dotenv()  # Load variables from .env
api_key = os.getenv("OPENAI_API_KEY")
# Initialize your model (for OpenAI, or use Ollama, HuggingFace, etc.)
chat = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=api_key
)

# Create the chat prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Use the provided context to answer the question."),
    ("human", "Context:\n{context}"),
    ("human", "{user_input}")
])

def get_chat_response(user_input: str) -> str:
    # Step 1: Fetch and parse HTML
    #mysite = requests.get("https://main.d2raojdfdmfx91.amplifyapp.com/")
    #soup = BeautifulSoup(mysite.content, 'html.parser')
    #context = ""
    #for paragraph in soup.find_all('p'):
    #    text += paragraph.get_text()
    # Step 2: Extract visible text (to use as context)
    #context = soup.get_text(separator="\n", strip=True)
    loader = PyPDFLoader("SanjayDubey.pdf")  # Replace with your actual PDF file path
    pages = loader.load()

    # 2. Extract content (you can also chunk or filter this)
    context = "\n\n".join([page.page_content for page in pages])
    #print("Extracted content from PDF:", context)  # Print first 500 characters for debugging

    messages = chat_prompt.format_messages(context=context, user_input=user_input)
    response = chat(messages)
    return response.content

