
from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI  # Replace with your model, e.g., Ollama
import os
from dotenv import load_dotenv

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
    ("system", "You are a helpful assistant."),
    ("human", "{user_input}")
])

def get_chat_response(user_input: str) -> str:
    messages = chat_prompt.format_messages(user_input=user_input)
    response = chat(messages)
    return response.content
