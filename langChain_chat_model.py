from langchain.prompts import ChatPromptTemplate
from langchain.schema import SystemMessage, HumanMessage
from langchain.chat_models import ChatOpenAI
import os
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.tools.tavily_search import TavilySearchResults

# Load environment variables
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
tavily_key = os.getenv("TAVILY_API_KEY")
os.environ["TAVILY_API_KEY"] = tavily_key

# Initialize OpenAI Chat model
chat = ChatOpenAI(
    model_name="gpt-3.5-turbo",
    temperature=0.7,
    openai_api_key=api_key
)

# Embedding and vector store
embedding_model = OpenAIEmbeddings(openai_api_key=api_key)

# Build vector DB if not already there
def build_vectorstore():
    loader = PyPDFLoader("SanjayDubey.pdf")
    pages = loader.load_and_split()
    return Chroma.from_documents(pages, embedding_model, persist_directory="./chroma")

vectorstore = build_vectorstore()
retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

# Tavily search
search = TavilySearchResults(max_results=4)

# Prompt template
chat_prompt = ChatPromptTemplate.from_messages([
    ("system", 
     "You are a helpful assistant. Use the provided context to answer the question as accurately as possible. "
     "If the answer cannot be found in the context, attempt to answer using your own knowledge. "
     "If you're still unable to answer confidently, reply with 'NOT_FOUND'."),
    ("human", "Context:\n{context}"),
    ("human", "{user_input}")
])


def get_chat_response(user_input: str) -> str:
    # Step 1: Try PDF context (RAG)
    docs = retriever.get_relevant_documents(user_input)
    pdf_context = "\n\n".join([doc.page_content for doc in docs]) if docs else ""

    # Step 2: Ask LLM using PDF context
    messages = chat_prompt.format_messages(context=pdf_context, user_input=user_input)
    response = chat(messages)

    if "NOT_FOUND" not in response.content and len(response.content.strip()) > 10:
        return f"[From KB]\n{response.content.strip()}"

    # Step 3: Fall back to Tavily search
    search_results = search.run(user_input)
    web_context = "\n\n".join([r['content'] for r in search_results])
    messages = chat_prompt.format_messages(context=web_context, user_input=user_input)
    response = chat(messages)

    return f"[From Web]\n{response.content.strip()}"
