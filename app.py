from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.document_loaders import UnstructuredFileLoader, TextLoader
from langchain.embeddings import CacheBackedEmbeddings, OpenAIEmbeddings
from langchain.schema.runnable import RunnableLambda, RunnablePassthrough
from langchain.storage import LocalFileStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores.faiss import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.callbacks.base import BaseCallbackHandler
from langchain.memory import ConversationBufferMemory
import streamlit as st
import os
from dotenv import load_dotenv

st.set_page_config(
    page_title="DocumentGPT",
    page_icon="📃",
)

# Load environment variables
load_dotenv()


class ChatCallbackHandler(BaseCallbackHandler):
    message = ""

    def on_llm_start(self, *args, **kwargs):
        self.message_box = st.empty()

    def on_llm_end(self, *args, **kwargs):
        save_message(self.message, "ai")

    def on_llm_new_token(self, token, *args, **kwargs):
        self.message += token
        self.message_box.markdown(self.message)


# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")

    # API Key input
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Enter your OpenAI API key",
    )

    # GitHub link
    st.markdown(
        "[View on GitHub](https://github.com/sammitako/llm)",
        unsafe_allow_html=True,
    )

# Initialize session state for messages if not exists
if "messages" not in st.session_state:
    st.session_state["messages"] = []

# Initialize memory if not exists
if "memory" not in st.session_state:
    st.session_state["memory"] = ConversationBufferMemory(
        return_messages=True, memory_key="history"
    )

# Initialize LLM
llm = ChatOpenAI(
    temperature=0.1,
    streaming=True,
    callbacks=[ChatCallbackHandler()],
    openai_api_key=api_key,
)

# Initialize embeddings
embeddings = OpenAIEmbeddings(openai_api_key=api_key)
cache_dir = LocalFileStore("./.cache/")
cached_embeddings = CacheBackedEmbeddings.from_bytes_store(embeddings, cache_dir)

# Text splitter
splitter = CharacterTextSplitter.from_tiktoken_encoder(
    separator="\n",
    chunk_size=600,
    chunk_overlap=100,
)

# RAG prompt template
prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful assistant. Answer questions using only the following context. If you don't know the answer just say you don't know, don't make it up:\n\n{context}",
        ),
        MessagesPlaceholder(variable_name="history"),
        ("human", "{question}"),
    ]
)


def load_memory(_):
    return st.session_state["memory"].load_memory_variables({})["history"]


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data(show_spinner="Processing document...")
def process_document(file):
    file_content = file.read()
    file_path = f"./.cache/files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)

    # Try TextLoader first, fall back to UnstructuredFileLoader
    try:
        loader = TextLoader(file_path)
    except:
        loader = UnstructuredFileLoader(file_path)

    docs = loader.load_and_split(text_splitter=splitter)
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    return vectorstore.as_retriever()


def save_message(message, role):
    st.session_state["messages"].append({"message": message, "role": role})


def send_message(message, role, save=True):
    with st.chat_message(role):
        st.markdown(message)
    if save:
        save_message(message, role)


def paint_history():
    for message in st.session_state["messages"]:
        send_message(
            message["message"],
            message["role"],
            save=False,
        )


st.title("DocumentGPT")

st.markdown(
    """
Welcome!
            
Use this chatbot to ask questions to an AI about your files!

Upload your files on the sidebar.
"""
)

with st.sidebar:
    file = st.file_uploader(
        "Upload a .txt .pdf or .docx file",
        type=["pdf", "txt", "docx"],
    )

if file:
    retriever = process_document(file)
    send_message("I'm ready! Ask away!", "ai", save=False)
    paint_history()

    message = st.chat_input("Ask anything about your file...")
    if message:
        send_message(message, "human")

        # Create the RAG chain with memory
        chain = (
            {
                "context": retriever | RunnableLambda(format_docs),
                "question": RunnablePassthrough(),
                "history": RunnablePassthrough.assign(history=load_memory)["history"],
            }
            | prompt
            | llm
        )

        with st.chat_message("ai"):
            response = chain.invoke(message)
            # Save to memory
            st.session_state["memory"].save_context(
                {"input": message}, {"output": response.content}
            )
else:
    st.session_state["messages"] = []
