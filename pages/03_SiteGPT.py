from langchain.document_loaders import SitemapLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings, CacheBackedEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough, RunnableLambda
from langchain.storage import LocalFileStore
from langchain.callbacks.base import BaseCallbackHandler
from fake_useragent import UserAgent
from bs4 import BeautifulSoup
import streamlit as st
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

st.set_page_config(
    page_title="SiteGPT",
    page_icon="🌐",
)

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # API Key input
    api_key = st.text_input(
        "OpenAI API Key",
        type="password",
        value=os.getenv("OPENAI_API_KEY", ""),
        help="Enter your OpenAI API key"
    )
    
    # GitHub link
    st.markdown(
        "[View on GitHub](https://github.com/sammitako/llm)",
        unsafe_allow_html=True
    )

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

def parse_page(soup: BeautifulSoup):
    header = soup.find("header")
    footer = soup.find("footer")
    if header:
        header.decompose()
    if footer:
        footer.decompose()
    return str(soup.get_text()).replace("\n", "")

# Initialize embeddings and vector store
@st.cache_resource(show_spinner="Loading Cloudflare documentation...")
def load_documentation():
    # Cloudflare documentation sitemaps
    sitemaps = [
        "https://developers.cloudflare.com/ai-gateway/sitemap.xml",
        "https://developers.cloudflare.com/vectorize/sitemap.xml",
        "https://developers.cloudflare.com/workers-ai/sitemap.xml"
    ]
    
    # Load and combine all documentation
    docs = []
    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        chunk_size=1000,
        chunk_overlap=200,
    )
    
    for sitemap in sitemaps:
        loader = SitemapLoader(
            sitemap,
            filter_urls=[
                r"https:\/\/developers.cloudflare.com/ai-gateway.*",
                r"https:\/\/developers.cloudflare.com/vectorize.*",
                r"https:\/\/developers.cloudflare.com/workers-ai.*",
            ],
            parsing_function=parse_page,
        )
        ua = UserAgent()
        loader.headers = {"User-Agent": ua.random}
        docs.extend(loader.load_and_split(splitter))
    
    # Create embeddings with caching
    embeddings = OpenAIEmbeddings(openai_api_key=api_key)
    cache_dir = LocalFileStore("./.cache/cloudflare_docs/")
    cached_embeddings = CacheBackedEmbeddings.from_bytes_store(
        embeddings,
        cache_dir,
    )
    
    vectorstore = FAISS.from_documents(docs, cached_embeddings)
    return vectorstore.as_retriever()

# Initialize the retriever
retriever = load_documentation()

# Initialize the LLMs
llm_for_get_answer = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    openai_api_key=api_key,
)

llm_for_choose_answer = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    openai_api_key=api_key,
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
)

# Create the answer generation prompt
answers_prompt = ChatPromptTemplate.from_template(
    """
    You are an expert at analyzing documentation and providing accurate answers.
    Your task is to answer the user's question using ONLY the provided context.
    If the information isn't available, be honest and say you don't know.
    Never invent or guess information.
                                                  
    After providing your answer, rate its quality on a scale of 0-5:
    - 5: Complete, accurate, and directly answers the question
    - 3-4: Partially answers or needs clarification
    - 0-2: Doesn't answer the question or information is missing

    Context: {context}
                                                  
    Examples:
                                                  
    Question: What is the distance to the moon?
    Answer: The moon orbits Earth at an average distance of 384,400 kilometers.
    Score: 5
                                                  
    Question: What is the temperature on Mars?
    Answer: I don't have enough information to answer this question.
    Score: 0
                                                  
    Now, please answer this question:

    Question: {question}
"""
)

def get_answers(inputs):
    docs = inputs["docs"]
    question = inputs["question"]

    answers_chain = answers_prompt | llm_for_get_answer
    return {
        "answers": [
            {
                "answer": answers_chain.invoke(
                    {
                        "question": question,
                        "context": doc.page_content,
                    }
                ).content,
                "source": doc.metadata["source"],
                "date": doc.metadata.get("lastmod", "Unknown"),
            }
            for doc in docs
        ],
        "question": question,
    }

# Create the answer selection prompt
choose_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """
        You are a documentation expert who excels at finding the most relevant information.
        Your task is to analyze multiple answers and select the most comprehensive one.

        Guidelines for selecting the best answer:
        1. Prioritize answers with higher scores (5 being best)
        2. For equal scores, prefer more recent information
        3. Always include the source URL
        4. If multiple answers are equally good, combine their unique insights

        Remember to:
        - Keep source URLs exactly as provided
        - Maintain the original wording of answers
        - Include the date of the information when available

        Here are the answers to analyze:

        {answers}
        ---
        Example format:
                                                  
        The moon's average distance from Earth is 384,400 kilometers.

        Source: https://example.com
        Last Updated: 2024-03-15
        """
    ),
    ("human", "{question}"),
])

def choose_answer(inputs):
    answers = inputs["answers"]
    question = inputs["question"]

    choose_chain = choose_prompt | llm_for_choose_answer
    condensed = "\n\n".join(
        f"{answer['answer']} \nSource:{answer['source']} \nDate:{answer['date']} \n\n"
        for answer in answers
    )

    return choose_chain.invoke({"answers": condensed, "question": question})

st.title("Cloudflare Documentation Assistant")

st.markdown(
    """
    👋 Welcome to your Cloudflare documentation expert!
            
    I'm here to help you understand:
    - 🤖 AI Gateway - Cloudflare's AI infrastructure solution
    - 🔍 Vectorize - Vector database for AI applications
    - ⚡ Workers AI - Serverless AI computing platform

    Feel free to ask any questions about these products!
    """
)

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What would you like to know about Cloudflare's products?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.session_state.chat_history.append(f"Human: {prompt}")
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)
    
    # Get AI response using the two-stage chain
    with st.chat_message("assistant"):
        chain = (
            {
                "docs": retriever,
                "question": RunnablePassthrough(),
            }
            | RunnableLambda(get_answers)
            | RunnableLambda(choose_answer)
        )
        response = chain.invoke(prompt)
        st.markdown(response.content)
        
        # Add AI response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response.content})
        st.session_state.chat_history.append(f"Assistant: {response.content}")
