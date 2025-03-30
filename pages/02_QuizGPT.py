import json
from langchain.document_loaders import UnstructuredFileLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.callbacks import StreamingStdOutCallbackHandler
import streamlit as st
from langchain.retrievers import WikipediaRetriever
from langchain.schema import BaseOutputParser
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()


class JsonOutputParser(BaseOutputParser):
    def parse(self, text):
        text = text.replace("```", "").replace("json", "")
        return json.loads(text)


output_parser = JsonOutputParser()

st.set_page_config(
    page_title="QuizGPT Turbo",
    page_icon="❓",
)

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
        "[View on GitHub](https://github.com/sammitako/llm)", unsafe_allow_html=True
    )

# Initialize session state
if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = None
if "current_question" not in st.session_state:
    st.session_state.current_question = 0
if "score" not in st.session_state:
    st.session_state.score = 0
if "show_results" not in st.session_state:
    st.session_state.show_results = False

# Function calling setup
function = {
    "name": "create_quiz",
    "description": "function that takes a list of questions and answers and returns a quiz",
    "parameters": {
        "type": "object",
        "properties": {
            "questions": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "question": {
                            "type": "string",
                        },
                        "answers": {
                            "type": "array",
                            "items": {
                                "type": "object",
                                "properties": {
                                    "answer": {
                                        "type": "string",
                                    },
                                    "correct": {
                                        "type": "boolean",
                                    },
                                },
                                "required": ["answer", "correct"],
                            },
                        },
                    },
                    "required": ["question", "answers"],
                },
            }
        },
        "required": ["questions"],
    },
}

llm = ChatOpenAI(
    temperature=0.1,
    model="gpt-3.5-turbo-1106",
    streaming=True,
    callbacks=[StreamingStdOutCallbackHandler()],
    openai_api_key=api_key,
).bind(
    function_call={"name": "create_quiz"},
    functions=[function],
)


def get_difficulty_prompt(difficulty):
    prompts = {
        "Easy": "Create simple, straightforward questions with obvious answers.",
        "Medium": "Create moderately challenging questions that require some thought.",
        "Hard": "Create complex questions with subtle distinctions and detailed answers.",
    }
    return prompts.get(difficulty, "Create moderately challenging questions.")


questions_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
    You are a helpful assistant that is role playing as a teacher.
         
    Based ONLY on the following context make 10 (TEN) questions to test the user's knowledge about the text.
    
    Each question should have 4 answers, three of them must be incorrect and one should be correct.
         
    Use (o) to signal the correct answer.
         
    {difficulty_instruction}
         
    Question examples:
         
    Question: What is the color of the ocean?
    Answers: Red|Yellow|Green|Blue(o)
         
    Question: What is the capital or Georgia?
    Answers: Baku|Tbilisi(o)|Manila|Beirut
         
    Context: {context}
""",
        )
    ]
)


@st.cache_data(show_spinner="Loading file...")
def split_file(file):
    file_content = file.read()
    file_path = f"./.cache/quiz_files/{file.name}"
    with open(file_path, "wb") as f:
        f.write(file_content)
    splitter = CharacterTextSplitter.from_tiktoken_encoder(
        separator="\n",
        chunk_size=600,
        chunk_overlap=100,
    )
    loader = UnstructuredFileLoader(file_path)
    docs = loader.load_and_split(text_splitter=splitter)
    return docs


@st.cache_data(show_spinner="Searching Wikipedia...")
def wiki_search(term):
    retriever = WikipediaRetriever(top_k_results=5)
    docs = retriever.get_relevant_documents(term)
    return docs


def format_docs(docs):
    return "\n\n".join(document.page_content for document in docs)


@st.cache_data(show_spinner="Making quiz...")
def run_quiz_chain(docs, difficulty):
    difficulty_instruction = get_difficulty_prompt(difficulty)
    chain = questions_prompt | llm
    response = chain.invoke(
        {"context": format_docs(docs), "difficulty_instruction": difficulty_instruction}
    )
    return json.loads(response.additional_kwargs["function_call"]["arguments"])


def display_question(question_data):
    st.subheader(f"Question {st.session_state.current_question + 1}")
    st.write(question_data["question"])

    answers = [answer["answer"] for answer in question_data["answers"]]
    selected_answer = st.radio("Select your answer:", answers)

    if st.button("Submit Answer"):
        correct_answer = next(
            answer["answer"] for answer in question_data["answers"] if answer["correct"]
        )
        if selected_answer == correct_answer:
            st.success("Correct!")
            st.session_state.score += 1
        else:
            st.error(f"Incorrect. The correct answer was: {correct_answer}")

        st.session_state.current_question += 1
        st.experimental_rerun()


def display_results():
    total_questions = len(st.session_state.quiz_data["questions"])
    st.subheader("Quiz Results")
    st.write(f"Your score: {st.session_state.score}/{total_questions}")

    if st.session_state.score == total_questions:
        st.balloons()
        st.success("Congratulations! You got all questions correct! 🎉")
    else:
        st.warning("Not all answers were correct. Would you like to retake the quiz?")
        if st.button("Retake Quiz"):
            st.session_state.current_question = 0
            st.session_state.score = 0
            st.experimental_rerun()


st.title("QuizGPT Turbo")

with st.sidebar:
    docs = None
    choice = st.selectbox(
        "Choose what you want to use.",
        ("File", "Wikipedia Article"),
    )

    if choice == "File":
        file = st.file_uploader(
            "Upload a .docx , .txt or .pdf file",
            type=["pdf", "txt", "docx"],
        )
        if file:
            docs = split_file(file)
    else:
        topic = st.text_input("Search Wikipedia...")
        if topic:
            docs = wiki_search(topic)

if not docs:
    st.markdown(
        """
    Welcome to QuizGPT Turbo.
                
    I will make a quiz from Wikipedia articles or files you upload to test your knowledge and help you study.
                
    Get started by uploading a file or searching on Wikipedia in the sidebar.
    """
    )
else:
    difficulty = st.selectbox("Select Quiz Difficulty", ["Easy", "Medium", "Hard"])

    if st.button("Generate Quiz"):
        st.session_state.quiz_data = run_quiz_chain(docs, difficulty)
        st.session_state.current_question = 0
        st.session_state.score = 0
        st.experimental_rerun()

    if st.session_state.quiz_data:
        if st.session_state.current_question < len(
            st.session_state.quiz_data["questions"]
        ):
            display_question(
                st.session_state.quiz_data["questions"][
                    st.session_state.current_question
                ]
            )
        else:
            display_results()
