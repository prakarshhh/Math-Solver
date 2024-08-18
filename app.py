import streamlit as st
from langchain_groq import ChatGroq
from langchain.chains import LLMMathChain, LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.agents import Tool, initialize_agent, AgentType
from langchain.callbacks import StreamlitCallbackHandler
import speech_recognition as sr
import json
import os

# Define the file path for chat history
history_file = "chat_history.json"

# Define functions for history management
def save_history():
    """Save chat history to a JSON file."""
    if "messages" in st.session_state:
        with open(history_file, "w") as f:
            json.dump(st.session_state["messages"], f)
        st.success("Chat history saved successfully.")

def clear_history():
    """Clear chat history from session state and delete the JSON file."""
    st.session_state["messages"] = []
    st.session_state['intro_message_shown'] = False
    if os.path.exists(history_file):
        os.remove(history_file)
    st.write("Chat history cleared.")  # Confirm clearing in the main content

def load_history():
    """Load chat history from the JSON file."""
    if os.path.exists(history_file):
        with open(history_file, "r") as f:
            st.session_state["messages"] = json.load(f)

# Set up the Streamlit app with custom layout
st.set_page_config(page_title="Math Problem Solver & Data Search Assistant", page_icon="🧮", layout="wide")

st.title("Math Problem Solver Using Google Gemma 2")

# Custom CSS for styling
st.markdown("""
    <style>
    body {
        font-family: 'Arial', sans-serif;
        background-color: #f4f4f9;
        color: #333;
    }
    .main-container {
        background-color: #ffffff;
        padding: 2rem;
        border-radius: 15px;
        box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
    }
    .title {
        font-size: 2.5rem;
        color: #1e3a8a;
        margin-bottom: 1rem;
        text-align: center;
    }
    .sidebar .sidebar-content {
        background-color: #e2e8f0;
        border-radius: 10px;
        padding: 1rem;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
    }
    .text-input, .stButton {
        width: 100%;
        margin-top: 1rem;
    }
    .stChatMessage {
        border-radius: 12px;
        padding: 1rem;
        margin: 0.5rem 0;
        background-color: #edf2f7;
        color: #2d3748;
        box-shadow: 0px 4px 10px rgba(0, 0, 0, 0.1);
        text-align: left;
        display: block;
        transition: background-color 0.3s ease, transform 0.3s ease;
        opacity: 0;
        animation: fadeIn 0.5s forwards;
    }
    .stChatMessage.user {
        background-color: #3182ce;
        color: white;
    }
    .stChatMessage.assistant {
        background-color: #f6e05e;
        color: #2d3748;
    }
    @keyframes fadeIn {
        to {
            opacity: 1;
        }
    }
    .stButton {
        width: 100%;
        margin-top: 1rem;
        background-color: #1e3a8a;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem;
        cursor: pointer;
        transition: background-color 0.3s ease;
    }
    .stButton:hover {
        background-color: #2b6cb0;
    }
    .stSidebar {
        position: fixed;
        top: 0;
        left: -270px;
        width: 270px;
        height: 100%;
        background-color: #f7fafc;
        padding: 1rem;
        border-right: 1px solid #ddd;
        box-shadow: 4px 0 10px rgba(0, 0, 0, 0.1);
        transition: left 0.5s ease-in-out;
    }
    .stSidebar.show {
        left: 0;
    }
    .stMain {
        margin-left: 300px;
        transition: margin-left 0.5s ease;
    }
    .stButton.voice-search {
        animation: bounce 1s infinite;
    }
    @keyframes bounce {
        0%, 20%, 50%, 80%, 100% {
            transform: translateY(0);
        }
        40% {
            transform: translateY(-20px);
        }
        60% {
            transform: translateY(-10px);
        }
    }
    </style>
""", unsafe_allow_html=True)

# Sidebar for API Key input and history management
with st.sidebar:
    st.header("Settings")
    groq_api_key = st.text_input(label="Groq API Key", type="password", placeholder="Enter your Groq API Key")

    if groq_api_key:
        # Check if there are messages in history to show "Clear History" button
        if "messages" in st.session_state and st.session_state["messages"]:
            if st.button("Clear History"):
                clear_history()

        if st.button("Show History"):
            st.session_state.show_history_buttons = not st.session_state.get('show_history_buttons', False)

        # Show history management buttons if applicable
        if st.session_state.get('show_history_buttons', False):
            # Only show the header if there are messages
            if "messages" in st.session_state and st.session_state["messages"]:
                st.write('<div class="history-header" style="color: #1e3a8a; font-size: 1.2rem;">Here’s a glimpse into your past queries:</div>', unsafe_allow_html=True)
                st.write('<div class="history-message">', unsafe_allow_html=True)
                for msg in st.session_state["messages"]:
                    st.write(f'<div class="stChatMessage {msg["role"]}">{msg["content"]}</div>', unsafe_allow_html=True)
                st.write('</div>', unsafe_allow_html=True)
            else:
                st.write("No chats are present.")

# Load history when app starts
if "messages" not in st.session_state:
    st.session_state["messages"] = []
load_history()

# Check if Groq API Key is provided
if not groq_api_key:
    st.info("Please add your Groq API Key to continue.")
    st.stop()

# Initialize Groq model
llm = ChatGroq(model="Gemma2-9b-It", groq_api_key=groq_api_key)

# Initialize tools
wikipedia_wrapper = WikipediaAPIWrapper()
wikipedia_tool = Tool(
    name="Wikipedia",
    func=wikipedia_wrapper.run,
    description="Search the Internet for information on various topics."
)

math_chain = LLMMathChain.from_llm(llm=llm)
calculator = Tool(
    name="Calculator",
    func=math_chain.run,
    description="Answer math-related questions. Provide a mathematical expression."
)

prompt = """
You are an agent tasked with solving users' mathematical questions. Logically arrive at the solution and provide a detailed explanation
displayed point-wise for the question below:
Question: {question}
Answer:
"""

prompt_template = PromptTemplate(
    input_variables=["question"],
    template=prompt
)

chain = LLMChain(llm=llm, prompt=prompt_template)

reasoning_tool = Tool(
    name="Reasoning Tool",
    func=chain.run,
    description="Answer logic-based and reasoning questions."
)

# Initialize the agent
assistant_agent = initialize_agent(
    tools=[wikipedia_tool, calculator, reasoning_tool],
    llm=llm,
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
    handle_parsing_errors=True
)

# Input and submission
with st.container():
    # Voice search functionality
    st.subheader("Or ask using your voice")
    if st.button("Start Voice Search"):
        st.session_state["voice_input"] = ""
        recognizer = sr.Recognizer()
        with sr.Microphone() as source:
            st.info("Listening...")
            audio = recognizer.listen(source)
            st.info("Processing...")
            try:
                st.session_state["voice_input"] = recognizer.recognize_google(audio)
                st.success(f"Voice Input: {st.session_state['voice_input']}")
            except sr.UnknownValueError:
                st.error("Sorry, I could not understand the audio.")
            except sr.RequestError:
                st.error("Sorry, there was an error with the speech recognition service.")

    # Text area for manual input
    question = st.text_area("Enter your question:", value=st.session_state.get("voice_input", ""))

    # Button for finding answer
    if st.button("Find My Answer"):
        if question:
            with st.spinner("Generating response..."):
                st.session_state.messages.append({"role": "user", "content": question})
                st.markdown(f'<div class="stChatMessage user">{question}</div>', unsafe_allow_html=True)
                response = assistant_agent({"question": question})
                st.markdown(f'<div class="stChatMessage assistant">{response}</div>', unsafe_allow_html=True)
                st.session_state.messages.append({"role": "assistant", "content": response})
                save_history()
        else:
            st.warning("Please enter a question.")

# Placeholder for the footer
st.markdown("""
    <footer style="text-align: center; padding: 1rem; background-color: black; border-top: 1px solid #ddd;">
        <p style="">Made with Streamlit</p>
    </footer>
""", unsafe_allow_html=True)
