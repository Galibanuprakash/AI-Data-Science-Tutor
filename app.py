import streamlit as st
from langchain.memory import ConversationBufferMemory
from langchain.chat_models import ChatGoogleGenerativeAI
from langchain.schema import SystemMessage, HumanMessage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GOOGLE_API_KEY = os.getenv("AIzaSyCFbnID7J4KnD-hoveRc37CEx_MV9eXUEk")

# Initialize the model
llm = ChatGoogleGenerativeAI(model="gemini-1.5-pro", google_api_key=GOOGLE_API_KEY)

# Streamlit UI
st.set_page_config(page_title="AI Data Science Tutor", page_icon="🧠")
st.title("🧠 AI Conversational Data Science Tutor")

# Memory Initialization
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory()

# Chat history container
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage(content="You are an AI tutor that only answers Data Science questions.")
    ]

# Display chat history
for msg in st.session_state.messages:
    if isinstance(msg, SystemMessage):
        continue
    role = "user" if isinstance(msg, HumanMessage) else "assistant"
    with st.chat_message(role):
        st.markdown(msg.content)

# User input
user_input = st.chat_input("Ask a Data Science question...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    # Store user message
    st.session_state.messages.append(HumanMessage(content=user_input))
    
    # Generate response
    with st.spinner("Thinking..."):
        response = llm.invoke(st.session_state.messages)

    # Store assistant response
    st.session_state.messages.append(response)

    # Display response
    with st.chat_message("assistant"):
        st.markdown(response.content)
