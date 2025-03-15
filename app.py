import streamlit as st
import google.generativeai as genai
from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from config import GOOGLE_API_KEY
from memory import memory

# Set up the API Key
genai.configure(api_key=GOOGLE_API_KEY)

# Initialize LangChain AI Model
chat_model = ChatOpenAI(model_name="gemini-1.5-pro", temperature=0.7)

# Create Conversation Chain with Memory
conversation = ConversationChain(
    llm=chat_model,
    memory=memory
)

# Streamlit UI
st.set_page_config(page_title="AI Data Science Tutor", layout="wide")
st.title("🤖 AI Data Science Tutor")
st.write("Ask your data science doubts!")

# Chat Interface
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# User Input
query = st.chat_input("Ask a Data Science question...")

if query:
    # Store user query
    st.session_state.messages.append({"role": "user", "content": query})

    # Get AI Response
    response = conversation.predict(input=query)

    # Store response
    st.session_state.messages.append({"role": "assistant", "content": response})

    # Display AI Response
    with st.chat_message("assistant"):
        st.markdown(response)
