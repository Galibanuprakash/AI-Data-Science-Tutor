from langchain.memory import ConversationBufferMemory

# Memory for tracking user queries
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
