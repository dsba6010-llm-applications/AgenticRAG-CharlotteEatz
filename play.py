from langchain.chains import ConversationChain
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory

# Set up the chat model
llm = ChatOpenAI(model="gpt-4", temperature=0.7)

# Set up memory to track conversation history
memory = ConversationBufferMemory()

# Create a ConversationChain
conversation = ConversationChain(llm=llm, memory=memory)

# Simulate a conversation
response = conversation.run("Hello, who are you?")
print(response)  # AI introduces itself

response = conversation.run("What can you do?")
print(response)  # AI describes its capabilities

response = conversation.run("What did I ask you first?")
print(response)  # AI recalls the initial query based on conversation history
