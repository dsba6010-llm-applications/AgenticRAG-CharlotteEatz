import re
import streamlit as st
from backend.agent.graph import graph
from backend.agent.utils import _print_event
from langchain_core.messages import ToolMessage, HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import ChatMessage

import uuid
from dotenv import load_dotenv
import langgraph
from langgraph.pregel.io import AddableValuesDict

# Initialize unique thread ID for the session
thread_id = str(uuid.uuid4())

# Configuration dictionary
config = {
    "configurable": {
        "thread_id": thread_id,
    }
}

# Set to track printed events
_printed = set()

class StreamHandler(BaseCallbackHandler):
    def __init__(self, container, initial_text=""):
        self.container = container
        self.text = initial_text

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.text += token
        self.container.markdown(self.text)
        
from typing import TypedDict, Annotated, Dict, Any, List

# from langgraph.graph import add_messages
from backend.src.agent.utils import add_messages_to_dict
from langchain_core.messages import AnyMessage


class State(TypedDict):
    messages: Annotated[Dict[str, list[AnyMessage]], add_messages_to_dict]
    salesforce_case: Dict[str, Any]
    salesforce_cases: List[Dict[str, Any]]
    current_persona: str
    
# Function to concatenate chat history into a single string
def get_conversation_history():
    history = []
    for msg in st.session_state.messages:
        if isinstance(msg, ChatMessage):
            role = "You" if msg.role == "user" else "Dinebot"
            history.append(f"{role}: {msg.content}")
    return "\n".join(history)

# Streamlit UI
st.set_page_config(page_title="DineBot - Your Cheif on the go", page_icon="🍽️")

st.title("🍽️ Welcome to Dinebot")
st.markdown(
    """
    <style>
    .stApp {
        background-color: #2856b0;
    }
    .title {
        color: #4c79a1;
    }
    .header-text {
        font-family: 'Helvetica', sans-serif;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.image("image2.png", use_column_width=True)


st.markdown(
    "<h2 class='header-text'>Your Personal Restaurant Assistant 🤖</h2>",
    unsafe_allow_html=True,
)
st.markdown(
    "Dinebot can assist you with cab booking, table reservations, and provide restaurant information. Ask me anything!"
)

        
if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]       
    
for msg in st.session_state.messages:
    if isinstance(msg, ChatMessage):
        st.chat_message(msg.role).write(msg.content)
    else:
        print(f"Skipping invalid message: {msg}")

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)
    
    # Prepare chat history for the agent
    conversation_history = get_conversation_history()
    
    with st.chat_message("assistant"):
        stream_handler = StreamHandler(st.empty())
        persona = "General Agent"
        # Call the agent with user input
        response = graph.stream(
            {"messages": {"General Agent": ("user", conversation_history)}, "current_persona": persona},
            config,
            stream_mode="values",
        )
        # Process the events and display them
        for event in response:
            _print_event(event, _printed)
            if isinstance(event, dict):
                # Retrieve the 'messages' dictionary and the 'current_persona'
                messages = event.get("messages", {})
                current_persona = event.get("current_persona", "General Agent")

                # Get the list of messages for the current persona
                persona_messages = messages.get(current_persona, [])
                
                # Find the most recent AIMessage with non-empty content
                recent_ai_message_content = None
                for msg in reversed(persona_messages):
                    if isinstance(msg, AIMessage) and msg.content.strip():
                        recent_ai_message_content = msg.content
                        break
                    
                # Check if a recent AI message content was found and display it
                if recent_ai_message_content:
                    st.markdown(recent_ai_message_content)
                    st.session_state.messages.append( ChatMessage(role="assistant", content=recent_ai_message_content))
            else:
            # Log or handle cases where 'event' is not a dictionary, if needed
                print("Unexpected event type:", type(event))

        
            