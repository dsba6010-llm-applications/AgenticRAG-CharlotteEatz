import re
import streamlit as st
from backend.agent.graph import graph
from backend.agent.utils import _print_event
from langchain_core.messages import HumanMessage, AIMessage
from langchain.callbacks.base import BaseCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from langchain.schema import ChatMessage
import uuid
from dotenv import load_dotenv
import os
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

def get_conversation_history():
    history = []
    for msg in st.session_state.messages:
        if isinstance(msg, ChatMessage):
            role = "You" if msg.role == "user" else "Dinebot"
            history.append(f"{role}: {msg.content}")
    return "\n".join(history)

# Load environment variables
load_dotenv()

# Streamlit UI
st.set_page_config(page_title="DineBot - Your Chef on the Go", page_icon="🍽️")

# Side panel for API Key input
with st.sidebar:
    with st.expander("Configuration", expanded=False):
        api_key = st.text_input(
                "Unlock your digital chef’s kitchen! Enter your OpenAI API Key🔑 to get cooking with Dinebot:",
                type="password"
            )
if api_key:
    os.environ["OPENAI_API_KEY"] = api_key
else:
    st.sidebar.warning("Please provide your OpenAI API Key to continue.")
    st.sidebar.info("Go to the 'Configuration' section in the sidebar to enter your API Key.")
    st.markdown("<h3 style='color:red;'>Enter your OpenAI API Key🔑 to unlock your restaurant assistant!</h3>", unsafe_allow_html=True)
    st.stop()
    
    
st.title("🍽️ Welcome to Charlotte Eatz")
st.markdown(
    "<h2>  Hi! Nice to meet you, I am Dinebot 🤖</h2>",
    unsafe_allow_html=True,
)
st.markdown("🤖 I can assist you with cab booking, table reservations, and provide restaurant information. Ask me anything!")

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

if "interrupt_action" not in st.session_state:
    st.session_state["interrupt_action"] = None

if "interrupt_processed" not in st.session_state:
    st.session_state["interrupt_processed"] = False

if "waiting_for_approval" not in st.session_state:
    st.session_state["waiting_for_approval"] = False

# Show conversation history
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

if prompt := st.chat_input():
    st.session_state.messages.append(ChatMessage(role="user", content=prompt))
    st.chat_message("user").write(prompt)

    # Prepare chat history for the agent
    conversation_history = get_conversation_history()

    # Call the agent with user input
    response = graph.stream(
        {"messages": {"General Agent": ("user", conversation_history)}, "current_persona": "General Agent"},
        config,
        stream_mode="values",
    )

    # Process the events and display them
    for event in response:
        _print_event(event, _printed)
        if isinstance(event, dict):
            messages = event.get("messages", {})
            current_persona = event.get("current_persona", "General Agent")
            persona_messages = messages.get(current_persona, [])

            # Find the most recent AIMessage with non-empty content
            recent_ai_message_content = None
            for msg in reversed(persona_messages):
                if isinstance(msg, AIMessage) and msg.content.strip():
                    recent_ai_message_content = msg.content
                    break

            # Check if a recent AI message content was found and display it
            if recent_ai_message_content:
                st.session_state.messages.append(ChatMessage(role="assistant", content=recent_ai_message_content))

    # Display the latest assistant message
    if st.session_state.messages[-1].role == "assistant":
        st.chat_message("assistant").write(st.session_state.messages[-1].content)

    # Interrupt Handling
    snapshot = graph.get_state(config)
    if snapshot.next and not st.session_state.get("waiting_for_approval", False):
        st.session_state["waiting_for_approval"] = True
        st.session_state["interrupt_details"] = snapshot.next
        # Add the approval request to session state messages and display it
        alert_message = (
            "⚠️ The agent is requesting approval for an action.\n\n"
            "### 🛠️ Action Required\n\n"
            "The assistant has identified an action that requires your confirmation. "
            "Please type **'Y'**, **'Yes'**, **'Approve'**, **'Ok'**, or **'Sure'** to approve, "
            "or **'N'**, **'No'**, **'Deny'**, or **'Reject'** to deny."
        )                   
        # Append to session state and show in chat
        st.session_state.messages.append(ChatMessage(role="assistant", content=alert_message))
        st.chat_message("assistant").markdown(alert_message)

    elif st.session_state.get("waiting_for_approval", False):
        approval_response = prompt.lower()

        if approval_response in ['yes', 'y', 'approve', 'ok', 'sure']:
            st.session_state["interrupt_action"] = "approved"
            st.session_state["interrupt_processed"] = True
            st.write("Approval processed, continuing with the agent.")

            # Execute the approved action
            result = graph.invoke(None, config)

            # Process the result and continue the conversation
            conversation_history = get_conversation_history()
            response = graph.stream(
                {"messages": {"General Agent": ("user", conversation_history)}, "current_persona": "General Agent"},
                config,
                stream_mode="values",
            )

            new_messages = []
            for event in response:
                _print_event(event, _printed)
                if isinstance(event, dict):
                    messages = event.get("messages", {})
                    current_persona = event.get("current_persona", "General Agent")
                    persona_messages = messages.get(current_persona, [])

                    for msg in persona_messages:
                        if isinstance(msg, AIMessage) and msg.content.strip():
                            new_messages.append(msg.content)

            # Combine all new messages into a single response
            if new_messages:
                combined_response = " ".join(new_messages)
                st.session_state.messages.append(ChatMessage(role="assistant", content=combined_response))
                st.chat_message("assistant").write(combined_response)

            # Reset interrupt flags
            st.session_state["waiting_for_approval"] = False
            st.session_state["interrupt_processed"] = False
            st.session_state["interrupt_action"] = None
            st.session_state["interrupt_details"] = None

        elif approval_response in ['no', 'n', 'deny', 'reject']:
            st.session_state["interrupt_action"] = "denied"
            st.session_state["interrupt_processed"] = True
            st.write("Action denied. Continuing the conversation.")

            # Remove the last assistant message (approval request) from the session state
            if st.session_state.messages and st.session_state.messages[-1].role == "assistant":
                st.session_state.messages.pop()

            # Instead of sending a message, we'll just invoke the graph with None
            # This should trigger the graph to handle the denial internally
            result = graph.invoke(None, config)

            new_messages = []
            for event in response:
                _print_event(event, _printed)
                if isinstance(event, dict):
                    messages = event.get("messages", {})
                    current_persona = event.get("current_persona", "General Agent")
                    persona_messages = messages.get(current_persona, [])

                    for msg in persona_messages:
                        if isinstance(msg, AIMessage) and msg.content.strip():
                            new_messages.append(msg.content)

            # Combine all new messages into a single response
            if new_messages:
                combined_response = " ".join(new_messages)

                # Check if the new message is a duplicate of the last assistant message
                if not (st.session_state.messages and 
                        st.session_state.messages[-1].role == "assistant" and 
                        st.session_state.messages[-1].content.strip() == combined_response.strip()):
                    st.session_state.messages.append(ChatMessage(role="assistant", content=combined_response))
                    st.chat_message("assistant").write(combined_response)

            # Reset interrupt flags
            st.session_state["waiting_for_approval"] = False
            st.session_state["interrupt_processed"] = False
            st.session_state["interrupt_action"] = None
            st.session_state["interrupt_details"] = None
