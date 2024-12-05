import re
import streamlit as st
from backend.src.agent.graph import graph
from backend.src.agent.utils import _print_event
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


def get_conversation_history():
    history = []
    for msg in st.session_state.messages:
        if isinstance(msg, ChatMessage):
            role = "You" if msg.role == "user" else "Dinebot"
            history.append(f"{role}: {msg.content}")
    return "\n".join(history)


# Function to display approval/denial buttons and return the user's choice
def display_approval_buttons(interrupt_id):
    col1, col2 = st.columns(2)
    approve = col1.button("approve", key=f"approve_{interrupt_id}")
    deny = col2.button("Deny", key=f"deny_{interrupt_id}")
    
    if approve==1:
        return "approve"
    elif deny:
        return "deny"
    return None  # Return None if no button was clicked


# Streamlit UI
st.set_page_config(page_title="DineBot - Your Chef on the Go", page_icon="🍽️")

st.title("🍽️ Welcome to Dinebot")
st.markdown(
    "<h2>Your Personal Restaurant Assistant 🤖</h2>",
    unsafe_allow_html=True,
)
st.markdown("Dinebot can assist you with cab booking, table reservations, and provide restaurant information. Ask me anything!")

if "messages" not in st.session_state:
    st.session_state["messages"] = [ChatMessage(role="assistant", content="How can I help you?")]

if "interrupt_action" not in st.session_state:
    st.session_state["interrupt_action"] = None

if "interrupt_processed" not in st.session_state:
    st.session_state["interrupt_processed"] = False

# Show conversation history
for msg in st.session_state.messages:
    st.chat_message(msg.role).write(msg.content)

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
                    st.markdown(recent_ai_message_content)
                    st.session_state.messages.append(ChatMessage(role="assistant", content=recent_ai_message_content))

        # Interrupt Handling: Show approval/denial widget
        snapshot = graph.get_state(config)
        if snapshot.next:
            interrupt_id = str(uuid.uuid4())

            # Show widget for interrupt approval/denial
            st.markdown("⚠️ The agent is requesting approval for an action.")
            with st.expander("Action Details"):
                st.write(snapshot.next)

            # Display the approval/denial buttons and wait for user input
            interrupt_action = display_approval_buttons(interrupt_id)
            st.write(interrupt_action)
            
            if interrupt_action == "approve":
                st.session_state["interrupt_action"] = "approved"
                st.session_state["interrupt_processed"] = True
                st.write("Approval processed, continuing with the agent.")
                # Call the agent with approval logic
                result = graph.invoke(None, config, stream_mode="values")
                for res_event in result:
                    if isinstance(res_event, dict):
                        messages = res_event.get("messages", {})
                        persona_messages = messages.get(persona, [])
                        for msg in persona_messages:
                            if isinstance(msg, AIMessage) and msg.content.strip():
                                st.markdown(msg.content)
                                st.session_state.messages.append(ChatMessage(role="assistant", content=msg.content))

            elif interrupt_action == "deny":
                st.session_state["interrupt_action"] = "denied"
                st.session_state["interrupt_processed"] = True
                st.write("Denial processed, waiting for reason.")
                
                deny_reason = st.text_input("Reason for denial:", key=f"deny_reason_{interrupt_id}")
                if st.button("Submit Denial", key=f"submit_deny_{interrupt_id}"):
                    st.write("Denial reason submitted.")
                    result = graph.invoke(
                        {
                            "messages": {
                                persona: ToolMessage(
                                    tool_call_id=interrupt_id,
                                    content=f"API call denied by user. Reason: '{deny_reason}'. Continue assisting, accounting for the user's input.",
                                )
                            }
                        },
                        config,
                    )
                    for res_event in result:
                        if isinstance(res_event, dict):
                            messages = res_event.get("messages", {})
                            persona_messages = messages.get(persona, [])
                            for msg in persona_messages:
                                if isinstance(msg, AIMessage) and msg.content.strip():
                                    st.markdown(msg.content)
                                    st.session_state.messages.append(ChatMessage(role="assistant", content=msg.content))

            else:
                st.write("Please click Approve or Deny.")

        # If the interrupt is processed, continue the conversation with the agent
        if st.session_state["interrupt_processed"]:
            st.write("Interrupt processed, continuing the conversation...")
            
            # Continue the conversation with the agent (chat mode)
            conversation_history = get_conversation_history()
            response = graph.stream(
                {"messages": {"General Agent": ("user", conversation_history)}, "current_persona": persona},
                config,
                stream_mode="values",
            )

            for event in response:
                _print_event(event, _printed)
                if isinstance(event, dict):
                    messages = event.get("messages", {})
                    current_persona = event.get("current_persona", "General Agent")
                    persona_messages = messages.get(current_persona, [])

                    # Find and display the most recent AI message
                    recent_ai_message_content = None
                    for msg in reversed(persona_messages):
                        if isinstance(msg, AIMessage) and msg.content.strip():
                            recent_ai_message_content = msg.content
                            break

                    if recent_ai_message_content:
                        st.markdown(recent_ai_message_content)
                        st.session_state.messages.append(ChatMessage(role="assistant", content=recent_ai_message_content))
