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
                    st.session_state.messages.append(ChatMessage(role="assistant", content=recent_ai_message_content))
            else:
                # Log or handle cases where 'event' is not a dictionary, if needed
                print("Unexpected event type:", type(event))

        # Interrupt Handling
        if "interrupt_id" not in st.session_state:
            st.session_state["interrupt_id"] = None
        if "handled_interrupts" not in st.session_state:
            st.session_state["handled_interrupts"] = set()
        if "interrupt_action_processed" not in st.session_state:
            st.session_state["interrupt_action_processed"] = False

        snapshot = graph.get_state(config)

        while snapshot.next and not st.session_state["interrupt_action_processed"]:
            interrupt_id = (
                snapshot.next[0]
                if isinstance(snapshot.next, tuple) and len(snapshot.next) > 0
                else str(uuid.uuid4())
            )

            # Skip already handled interrupts
            if interrupt_id in st.session_state["handled_interrupts"]:
                snapshot = graph.get_state(config)  # Update snapshot for the next interrupt
                continue

            # Save the interrupt ID to session state
            st.session_state["interrupt_id"] = interrupt_id

            # Show the widget once
            if "interrupt_widget_shown" not in st.session_state:
                st.markdown("⚠️ The agent is requesting approval for an action.")
                with st.expander("Action Details"):
                    st.write(snapshot.next)

                col1, col2 = st.columns(2)
                with col1:
                    approve = st.button("Approve", key=f"approve_{interrupt_id}")
                with col2:
                    deny = st.button("Deny", key=f"deny_{interrupt_id}")
                st.session_state["interrupt_widget_shown"] = True

            # Process approval
            if approve:
                st.write("User approved!")
                st.session_state["handled_interrupts"].add(interrupt_id)
                st.session_state["interrupt_action_processed"] = True

                # Call the agent with approval logic
                result = graph.invoke(None, config, stream_mode="values")
                interrupt_response = ""
                for res_event in result:
                    if isinstance(res_event, dict):
                        messages = res_event.get("messages", {})
                        persona_messages = messages.get(persona, [])
                        for msg in persona_messages:
                            if isinstance(msg, AIMessage) and msg.content.strip():
                                interrupt_response += msg.content + "\n"

                st.markdown(interrupt_response.strip())
                st.session_state.messages.append(
                    ChatMessage(role="assistant", content=interrupt_response.strip())
                )
                break

            # Process denial
            if deny:
                st.write("User denied!")
                st.session_state["handled_interrupts"].add(interrupt_id)
                st.session_state["interrupt_action_processed"] = True

                # Capture denial reason
                deny_reason = st.text_input("Reason for denial:", key=f"deny_reason_{interrupt_id}")
                if st.button("Submit Denial", key=f"submit_deny_{interrupt_id}"):
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
                    interrupt_response = ""
                    for res_event in result:
                        if isinstance(res_event, dict):
                            messages = res_event.get("messages", {})
                            persona_messages = messages.get(persona, [])
                            for msg in persona_messages:
                                if isinstance(msg, AIMessage) and msg.content.strip():
                                    interrupt_response += msg.content + "\n"

                    st.markdown(interrupt_response.strip())
                    st.session_state.messages.append(
                        ChatMessage(role="assistant", content=interrupt_response.strip())
                    )
                    break

            # Update snapshot for the next iteration
            snapshot = graph.get_state(config)
