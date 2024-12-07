import streamlit as st
import uuid
from backend.src.agent.graph import graph
from backend.src.agent.utils import _print_event
from langchain_core.messages import ToolMessage

def main():
    st.title("General Agent Chat")

    # Initialize session state
    if 'thread_id' not in st.session_state:
        st.session_state.thread_id = str(uuid.uuid4())
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if '_printed' not in st.session_state:
        st.session_state._printed = set()

    config = {
        "configurable": {
            "thread_id": st.session_state.thread_id,
        }
    }

    # Display chat history
    for message in st.session_state.chat_history:
        st.text(message)

    # User input
    user_query = st.text_input("Enter your query:", key="user_input")

    if st.button("Send"):
        if user_query:
            persona = "General Agent"
            st.session_state.chat_history.append(f"You: {user_query}")

            events = graph.stream(
                {
                    "messages": {"General Agent": ("user", user_query)},
                    "current_persona": persona,
                },
                config,
                stream_mode="values",
            )

            for event in events:
                _print_event(event, st.session_state._printed)
                st.session_state.chat_history.append(f"Agent: {event}")

            snapshot = graph.get_state(config)
            while snapshot.next:
                # We have an interrupt! The agent is trying to use a tool
                st.write("The agent is requesting to use a tool:")
                st.write(event["messages"][-1].tool_calls[0])
                
                user_approval = st.radio(
                    "Do you approve of the above action?",
                    options=["Yes", "No"],
                    key=f"approval_{len(st.session_state.chat_history)}"
                )

                if user_approval == "Yes":
                    result = graph.invoke(None, config, stream_mode="values")
                else:
                    user_input = st.text_input(
                        "Please explain your requested changes:",
                        key=f"changes_{len(st.session_state.chat_history)}"
                    )
                    if user_input:
                        result = graph.invoke(
                            {
                                "messages": {
                                    persona: ToolMessage(
                                        tool_call_id=event["messages"][-1].tool_calls[0]["id"],
                                        content=f"API call denied by user. Reasoning: '{user_input}'. Continue assisting, accounting for the user's input.",
                                    )
                                }
                            },
                            config,
                        )

                _print_event(result, st.session_state._printed)
                st.session_state.chat_history.append(f"Agent: {result}")
                snapshot = graph.get_state(config)

            # Clear the input box after sending
            st.session_state.user_input = ""

if __name__ == "__main__":
    main()