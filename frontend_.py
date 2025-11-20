#------------------------------------------------------------Imports-----------------------------------------------------------

import streamlit as st                                   
from backend_ import chatbots, retrieve_all_threads 
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
import uuid 
from rag import add_document 
from PyPDF2 import PdfReader

#--------------------------------------------------------------Functions---------------------------------------------------------
def generate_thread_id(): 
    """Generate a unique ID for each chat session"""
    return uuid.uuid4()    

def reset_chat():   
    """
    Start a fresh chat session
    Clears history, assigns a new thread ID and registers
    the new thread so it appears in the sidebar
    """     
    thread_id = generate_thread_id()       
    st.session_state.thread_id = thread_id 
    add_thread(thread_id)               
    st.session_state.history = []          

def add_thread(thread_id):   
    """Store the thread ID in state if not already present."""                          
    if thread_id not in st.session_state.chat_thread:  
        st.session_state.chat_thread.append(thread_id)

def load_conversation(thread_id):
    """
    Retrieve saved message history from backend storage
    based on the selected thread ID
    """         
    state = chatbots.get_state(config={'configurable': {'thread_id': thread_id}})
    values = getattr(state, "values", {})  
    return values.get("messages", [])

def read_uploaded_file(uploaded):
    """
    Extract raw text from uploaded txt or pdf files
    Used for adding custom documents into RAG memory
    """
    if uploaded is None:
        return None

    if uploaded.type == "text/plain":
        return uploaded.read().decode("utf-8")

    if uploaded.type == "application/pdf":
        reader = PdfReader(uploaded)
        text = ""
        for page in reader.pages:
            text += page.extract_text() or ""
        return text

    return None

#----------------------------------------------------------Sessions---------------------------------------------------------------------

# Session state is a dictionary and it requires keys
# they ensure session_state keys always exist

if "history" not in st.session_state:
    st.session_state.history = []

if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "chat_thread" not in st.session_state:
    st.session_state.chat_thread = retrieve_all_threads()

add_thread(st.session_state.thread_id)

#-----------------------------------------------------------Sidebar----------------------------------------------------------------------
# Using Sidebar

with st.sidebar:
    st.title("AI Agent")


    if st.button("New Chat"):
        reset_chat()

    # List of all stored chats
    st.header("Chats")
    for thread_id in st.session_state.chat_thread[::-1]:
        if st.button(str(thread_id)):
            st.session_state.thread_id = thread_id
            msgs = load_conversation(thread_id)

            # Converting messages into dictinaries
            temp = []
            for m in msgs:
                role = "user" if isinstance(m, HumanMessage) else "assistant"
                temp.append({"role": role, "content": m.content})

            st.session_state.history = temp

    # file upload for rag
    uploaded = st.file_uploader("", type=["txt", "pdf"])

    if uploaded:
        extracted_text = read_uploaded_file(uploaded)

        if extracted_text:
            add_document(extracted_text)
            st.success("Added to knowledge base!")
        else:
            st.error("Could not read file.")

#---------------------------------------------------------Chat Interface Rendering and Message Handling------------------------------------
# Display chats
for message in st.session_state.history:
    with st.chat_message(message["role"]):
        st.write(message["content"])

# Chat input and Streaming
ask = st.chat_input("Ask Anything...")

CONFIG = {"configurable": {"thread_id": st.session_state.thread_id}}

if ask:
    # Show user's message
    st.session_state["history"].append({"role": "user", "content": ask})
    with st.chat_message("user"):
        st.text(ask)

    CONFIG = {
        "configurable": {"thread_id": st.session_state["thread_id"]},
        "metadata": {"thread_id": st.session_state["thread_id"]},
        "run_name": "chat_turn",
    }

    # Assistant streaming block
    with st.chat_message("assistant"):
        # Use a mutable holder so the generator can set/modify it
        status_holder = {"box": None}

        def ai_only_stream():
            for message_chunk, metadata in chatbots.stream(
                {"messages": [HumanMessage(content=ask)]},
                config=CONFIG,
                stream_mode="messages",
            ):
                # Lazily create & update the SAME status container when any tool runs
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")
                    if status_holder["box"] is None:
                        status_holder["box"] = st.status(
                            f"🔧 Using `{tool_name}` …", expanded=True
                        )
                    else:
                        status_holder["box"].update(
                            label=f"🔧 Using `{tool_name}` …",
                            state="running",
                            expanded=True,
                        )

                # Stream ONLY assistant tokens
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

        ai_message = st.write_stream(ai_only_stream())

        # Finalize only if a tool was actually used
        if status_holder["box"] is not None:
            status_holder["box"].update(
                label="✅ Tool finished", state="complete", expanded=False
            )

    # Save assistant message
    st.session_state["history"].append(
        {"role": "assistant", "content": ai_message}
    )

