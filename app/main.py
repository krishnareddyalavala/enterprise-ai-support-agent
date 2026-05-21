import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st

from src.rag_pipeline import (
    load_documents,
    split_documents,
    create_vector_store,
    retrieve_documents,
    generate_response
)

from src.evaluation import evaluate_grounding

from mcp_tools.tools import create_support_ticket, check_ticket_status


st.title("Enterprise AI Support Agent")


if "tickets" not in st.session_state:
    st.session_state.tickets = {}

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []


st.subheader("Upload Enterprise Document")

uploaded_file = st.file_uploader(
    "Upload a PDF policy document",
    type=["pdf"]
)

if uploaded_file is not None:
    file_path = os.path.join("data", uploaded_file.name)

    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"Uploaded {uploaded_file.name} successfully.")


docs = load_documents()
chunks = split_documents(docs)
vector_store = create_vector_store(chunks)


st.subheader("Ask Enterprise Knowledge Base")

query = st.text_input("Ask a company question")

submit = st.button("Submit")

if submit:

    if not query:
        st.warning("Please enter a question before submitting.")

    else:
        with st.spinner("Searching enterprise knowledge base..."):

            retrieved_docs = retrieve_documents(vector_store, query)

            response = generate_response(query, retrieved_docs)

        st.subheader("AI Response")
        st.write(response)

        grounding_result = evaluate_grounding(
            response,
            retrieved_docs
        )

        st.subheader("Grounding Evaluation")

        if grounding_result == "Grounded":
            st.success(grounding_result)

        else:
            st.error(grounding_result)

        st.subheader("Sources Used")

        for i, doc in enumerate(retrieved_docs, start=1):
            st.markdown(f"**Source {i}:**")
            st.write(doc.page_content)
            st.caption(f"File: {doc.metadata.get('source', 'Unknown')}")

        st.session_state.chat_history.append(
            {
                "question": query,
                "answer": response
            }
        )


st.divider()

st.subheader("Support Ticket Tools")

issue = st.text_area("Describe your IT issue")

create_ticket = st.button("Create Support Ticket")

if create_ticket:

    if not issue:
        st.warning("Please describe the issue before creating a ticket.")

    else:
        ticket = create_support_ticket(
            issue,
            st.session_state.tickets
        )

        st.success(ticket["message"])
        st.write(f"Ticket ID: {ticket['ticket_id']}")
        st.write(f"Status: {ticket['status']}")


ticket_id_input = st.text_input("Enter Ticket ID to check status")

check_status = st.button("Check Ticket Status")

if check_status:

    if not ticket_id_input:
        st.warning("Please enter a ticket ID.")

    else:
        status = check_ticket_status(
            ticket_id_input,
            st.session_state.tickets
        )

        if "error" in status:
            st.error(status["error"])

        else:
            st.write(f"Issue: {status['issue']}")
            st.write(f"Status: {status['status']}")


st.divider()

st.subheader("Conversation History")

clear_chat = st.button("Clear Conversation History")

if clear_chat:
    st.session_state.chat_history = []
    st.success("Conversation history cleared.")

for chat in st.session_state.chat_history:

    st.markdown(f"**User:** {chat['question']}")

    st.markdown(f"**AI:** {chat['answer']}")
