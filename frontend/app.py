import streamlit as st
from api_client import check_health, query_documents

st.set_page_config(
    page_title="PyTorch Docs RAG Assistant",
    page_icon="🔥",
    layout="centered",
)

st.title(" PyTorch Docs RAG Assistant")
st.caption("Ask questions about PyTorch documentation and get grounded, cited answers.")

# --- Sidebar ---
with st.sidebar:
    st.header("ℹ️ About")
    st.markdown(
        "This assistant uses **Retrieval-Augmented Generation (RAG)** to answer "
        "questions from official PyTorch documentation.\n\n"
        "Answers are grounded in the retrieved context and include source citations."
    )
    st.divider()
    if check_health():
        st.success("✅ Backend connected")
    else:
        st.error("❌ Backend unavailable")

# --- Chat History ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander("📄 Sources"):
                for source in message["sources"]:
                    st.markdown(f"- {source}")

# --- Chat Input ---
if prompt := st.chat_input("Ask a question about PyTorch..."):
    # Show user message
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # Get response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                result = query_documents(prompt)
                answer = result["answer"]
                sources = result["sources"]

                st.markdown(answer)
                if sources:
                    with st.expander("📄 Sources"):
                        for source in sources:
                            st.markdown(f"- {source}")

                st.session_state.messages.append(
                    {"role": "assistant", "content": answer, "sources": sources}
                )
            except Exception as e:
                error_msg = (
                    "⚠️ Sorry, something went wrong while processing your question. "
                    "Please make sure the backend is running and try again."
                )
                st.error(error_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": error_msg}
                )
