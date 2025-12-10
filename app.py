import os
import streamlit as st
from dotenv import load_dotenv, find_dotenv

from create_memory_llm import (
    load_pdfs_from_streamlit,
    create_chunks,
    get_embedding_model,
    build_faiss_index,
    DB_FAISS_PATH,
)
from connect_memory_llm import get_qa_chain


# ---------- BASIC CONFIG ----------

load_dotenv(find_dotenv())

st.set_page_config(
    page_title="Document QA Chatbot",
    page_icon="🤖",
    layout="wide",
)

st.title("📄 Document QA Chatbot")
st.caption("Upload PDFs, build a knowledge base, and chat with your documents.")


# ---------- SIDEBAR: STATUS & CONFIG ----------

st.sidebar.title("🔧 Configuration & Status")

hf_token = os.getenv("HF_TOKEN")
if hf_token:
    st.sidebar.success("✓ HF_TOKEN found.")
else:
    st.sidebar.warning("⚠ HF_TOKEN not set. LLM may fail to load.")

index_exists = os.path.exists(DB_FAISS_PATH)
if index_exists:
    st.sidebar.success(f"✓ Vector store found at `{DB_FAISS_PATH}`")
else:
    st.sidebar.info("ℹ No vector store found yet. Please upload PDFs and build index.")


# ---------- SESSION STATE ----------

if "qa_chain" not in st.session_state:
    st.session_state.qa_chain = None

if "messages" not in st.session_state:
    st.session_state.messages = []  # for UI display only


# ---------- TABS: (1) Build Index  (2) Chat ----------

tab_build, tab_chat = st.tabs(["📥 Upload & Build Index", "💬 Chat with Docs"])


# === TAB 1: UPLOAD & BUILD INDEX ===

with tab_build:
    st.subheader("1️⃣ Upload PDFs and Build Knowledge Base")

    uploaded_files = st.file_uploader(
        "Upload one or more PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="These PDFs will be processed, chunked, embedded, and stored in a FAISS index.",
    )

    if uploaded_files:
        st.write(f"Selected **{len(uploaded_files)}** file(s):")
        for f in uploaded_files:
            st.write("•", f.name)

    if st.button("🚀 Build / Rebuild Vector Store"):
        if not uploaded_files:
            st.warning("Please upload at least one PDF before building the index.")
        else:
            with st.spinner("Processing PDFs and building FAISS index..."):
                try:
                    # 1) Load PDFs from Streamlit
                    documents = load_pdfs_from_streamlit(uploaded_files)

                    # 2) Chunk
                    chunks = create_chunks(documents)

                    # 3) Embeddings
                    embedding_model = get_embedding_model()

                    # 4) Build & Save FAISS index
                    build_faiss_index(chunks, embedding_model, DB_FAISS_PATH)

                    # 5) Create QA chain and store in session
                    st.session_state.qa_chain = get_qa_chain()

                    st.success("✅ Vector store built and QA chain initialized!")

                except Exception as e:
                    st.error(f"❌ Error while building index: {e}")


# === TAB 2: CHAT WITH DOCS ===

with tab_chat:
    st.subheader("2️⃣ Chat with Your Documents")

    # Ensure QA chain is available
    if st.session_state.qa_chain is None:
        # Try to create it if an index already exists
        if index_exists:
            try:
                st.session_state.qa_chain = get_qa_chain()
                st.info("Loaded existing vector store and initialized QA chain.")
            except Exception as e:
                st.error(f"❌ Failed to load QA chain: {e}")
        else:
            st.warning("No vector store available. Please first upload PDFs and build the index in the 'Upload & Build Index' tab.")
    
    qa_chain = st.session_state.qa_chain

    # If still no chain, stop here
    if qa_chain is None:
        st.stop()

    # Display past messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Chat input
    user_query = st.chat_input("Ask a question about your documents...")

    if user_query:
        # Show user message
        st.session_state.messages.append({"role": "user", "content": user_query})
        with st.chat_message("user"):
            st.markdown(user_query)

        # Get response from QA chain
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    response = qa_chain.invoke({"query": user_query})
                    answer = response["result"]
                    source_docs = response.get("source_documents", [])
                except Exception as e:
                    answer = f"❌ Error during retrieval or generation: {e}"
                    source_docs = []

            st.markdown(answer)
            st.session_state.messages.append({"role": "assistant", "content": answer})

            # Show sources (if any)
            if source_docs:
                with st.expander("📚 Show source documents"):
                    for i, doc in enumerate(source_docs, start=1):
                        st.markdown(f"**Source {i}** — {doc.metadata.get('source', 'Unknown')}")
                        st.write(doc.page_content[:800] + "...")
                        st.write("---")
