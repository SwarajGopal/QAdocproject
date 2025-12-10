import tempfile

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS


# ---------- CONFIG ----------

DB_FAISS_PATH = "vectorstore/db_faiss"


# ---------- PDF LOADING FROM STREAMLIT ----------

def load_pdfs_from_streamlit(uploaded_files):
    """
    Load PDFs coming from Streamlit's st.file_uploader.
    Each UploadedFile is saved to a temporary file and loaded via PyPDFLoader.
    """
    all_docs = []

    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getbuffer())
            tmp_path = tmp_file.name

        loader = PyPDFLoader(tmp_path)
        docs = loader.load()

        # Store original filename in metadata
        for d in docs:
            d.metadata["source"] = uploaded_file.name

        all_docs.extend(docs)

    if not all_docs:
        raise ValueError("No pages extracted from uploaded PDFs.")

    print(f"Loaded {len(all_docs)} pages from uploaded PDFs")
    return all_docs


# ---------- CHUNKING ----------

def create_chunks(extracted_data, chunk_size: int = 500, chunk_overlap: int = 50):
    """
    Split documents into overlapping text chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    text_chunks = text_splitter.split_documents(extracted_data)

    print(
        f"Created {len(text_chunks)} chunks "
        f"(chunk_size={chunk_size}, overlap={chunk_overlap})"
    )

    return text_chunks


# ---------- EMBEDDINGS ----------

def get_embedding_model():
    """
    Load the HuggingFace embedding model.
    """
    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"},          # change to "cuda" if you have GPU
        encode_kwargs={"normalize_embeddings": True},
    )
    return embedding_model


# ---------- FAISS INDEX HELPERS ----------

def build_faiss_index(chunks, embedding_model, save_path: str = DB_FAISS_PATH):
    """
    Build a FAISS index from document chunks and save it locally.
    """
    db = FAISS.from_documents(chunks, embedding_model)
    db.save_local(save_path)
    print(f"FAISS index saved to {save_path}")
    return db


def load_faiss_index(embedding_model, save_path: str = DB_FAISS_PATH):
    """
    Load an existing FAISS index from local storage.
    """
    db = FAISS.load_local(
        save_path,
        embedding_model,
        allow_dangerous_deserialization=True,
    )
    print(f"FAISS index loaded from {save_path}")
    return db
