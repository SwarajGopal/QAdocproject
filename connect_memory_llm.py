import os

from dotenv import load_dotenv, find_dotenv

from langchain_classic.chains import RetrievalQA
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.memory import ConversationBufferMemory

from create_memory_llm import (
    get_embedding_model,
    load_faiss_index,
    DB_FAISS_PATH,
)

# ---------- ENV & CONFIG ----------

load_dotenv(find_dotenv())

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# Groq models available (all very fast):
# - "llama-3.3-70b-versatile" (best quality, recommended)
# - "llama-3.1-70b-versatile" (excellent quality)
# - "llama-3.1-8b-instant" (fastest)
# - "gemma2-9b-it" (good balance)
# - "mixtral-8x7b-32768" (DEPRECATED - do not use)

GROQ_MODEL = "llama-3.3-70b-versatile"

# ---------- LLM SETUP ----------

def load_llm(model: str = GROQ_MODEL):
    """
    Load a ChatGroq LLM.
    Requires GROQ_API_KEY in environment.
    Get your free API key at: https://console.groq.com
    """
    if not GROQ_API_KEY:
        raise ValueError(
            "GROQ_API_KEY is not set. Please:\n"
            "1. Get a free API key from https://console.groq.com\n"
            "2. Add it to your .env file as: GROQ_API_KEY=your_key_here"
        )

    llm = ChatGroq(
        model=model,
        groq_api_key=GROQ_API_KEY,
        temperature=0.3,
        max_tokens=512,
    )
    return llm


# ---------- PROMPT ----------

CUSTOM_PROMPT_TEMPLATE = """You are a helpful AI assistant. Answer the question based on the context provided below.

If the answer is not in the context, say "I don't have enough information to answer that question based on the provided documents."

Context:
{context}

Question: {question}

Answer:"""


def get_custom_prompt():
    """
    Return a LangChain PromptTemplate for the RetrievalQA chain.
    """
    prompt = PromptTemplate(
        template=CUSTOM_PROMPT_TEMPLATE,
        input_variables=["context", "question"],
    )
    return prompt


# ---------- QA CHAIN WITH MEMORY ----------

def get_qa_chain():
    """
    Create and return a RetrievalQA chain with:
    - Groq LLM (fast and free)
    - FAISS vector store as retriever
    - ConversationBufferMemory
    - Custom prompt
    """
    # 1) Embeddings + FAISS
    print("Loading embeddings and vector store...")
    embedding_model = get_embedding_model()
    db = load_faiss_index(embedding_model, DB_FAISS_PATH)

    # 2) LLM
    print("Initializing Groq LLM...")
    llm = load_llm()

    # 3) Memory
    memory = ConversationBufferMemory(
        memory_key="chat_history",
        input_key="query",
        output_key="result",
        return_messages=True,
    )

    # 4) Prompt
    prompt = get_custom_prompt()

    # 5) RetrievalQA chain
    print("Building QA chain...")
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=db.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True,
        chain_type_kwargs={"prompt": prompt},
        memory=memory,
    )

    return qa_chain


# ---------- OPTIONAL: CLI TEST ----------

if __name__ == "__main__":
    """
    Simple CLI test so you can run:
        python connect_memory_llm.py
    and check that everything works before wiring to Streamlit.
    """
    print("="*60)
    print("  Document QA System with Groq")
    print("="*60)
    print("\nInitializing system...\n")
    
    try:
        qa = get_qa_chain()
        print("\n✓ QA system ready!\n")
    except ValueError as e:
        print(f"\n✗ Configuration Error: {e}\n")
        exit(1)
    except Exception as e:
        print(f"\n✗ Failed to initialize: {e}\n")
        exit(1)

    print("Ask questions about your documents.")
    print("Type 'exit', 'quit', or 'q' to quit.\n")
    print("="*60 + "\n")
    
    while True:
        try:
            user_query = input("You: ").strip()
            
            if user_query.lower() in {"exit", "quit", "q"}:
                print("\nGoodbye! 👋\n")
                break
            
            if not user_query:
                continue

            print("\n🤔 Searching documents and generating answer...\n")
            
            response = qa.invoke({"query": user_query})
            
            print("="*60)
            print("🤖 Assistant:")
            print("-"*60)
            print(response["result"])
            
            # Show source documents
            if response.get("source_documents"):
                print("\n📚 Sources:")
                for i, doc in enumerate(response["source_documents"], 1):
                    source = doc.metadata.get('source', 'Unknown')
                    # Show snippet of content
                    snippet = doc.page_content[:100].replace('\n', ' ')
                    print(f"  [{i}] {source}")
                    print(f"      Preview: {snippet}...")
            
            print("="*60 + "\n")
            
        except KeyboardInterrupt:
            print("\n\nInterrupted. Goodbye! 👋\n")
            break
        except Exception as e:
            print(f"\n❌ Error: {e}")
            print("Please try again with a different question.\n")