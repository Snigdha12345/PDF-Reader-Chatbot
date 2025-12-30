import os
import re
import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
from PyPDF2 import PdfReader

# LangChain
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


# ------------------ LOAD ENV ------------------
load_dotenv()
API_KEY = os.getenv("GOOGLE_API_KEY")

if not API_KEY:
    raise RuntimeError("GOOGLE_API_KEY is missing in .env")

genai.configure(api_key=API_KEY)


# ------------------ FREE MODELS ------------------
FREE_MODELS = [
    "models/gemini-2.0-flash-lite",
    "models/gemini-2.0-flash-lite-001",
    "models/gemini-2.0-flash-lite-preview-02-05",
    "models/gemini-1.5-flash-lite",
    "models/gemini-flash-latest",
]


@st.cache_resource
def load_gemini_model():
    for model_name in FREE_MODELS:
        try:
            model = genai.GenerativeModel(model_name)
            return model_name, model
        except Exception:
            continue
    return None, None


# ------------------ PDF EXTRACTION ------------------
def extract_pdf_pages(files):
    pages = []

    for file in files:
        reader = PdfReader(file)
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append({
                    "text": text.strip(),
                    "source": file.name,
                    "page": i + 1
                })

    return pages


# ------------------ DOCUMENT TYPE DETECTION ------------------
def detect_doc_type(filename):
    name = filename.lower()
    if "statute" in name or "rsmo" in name:
        return "statute"
    if "csr" in name or "regulation" in name:
        return "csr"
    return "manual"


# ------------------ CHUNKING ------------------
def split_pages_into_chunks(pages):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=200,
        separators=["\n\n", "\n", ".", " "]
    )

    documents = []

    for page in pages:
        doc_type = detect_doc_type(page["source"])
        chunks = splitter.split_text(page["text"])

        for chunk in chunks:
            documents.append(
                Document(
                    page_content=chunk,
                    metadata={
                        "source": page["source"],
                        "page": page["page"],
                        "doc_type": doc_type
                    }
                )
            )

    return documents


# ------------------ VECTOR STORE ------------------
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


def create_vector_store(documents):
    return FAISS.from_documents(documents, embedding_model)


# ------------------ QUESTION REWRITING ------------------
def rewrite_question(question):
    q = question.lower()

    if "formal" in q and "bidding" in q:
        return question + " What dollar threshold requires this?"

    if "required" in q and "bidding" in q:
        return question + " What law defines this requirement?"

    return question


# ------------------ SEARCH WITH STATUTE PRIORITY ------------------
DOC_PRIORITY = {
    "statute": 0,
    "csr": 1,
    "manual": 2
}


def search_chunks(query, vector_db, base_k=6):
    rewritten_query = rewrite_question(query)
    raw_results = vector_db.similarity_search_with_score(rewritten_query, k=20)

    # Sort by (document authority, similarity score)
    sorted_results = sorted(
        raw_results,
        key=lambda x: (
            DOC_PRIORITY.get(x[0].metadata.get("doc_type", "manual"), 2),
            x[1]
        )
    )

    return sorted_results[:base_k]


# ------------------ ANSWER WITH CONFIDENCE EXPANSION ------------------
def answer_with_gemini(question, docs_with_scores):
    if not docs_with_scores:
        return "I don't know based on the provided PDFs."

    model_name, chat_model = load_gemini_model()
    if chat_model is None:
        return "Gemini is temporarily unavailable."

    MAX_CONTEXT_CHARS = 6000
    context = ""

    # Confidence-based expansion
    for i, (doc, score) in enumerate(docs_with_scores):
        if len(context) + len(doc.page_content) > MAX_CONTEXT_CHARS:
            break

        context += f"\n\n---\n\n[{doc.metadata['doc_type'].upper()} | {doc.metadata['source']} | Page {doc.metadata['page']}]\n"
        context += doc.page_content

        # Stop early if top result is strong
        if i == 2 and score < 0.2:
            break

    prompt = f"""
Use ONLY the context below to answer the question.
Cite sources inline using [SOURCE | PAGE].
If the answer is not in the context, reply exactly:
"I don't know based on the provided PDFs."

Context:
{context}

Question: {question}

Answer:
"""

    try:
        response = chat_model.generate_content(prompt)
        return response.text.strip()
    except Exception as e:
        return f"Model error: {str(e)}"


# ------------------ STREAMLIT APP ------------------
def main():
    st.set_page_config(page_title="Chat with your PDFs", layout="wide")
    st.title("Chat with your PDFs")

    if "db" not in st.session_state:
        st.session_state.db = None

    if "history" not in st.session_state:
        st.session_state.history = []

    for role, message in st.session_state.history:
        st.chat_message(role).write(message)

    with st.sidebar:
        st.header("Upload PDFs")
        uploaded_files = st.file_uploader(
            "Upload PDF files",
            type=["pdf"],
            accept_multiple_files=True
        )

        if st.button("Process PDFs"):
            if not uploaded_files:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing PDFs..."):
                    pages = extract_pdf_pages(uploaded_files)
                    documents = split_pages_into_chunks(pages)
                    st.session_state.db = create_vector_store(documents)
                st.success("PDFs processed successfully.")

    question = st.text_input("Ask a question about your PDFs:")

    if question:
        if not st.session_state.db:
            st.error("Please upload and process PDFs first.")
            return

        st.session_state.history.append(("user", question))
        st.chat_message("user").write(question)

        with st.spinner("Searching..."):
            results = search_chunks(question, st.session_state.db)

        with st.spinner("Answering..."):
            answer = answer_with_gemini(question, results)

        st.session_state.history.append(("assistant", answer))
        st.chat_message("assistant").write(answer)


if __name__ == "__main__":
    main()
