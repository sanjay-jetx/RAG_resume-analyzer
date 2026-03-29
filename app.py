import streamlit as st
from dotenv import load_dotenv

from rag.loader import load_pdfs
from rag.splitter import split_documents
from rag.vectorstore import create_vector_store
from rag.qa_chain import get_qa_chain

# Load API key
load_dotenv()

# UI
st.set_page_config(page_title="AI Resume Analyzer")
st.title("🚀 AI Resume Analyzer")
st.caption("Upload resumes and analyze candidates using AI")

# Upload PDFs
uploaded_files = st.file_uploader(
    "Upload Resume PDFs",
    type="pdf",
    accept_multiple_files=True
)

# Optional name filter
name = st.text_input("Enter candidate name (optional)")

if uploaded_files:
    with st.spinner("Processing resumes..."):
        docs = load_pdfs(uploaded_files)
        chunks = split_documents(docs)
        vector_db = create_vector_store(chunks)
        qa_chain = get_qa_chain(vector_db)

    st.success("✅ Resumes processed!")

    # Query input
    query = st.text_input("Ask about candidates")

    # 🔥 Ranking button
    if st.button("Find Best Candidate for AI Role"):
        result = qa_chain.invoke("Who is the best candidate for an AI role?")
        st.subheader("🏆 Best Candidate")
        st.write(result.content)

    # Normal query
    if query:
        # Apply name filter if provided
        if name:
            query = f"{query} for candidate {name}"

        result = qa_chain.invoke(query)

        st.subheader("💡 Answer")
        st.write(result.content)

        # Optional context display
        with st.expander("📚 Retrieved Context"):
            st.write("Relevant resume chunks were used to generate this answer.")