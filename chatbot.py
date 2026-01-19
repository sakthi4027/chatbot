.txt streamlit
langchain
langchain-google-genai
langchain-huggingface
langchain-chroma
langchain-community
pypdf
chromadb

GEMINI_API_KEY = "AIzaSyBSUzDHGF4bMWbF_OH1VEF_utOphDxg-mg"

# -*- coding: utf-8 -*-
import streamlit as st
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
import os
import tempfile

# Page config - MUST be first Streamlit command
st.set_page_config(
    page_title="Chatbot Macha",
    page_icon=" hi",
    layout="wide",
    initial_sidebar_state="auto"
)

st.title("Welcome to My Chatbot ")

# Get API key from Streamlit Secrets (do NOT hardcode!)
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # or "GOOGLE_API_KEY" if that's what you use

if not GEMINI_API_KEY:
    st.error("Gemini API key missing! Add it in Streamlit Cloud → Manage app → Secrets.")
    st.stop()

# LLM and embeddings
llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=GEMINI_API_KEY, temperature=0.3)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# Temp dir for PDFs (Cloud safe)
DB_DIR = "chroma_db"  # relative path
os.makedirs(DB_DIR, exist_ok=True)

# Sidebar or main area for upload
st.markdown("**Upload your PDF resume → Create Database → Ask questions!**")

uploaded_files = st.file_uploader("Upload PDF(s)", type=["pdf"], accept_multiple_files=True)

if uploaded_files:
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name

        # Process file
        docs = PyPDFLoader(tmp_path).load()
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(docs)
        
        # Create / update DB
        Chroma.from_documents(
            chunks,
            embeddings,
            collection_name="chatbot_resume",
            persist_directory=DB_DIR
        )
        os.unlink(tmp_path)  # clean up
    st.success(f"Database updated with {len(uploaded_files)} file(s)! Now ask questions.")

# Query section
query = st.text_input("Ask about Sakthivel (e.g., phone number, experience...)", "")

if query:
    with st.spinner("Thinking..."):
        try:
            db = Chroma(
                collection_name="chatbot_resume",
                embedding_function=embeddings,
                persist_directory=DB_DIR
            )
            retriever = db.as_retriever(search_kwargs={"k": 4})

            prompt = ChatPromptTemplate.from_template(
                "Answer only from the provided resume context. If not found, say 'Not in resume'.\n"
                "Context: {context}\n"
                "Question: {input}"
            )

            chain = (
                {"context": retriever, "input": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
            )

            answer = chain.invoke(query)
            st.success(answer)
            if "not in resume" not in answer.lower():
                st.balloons()
        except Exception as e:
            st.error(f"Error: {str(e)}\n(Create database first if empty, or check API key.)")
