# unstructured_query_engine.py

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
# UPDATED: Import a different chain type that supports returning sources
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
import os
import tempfile

# --- Get API Key ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your environment variables.")
    
def process_uploaded_files(uploaded_files):
    """
    Processes a list of uploaded PDF files, extracts text, and splits it into chunks.
    """
    all_chunks = []
    for uploaded_file in uploaded_files:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmpfile:
            tmpfile.write(uploaded_file.getvalue())
            tmp_file_path = tmpfile.name

        loader = PyPDFLoader(tmp_file_path)
        documents = loader.load()
        
        # Add the source filename to each document's metadata
        for doc in documents:
            doc.metadata["source"] = uploaded_file.name

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        chunks = text_splitter.split_documents(documents)
        all_chunks.extend(chunks)
        
        os.remove(tmp_file_path)
        
    return all_chunks

def create_vector_store(text_chunks):
    """
    Creates a FAISS vector store from text chunks.
    """
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
    vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
    return vector_store

def get_qa_chain(vector_store):
    """
    Creates and returns a RetrievalQA chain with a custom prompt and source returning.
    """
    prompt_template = """
    You are an expert data-mining assistant. Your task is to provide detailed and accurate answers based on the provided context from a set of documents.

    Context:
    {context}

    Question:
    {question}

    Instructions:
    1.  Thoroughly analyze the provided context to find the most relevant information.
    2.  Synthesize the information into a coherent and comprehensive answer.
    3.  If the context does not contain the answer, state clearly "The provided documents do not contain information on this topic."
    4.  Do not make up information. Your answers must be grounded in the context.

    Answer:
    """
    
    model = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)
    
    prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
    
    # Use RetrievalQA chain which is designed to work with a retriever (our vector store)
    # and can return source documents.
    chain = RetrievalQA.from_chain_type(
        llm=model,
        chain_type="stuff",
        retriever=vector_store.as_retriever(search_kwargs={"k": 5}),
        chain_type_kwargs={"prompt": prompt},
        return_source_documents=True # This is the crucial part!
    )
    
    return chain

def user_input(user_question):
    """
    Handles user input, runs the QA chain, and displays the results with sources.
    """
    if 'qa_chain' not in st.session_state:
        st.error("QA Chain not initialized. Please process documents first.")
        return

    # Run the QA chain
    response = st.session_state.qa_chain({"query": user_question})
    
    st.write("### Answer")
    st.write(response["result"])

    # Display the source documents in an expander
    with st.expander("View Sources"):
        st.write("The following document chunks were used to generate the answer:")
        for doc in response["source_documents"]:
            st.markdown(f"**Source:** `{doc.metadata.get('source', 'Unknown')}` (Page {doc.metadata.get('page', 'N/A')})")
            st.info(doc.page_content)


def data_mining_engine_section():
    """
    Main Streamlit interface for the data-mining engine.
    """
    st.header("Data-Mining Engine")
    st.write("Upload your unstructured documents (PDFs) and ask questions to find insights.")

    with st.sidebar:
        st.subheader("Document Corpus")
        uploaded_files = st.file_uploader(
            "Upload PDF documents", 
            type="pdf", 
            accept_multiple_files=True
        )

        if st.button("Process Documents"):
            if uploaded_files:
                if not GOOGLE_API_KEY:
                    st.error("Cannot process documents. GOOGLE_API_KEY is not configured.")
                    return

                with st.spinner("Processing documents... This may take a moment."):
                    try:
                        text_chunks = process_uploaded_files(uploaded_files)
                        vector_store = create_vector_store(text_chunks)
                        
                        # Create and store the QA chain in the session state
                        st.session_state.qa_chain = get_qa_chain(vector_store)
                        st.success("Documents processed successfully! You can now ask questions.")
                    except Exception as e:
                        st.error(f"An error occurred during processing: {e}")
            else:
                st.warning("Please upload at least one PDF file.")

    st.subheader("Ask a Question")
    
    if "qa_chain" in st.session_state:
        user_question = st.text_input("What do you want to know from your documents?")
        
        if user_question:
            if not GOOGLE_API_KEY:
                st.error("Cannot ask questions. GOOGLE_API_KEY is not configured.")
                return
            with st.spinner("Searching for answers..."):
                user_input(user_question)
    else:
        st.info("Please upload and process documents to activate the query engine.")