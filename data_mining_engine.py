# unstructured_query_engine.py

import streamlit as st
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain.prompts import PromptTemplate
# NEW IMPORTS for Multi-Query Retriever
from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_core.output_parsers import StrOutputParser
import os
import tempfile
from dotenv import load_dotenv
import asyncio
import nest_asyncio

# Apply nest_asyncio to handle nested event loops
nest_asyncio.apply()

# Load the environment variables right at the start of this module
load_dotenv()

# --- Get API Key ---
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in your .env file.")
    
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
    if not text_chunks:
        return None
    
    try:
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=GOOGLE_API_KEY)
        vector_store = FAISS.from_documents(text_chunks, embedding=embeddings)
        return vector_store
    except Exception as e:
        st.error(f"Error creating vector store: {e}")
        return None

def get_qa_chain(vector_store):
    """
    Creates and returns a RetrievalQA chain with an intelligent Multi-Query retriever.
    """
    if vector_store is None:
        st.error("Vector store is not available. Please process documents first.")
        return None
        
    try:
        prompt_template = """
        You are an expert data-mining assistant. Your task is to provide detailed and accurate answers based on the provided context from a set of documents.

        Context:
        {context}

        Question:
        {question}

        Instructions:
        1.  Thoroughly analyze the provided context to find the most relevant information.
        2.  Synthesize the information from all relevant sources into a single, coherent, and comprehensive answer.
        3.  If the context does not contain the answer, state clearly "The provided documents do not contain information on this topic."
        4.  Do not make up information. Your answers must be grounded in the context.

        Answer:
        """
        
        llm = ChatGoogleGenerativeAI(model="gemini-2.0-flash", temperature=0.3, google_api_key=GOOGLE_API_KEY)
        
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        
        # --- UPGRADED RETRIEVER ---
        # 1. Define the prompt for generating multiple queries
        query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an AI language model assistant. Your task is to generate five 
            different versions of the given user question to retrieve relevant documents from a vector 
            database. By generating multiple perspectives on the user question, your goal is to help
            the user overcome some of the limitations of distance-based similarity search. 
            Provide these alternative questions separated by newlines.
            Original question: {question}""",
        )

        # 2. Create the Multi-Query Retriever
        retriever = MultiQueryRetriever.from_llm(
            retriever=vector_store.as_retriever(), 
            llm=llm,
            prompt=query_prompt
        )
        
        chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever, # Use the new, more intelligent retriever
            chain_type_kwargs={"prompt": prompt},
            return_source_documents=True
        )
        
        return chain
    except Exception as e:
        st.error(f"Error creating QA chain: {e}")
        return None

def user_input(user_question):
    """
    Handles user input, runs the QA chain, and displays the results with sources.
    """
    if 'qa_chain' not in st.session_state or st.session_state.qa_chain is None:
        st.error("The Question-Answering system is not initialized. Please process your documents first.")
        return

    try:
        response = st.session_state.qa_chain({"query": user_question})
        
        st.write("### Answer")
        st.write(response["result"])

        with st.expander("View Sources"):
            st.write("The following document chunks were used to generate the answer:")
            for doc in response["source_documents"]:
                st.markdown(f"**Source:** `{doc.metadata.get('source', 'Unknown')}` (Page {doc.metadata.get('page', 'N/A')})")
                st.info(doc.page_content)
    except Exception as e:
        st.error(f"Error processing your question: {e}")
        st.info("This might be due to an async event loop issue. Please try again.")


def data_mining_engine_section():
    """
    Main Streamlit interface for the data-mining engine with improved file management.
    """
    st.header("Data-Mining Engine")
    st.write("Upload your unstructured documents (PDFs) and ask questions to find insights.")

    # Initialize session state for file management
    if "processed_files" not in st.session_state:
        st.session_state.processed_files = []
    if "all_chunks" not in st.session_state:
        st.session_state.all_chunks = []

    with st.sidebar:
        st.subheader("Document Corpus Management")
        uploaded_files = st.file_uploader(
            "Upload PDF documents to add to the corpus", 
            type="pdf", 
            accept_multiple_files=True
        )

        if st.button("Process New Documents"):
            if uploaded_files:
                if not GOOGLE_API_KEY:
                    st.error("Cannot process documents. GOOGLE_API_KEY is not configured.")
                    return

                new_files_to_process = [
                    file for file in uploaded_files if file.name not in st.session_state.processed_files
                ]

                if not new_files_to_process:
                    st.warning("All uploaded files have already been processed.")
                else:
                    with st.spinner(f"Processing {len(new_files_to_process)} new document(s)..."):
                        try:
                            new_chunks = process_uploaded_files(new_files_to_process)
                            st.session_state.all_chunks.extend(new_chunks)
                            
                            vector_store = create_vector_store(st.session_state.all_chunks)
                            if vector_store is None:
                                st.error("Failed to create vector store. Please check your API key and try again.")
                                return
                                
                            st.session_state.qa_chain = get_qa_chain(vector_store)
                            if st.session_state.qa_chain is None:
                                st.error("Failed to create QA chain. Please try again.")
                                return
                            
                            for file in new_files_to_process:
                                st.session_state.processed_files.append(file.name)
                            
                            st.success("New documents processed and added to the corpus!")
                        except Exception as e:
                            st.error(f"An error occurred: {e}")
                            st.info("This might be due to an async event loop issue. Please try again.")
            else:
                st.warning("Please upload at least one new PDF file to process.")

        if st.session_state.processed_files:
            st.markdown("---")
            st.write("**Active Documents in Corpus:**")
            for filename in st.session_state.processed_files:
                st.text(f"- {filename}")
            
            if st.button("Clear Corpus and History"):
                st.session_state.processed_files = []
                st.session_state.all_chunks = []
                if "qa_chain" in st.session_state:
                    del st.session_state.qa_chain
                st.rerun()

    st.subheader("Ask a Question")
    
    if "qa_chain" in st.session_state and st.session_state.qa_chain is not None:
        user_question = st.text_input("What do you want to know from your documents?")
        
        if user_question:
            if not GOOGLE_API_KEY:
                st.error("Cannot ask questions. GOOGLE_API_KEY is not configured.")
                return
            with st.spinner("Generating sub-queries and searching for answers..."):
                user_input(user_question)
    else:
        st.info("Please upload and process documents to activate the query engine.")
