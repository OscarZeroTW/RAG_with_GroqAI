import streamlit as st
import os
from langchain_groq import ChatGroq
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import OllamaEmbeddings
from dotenv import load_dotenv
import time

load_dotenv()

## load the Groq API key
groq_api_key = os.environ['GROQ_API_KEY']

## ollama base url
ollama_base_url = os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434")

st.title("Llama3 With Groq")

llm = ChatGroq(groq_api_key=groq_api_key,
               model_name="llama3-8b-8192")

prompt = ChatPromptTemplate.from_template(
    """
    Answer the questions based on the provided context only.
    Please provide the most accurate respone based on the question
    <context>
    {context}
    <context>
    questions:{input}
    """
)


def vector_embedding():
    if "vectors" not in st.session_state:
        st.session_state.embeddings = OllamaEmbeddings(model="nomic-embed-text", base_url=ollama_base_url)
        
        # Get all PDF file paths from the docs directory
        pdf_files = [os.path.join("./docs", f) for f in os.listdir("./docs") if f.endswith('.pdf')]
        
        if not pdf_files:
            st.warning("No PDF files found in the 'docs' directory.")
            return

        # Load all PDF documents
        all_docs = []
        for pdf_file in pdf_files:
            loader = PyPDFLoader(pdf_file)
            all_docs.extend(loader.load())

        st.session_state.docs = all_docs
        st.session_state.text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000,
                                                                        chunk_overlap=200)  # Chunk Creation
        st.session_state.final_documents = st.session_state.text_splitter.split_documents(
            st.session_state.docs[:50])  # Splitting
        st.session_state.vectors = FAISS.from_documents(st.session_state.final_documents,
                                                        st.session_state.embeddings)  # vector Ollama embeddings


prompt1 = st.text_input("Enter Your Question From Doduments")

if st.button("Documents Embedding"):
    vector_embedding()
    st.write("Vector Store DB Is Ready")

if prompt1:
    document_chain = create_stuff_documents_chain(llm, prompt)
    retriever = st.session_state.vectors.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, document_chain)
    start = time.process_time()
    response = retrieval_chain.invoke({"input": prompt1})
    print("Response time :", time.process_time() - start)
    st.write(response['answer'])

    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant chunks
        for i, doc in enumerate(response["context"]):
            st.write(doc.page_content)
            st.write("--------------------------------")
