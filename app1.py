import streamlit as st
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from concurrent.futures import ThreadPoolExecutor
import os
import tempfile
from dotenv import load_dotenv

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize LLM
llm = ChatGroq(api_key=GROQ_API_KEY, model='gemma2-9b-it')

# Sidebar for file uploading
st.sidebar.title("Upload Research Papers 📚")
uploaded_files = st.sidebar.file_uploader("Upload your PDF documents", type="pdf", accept_multiple_files=True)

# Function to process a single file
def process_single_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())  # Write the file to the temporary location
        tmp_file_path = tmp_file.name  # Get the path of the temporary file

    # Load the file using PyPDFLoader
    loader = PyPDFLoader(tmp_file_path)
    return loader.load()

# Process uploaded files using ThreadPoolExecutor for parallel processing
def process_uploaded_files(uploaded_files):
    documents = []
    with st.spinner("Processing the uploaded papers... 🧐"):
        with ThreadPoolExecutor() as executor:
            futures = [executor.submit(process_single_file, uploaded_file) for uploaded_file in uploaded_files]
            
            # Collect results from the futures
            for future in futures:
                documents.extend(future.result())
    return documents

# Main file processing logic
documents = []
if uploaded_files:
    documents = process_uploaded_files(uploaded_files)
    
    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    
    # Load embeddings and initialize FAISS vector store
    embeddings = HuggingFaceEmbeddings()
    vectord = FAISS.from_documents(chunks, embeddings)
    
    # Memory and retriever setup
    memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)
    hybrid_retriever = vectord.as_retriever()

    # Set up prompt template
    template = """You are an expert in answering questions based on the provided research papers. 
    Use the given context to generate an accurate and concise response.

    Context: {context}
    Chat History: {chat_history}
    Question: {question}

    Answer:
    """
    
    prompt = PromptTemplate(
        input_variables=["chat_history", "context", "question"],
        template=template
    )
    
    # Initialize ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=hybrid_retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

# Main screen UI
st.title("Chat with the Research Bot 🤖")

# Display instructions
st.write("Upload your research papers in the sidebar and ask questions about them below!")

# User Input for Question with Button
user_question = st.text_input("Ask a question about your research papers 🧐")
if user_question:
    if st.button("Get Answer 📝"):
        with st.spinner("Thinking... 💭"):
            response = chain.invoke({"question": user_question})
            st.write(f"**Answer:** {response['answer']} 📝")
else:
    st.write("Please enter a question to get started. 💬")

# Optional: Add a button to clear the conversation history (or reset the state)
if st.button("Clear Chat History 🔄"):
    memory.clear()
    st.write("Chat history cleared! Feel free to ask a new question.")

# Progress bar for document processing
if uploaded_files:
    progress_bar = st.progress(0)
    total_files = len(uploaded_files)
    
    for i, uploaded_file in enumerate(uploaded_files):
        progress_bar.progress((i + 1) / total_files)
