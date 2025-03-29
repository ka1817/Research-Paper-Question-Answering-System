import streamlit as st
from langchain_groq import ChatGroq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv
import os
import tempfile
from concurrent.futures import ThreadPoolExecutor

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

# Initialize LLM
llm = ChatGroq(api_key=GROQ_API_KEY, model='gemma2-9b-it')

# Sidebar for file uploading
st.sidebar.title("Upload Research Papers 📚")
uploaded_files = st.sidebar.file_uploader("Upload your PDF documents", type="pdf", accept_multiple_files=True)

# Optimized document processing function
def process_uploaded_files(uploaded_files):
    documents = []
    with st.spinner("Processing the uploaded papers... 🧐"):
        # Use ThreadPoolExecutor for concurrent processing
        with ThreadPoolExecutor() as executor:
            futures = []
            for uploaded_file in uploaded_files:
                futures.append(executor.submit(process_single_file, uploaded_file))

            # Collect results
            for future in futures:
                documents.extend(future.result())
    return documents

# Process a single file
def process_single_file(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getbuffer())  # Write the file to the temporary location
        tmp_file_path = tmp_file.name  # Get the path of the temporary file

    # Load the file using PyPDFLoader
    loader = PyPDFLoader(tmp_file_path)
    return loader.load()

# Process uploaded files and display them
if uploaded_files:
    documents = process_uploaded_files(uploaded_files)

    # Split documents into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
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
st.write("Upload your research papers in the sidebar and ask questions about them below! 💬")

# Chat Interface (Show previous questions and answers)
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    st.write(f"**{message['user']}:** {message['content']}")

# User Input for Question
user_question = st.text_input("Ask a question about your research papers 🧐")

# Button to submit the question
if st.button("Ask Question"):
    if user_question:  # If a question is entered
        with st.spinner("Thinking... 💭"):
            # Append the user message to chat history
            st.session_state.messages.append({"user": "You", "content": user_question})

            # Get the response from the chain
            response = chain.invoke({"question": user_question})
            answer = response['answer']

            # Append the bot response to chat history
            st.session_state.messages.append({"user": "Bot", "content": answer})

            # Display the bot's answer
            st.write(f"**Bot:** {answer} 📝")
    else:
        st.warning("Please enter a question before submitting.")

# Optional: Add a button to clear the conversation history
if st.button("Clear Chat History 🔄"):
    st.session_state.messages = []
    st.write("Chat history cleared! Feel free to ask a new question. ✨")
