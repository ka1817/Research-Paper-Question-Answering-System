import os
import shutil

from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from dotenv import load_dotenv  
from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.retrievers.ensemble import EnsembleRetriever

from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationSummaryMemory
from langchain.prompts import PromptTemplate
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

if not GROQ_API_KEY:
    raise ValueError("❌ ERROR: GROQ_API_KEY is missing! Please set it in your .env file.")

# Initialize FastAPI app
app = FastAPI()

# Initialize LLM (Groq's Gemma model)
llm = ChatGroq(api_key=GROQ_API_KEY, model='gemma-9b')

# Initialize memory
memory = ConversationSummaryMemory(llm=llm, memory_key="chat_history", return_messages=True)

# Prompt Template
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

# Global variables
vectorstore = None
hybrid_retriever = None


@app.post("/upload/")
async def upload_research_paper(file: UploadFile = File(...)):
    """
    Upload a research paper (PDF), process it, and create a hybrid retriever.
    """
    global vectorstore, hybrid_retriever

    # Validate file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    # Save the uploaded file temporarily
    file_path = f"uploads/{file.filename}"
    os.makedirs("uploads", exist_ok=True)
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    # Load the document
    loader = PyPDFLoader(file_path)
    documents = loader.load()

    if not documents:
        os.remove(file_path)
        raise HTTPException(status_code=400, detail="Failed to process the document.")

    # Split into chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=300, chunk_overlap=100)
    chunks = text_splitter.split_documents(documents)

    # Create FAISS vector store
    embeddings = HuggingFaceEmbeddings()
    vectorstore = FAISS.from_documents(chunks, embeddings)

    # Create BM25 retriever
    bm25_retriever = BM25Retriever.from_documents(chunks)

    # Hybrid Retriever (FAISS + BM25)
    hybrid_retriever = EnsembleRetriever(retrievers=[bm25_retriever, vectorstore.as_retriever()], weights=[0.6, 0.4])

    return {"message": "✅ Research paper uploaded and indexed successfully."}


@app.post("/query/")
async def ask_question(question: str = Form(...)):
    """
    Answer a user's question using the uploaded research paper.
    """
    if hybrid_retriever is None:
        raise HTTPException(status_code=400, detail="No research paper uploaded. Please upload a document first.")

    # Initialize ConversationalRetrievalChain
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=hybrid_retriever,
        memory=memory,
        combine_docs_chain_kwargs={"prompt": prompt}
    )

    # Get the response
    response = chain.invoke({"context": "", "question": question, "chat_history": []})

    return {"answer": response.get("answer", "No answer found.")}


# Run FastAPI server (for local execution)
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000, reload=True)
