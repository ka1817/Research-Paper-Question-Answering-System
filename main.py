from fastapi import FastAPI, File, UploadFile
from langchain_groq import ChatGroq
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.retrievers import BM25Retriever
from langchain.vectorstores import FAISS
from langchain.retrievers import BM25Retriever
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.schema import Document
from collections import defaultdict
from sentence_transformers import CrossEncoder
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
import os
from dotenv import load_dotenv
from io import BytesIO
import uvicorn
import tempfile
import string
load_dotenv()
GROQ_API_KEY = os.getenv('GROQ_API_KEY')

app = FastAPI(title="Research Paper Q & A")

llm = ChatGroq(api_key=GROQ_API_KEY, model='llama-3.3-70b-versatile')


reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def reciprocal_rank_fusion(results_list, k=60):
    scores = defaultdict(float)
    for results in results_list:
        for rank, doc in enumerate(results):
            scores[doc.page_content] += 1 / (rank + k)
    
    ranked_docs = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [Document(page_content=doc) for doc in ranked_docs]


def process_pdf(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp_file:
        temp_file.write(pdf_file.read())
        temp_file_path = temp_file.name
    
    loader = PyPDFLoader(temp_file_path)
    documents = loader.load()
    
    splitter = RecursiveCharacterTextSplitter(chunk_size=400, chunk_overlap=50)
    chunks = splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    
    vectorstore = FAISS.from_documents(chunks, embeddings)
    bm25_retriever = BM25Retriever.from_documents(chunks)
    
    return vectorstore, bm25_retriever, chunks

@app.post("/query_pdf/")
async def query_pdf(file: UploadFile = File(...), query: str = ""):
    pdf_content = await file.read()
    pdf_file = BytesIO(pdf_content)
    
    vectorstore, bm25_retriever, chunks = process_pdf(pdf_file)
    
    bm25_results = bm25_retriever.get_relevant_documents(query,k=5)
    faiss_results = vectorstore.as_retriever().get_relevant_documents(query,k=5)
    
    final_docs = reciprocal_rank_fusion([bm25_results, faiss_results])
    
    
    context_text = " ".join([doc.page_content for doc in final_docs[:3]])

    
    qa_prompt = PromptTemplate(
        input_variables=["context", "question"],
        template="You are an expert in AI. Use the following context to answer the question accurately and concisely. Context: {context} and Question:\n{question}"
    )

    qa_chain = qa_prompt | llm

    result = qa_chain.invoke({"context": context_text, "question": query})

    
    return {"answer": result.content}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=3000)

