import streamlit as st
import requests
from io import BytesIO
from dotenv import load_dotenv
import os

load_dotenv()
FASTAPI_URL = os.getenv('FASTAPI_URL', "http://backend:3000")

st.title("Research Paper Question Answering System 📝💭")

st.markdown("""
    Welcome to the **Research Paper Question Answering** system! 
    Upload a research paper in PDF format and ask any questions related to the paper. 
    I will try my best to provide you with the most accurate answer based on the content of the paper. 
    📚✨
""")

with st.sidebar:
    st.header("Upload PDF 📂")
    
    uploaded_file = st.file_uploader("Upload a PDF file 💑", type="pdf")

if uploaded_file:
    st.sidebar.success("File uploaded successfully! ✅")

query = st.text_input("Ask a question about the paper 🤔")

if st.button("Get Answer 🤨"):
    if uploaded_file and query:
        
        with st.spinner("Thinking... 🤯"):
            
            file_bytes = uploaded_file.read()
            
            
            files = {'file': ('uploaded_file.pdf', BytesIO(file_bytes), 'application/pdf')}
            params = {'query': query}

            try:
             
                response = requests.post(f"{FASTAPI_URL}/query_pdf/", files=files, params=params)

                if response.status_code == 200:
                    result = response.json()
                    st.subheader("Answer 📝:")
                    st.write(result["answer"])
                else:
                    st.error("Sorry, there was an issue processing the PDF. Please try again later.")
            except Exception as e:
                st.error(f"An error occurred: {e}")
    else:
        st.error("Please upload a PDF file and enter a question before submitting.")
