import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
# Disable LangChain tracing
os.environ["LANGCHAIN_TRACING_V2"] = "false"
os.environ["LANGCHAIN_API_KEY"] = ""
os.environ["LANGCHAIN_PROJECT"] = ""


import streamlit as st
from dotenv import load_dotenv
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA

from langchain.chat_models import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from config import Settings

load_dotenv()

st.title("🧾 1231 Chat with Your PDF")

# === Config Inputs ===
provider = st.selectbox("Choose LLM Provider", ["OpenAI", "Groq"])
# model_name = st.selectbox(
#     "Choose Model",
#     ["gpt-3.5-turbo", "gpt-4"] if provider == "OpenAI" else ["mixtral-8x7b-32768", "llama3-70b-8192"]
# )

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")

# --- Display chat history ---
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# --- User input ---
user_input = st.chat_input("Ask a question about the PDF")
# query = st.text_input("Ask a question:")

# === PDF Loader ===
def load_pdf(file):
    reader = PdfReader(file)
    return ''.join([page.extract_text() for page in reader.pages])

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
    return splitter.split_text(text)

def get_llm(provider=None):
    if provider == "OpenAI":
        return ChatOpenAI(model=Settings.OPENAI_MODEL, temperature=0)
    elif provider == "Groq":
        return ChatGroq(model_name="llama3-70b-8192", temperature=0)
    else:
        return ChatGroq(model_name="llama3-70b-8192", temperature=0)

# === Main ===
if uploaded_file:
    text = load_pdf(uploaded_file)
    chunks = chunk_text(text)
    
    # embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": "cpu"}  # or "cuda" if using GPU
    )

    vectordb = Chroma.from_texts(chunks, embedding=embeddings)

    llm = get_llm()
    print("llm  >>>>",llm)
    # qa = RetrievalQA.from_chain_type(
    #     llm=llm,
    #     retriever=vectordb.as_retriever(),
    #     return_source_documents=True
    # )
    
    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vectordb.as_retriever(),
        return_source_documents=False
    )
    
    if user_input:
        # Display user message
        st.chat_message("user").markdown(user_input)
        st.session_state.messages.append({"role": "user", "content": user_input})

        # Get answer
        response = qa.invoke({"query": user_input})
        answer = response["result"]

        # Display assistant response
        with st.chat_message("assistant"):
            st.markdown(answer)

        st.session_state.messages.append({"role": "assistant", "content": answer})

    # if query:
    #     result = qa.run(query)
    #     st.success(result)

if st.button("🧹 Clear Chat"):
    st.session_state.messages = []
