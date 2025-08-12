import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="PDF Summarizer & Chat", layout="wide")

st.title("📄 PDF Summarizer & Chatbot")

# Upload PDF
uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    # Read PDF
    pdf_reader = PdfReader(uploaded_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()

    # Chunk text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # Conversational Chain
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.5)
    chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever()
    )

    # Chat state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Summarize button
    if st.button("📌 Summarize PDF"):
        summary_prompt = "Summarize this PDF in detail:\n" + text[:10000]  # First 10k chars
        summary = llm.invoke(summary_prompt)
        st.subheader("📜 Summary")
        st.write(summary.content)

    # Chat interface
    st.subheader("💬 Chat with PDF")
    user_question = st.text_input("Ask something about the PDF:")

    if user_question:
        response = chain.invoke({"question": user_question, "chat_history": st.session_state.chat_history})
        st.session_state.chat_history.append((user_question, response["answer"]))
        st.write("🤖", response["answer"])
