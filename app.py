import streamlit as st
import fitz  # PyMuPDF
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain.vectorstores import FAISS
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

st.set_page_config(page_title="📄 Creative PDF Summarizer & Chatbot", layout="wide")
st.title("📄 Creative PDF Summarizer & Chatbot ✨")

uploaded_file = st.file_uploader("Upload your PDF", type=["pdf"])

if uploaded_file:
    # Read PDF
    doc = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()

    # Chunk text
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2500,
        chunk_overlap=300,
        length_function=len
    )
    chunks = text_splitter.split_text(text)

    # Create embeddings
    embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vectorstore = FAISS.from_texts(chunks, embeddings)

    # LLM - use PRO for creativity
    llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash", temperature=0.8)

    # Chat history
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Summarize
    if st.button("📌 Summarize PDF Creatively"):
        summary_prompt = f"""
        You are an expert summarizer with a flair for creativity.
        Summarize the following PDF content in a detailed yet engaging way.
        Use bullet points, examples, and analogies to make it interesting.
        PDF Content:
        {text[:15000]}
        """
        summary = llm.invoke(summary_prompt)
        st.subheader("📜 Creative Summary")
        st.write(summary.content)

    # Chat
    st.subheader("💬 Chat with PDF (Creative Mode)")
    user_question = st.text_input("Ask anything about the PDF:")

    if user_question:
        # Retrieve more context
        docs = vectorstore.similarity_search(user_question, k=6)
        context = "\n\n".join([d.page_content for d in docs])

        prompt = f"""
        You are a creative and knowledgeable assistant.
        Answer the question using the PDF content provided.
        Be detailed, insightful, and creative — use examples, analogies, and explanations
        that make the answer engaging and easy to understand.
        
        Question: {user_question}
        
        Relevant PDF Content:
        {context}
        """
        response = llm.invoke(prompt)
        st.session_state.chat_history.append((user_question, response.content))
        st.write("🤖", response.content)
