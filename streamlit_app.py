import streamlit as st
from PyPDF2 import PdfReader
import docx
import pandas as pd
import base64
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()

# === Fonctions pour récupérer le texte ===
def get_pdf_text(pdf_docs):
    text = ""
    for pdf in pdf_docs:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_docx_text(docx_docs):
    text = ""
    for doc in docx_docs:
        doc_reader = docx.Document(doc)
        for para in doc_reader.paragraphs:
            text += para.text + "\n"
    return text

def get_txt_text(txt_docs):
    text = ""
    for txt in txt_docs:
        text += txt.read().decode("utf-8") + "\n"
    return text

# === Splitter selon le modèle ===
def get_text_chunks(text, model_name):
    if model_name == "OpenAI":
        splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len)
    elif model_name == "Gemini":
        splitter = RecursiveCharacterTextSplitter(chunk_size=10000, chunk_overlap=1000)
    return splitter.split_text(text)

# === Vector Store ===
def get_vector_store(text_chunks, model_name, api_key):
    if model_name == "OpenAI":
        embeddings = OpenAIEmbeddings(api_key=api_key)
    elif model_name == "Gemini":
        embeddings = GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key)
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_index")
    return vector_store

# === Chaîne conversationnelle ===
def get_conversational_chain(model_name, vectorstore=None, api_key=None):
    if model_name == "OpenAI":
        llm = ChatOpenAI(model_name="gpt-4", api_key=api_key)
        memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
        return ConversationalRetrievalChain.from_llm(llm=llm, retriever=vectorstore.as_retriever(), memory=memory)
    elif model_name == "Gemini":
        prompt_template = """
        Answer the question as detailed as possible from the provided context. 
        If the answer is not in the context, say "answer is not available in the context".
        Context:\n {context}\n
        Question: \n{question}\n
        Answer:
        """
        model = ChatGoogleGenerativeAI(model="gemini-flash-0.2", temperature=0.3, google_api_key=api_key)
        prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])
        return load_qa_chain(model, chain_type="stuff", prompt=prompt)

# === Fonction principale de traitement ===
def user_input(user_question, model_name, api_key, pdf_docs, docx_docs, txt_docs, conversation_history):
    if api_key is None or (not pdf_docs and not docx_docs and not txt_docs):
        st.warning("Upload files and provide API key before processing.")
        return

    # Récupérer le texte
    text = ""
    if pdf_docs:
        text += get_pdf_text(pdf_docs)
    if docx_docs:
        text += get_docx_text(docx_docs)
    if txt_docs:
        text += get_txt_text(txt_docs)

    text_chunks = get_text_chunks(text, model_name)
    vector_store = get_vector_store(text_chunks, model_name, api_key)

    user_question_output = user_question
    response_output = ""

    if model_name == "OpenAI":
        chain = get_conversational_chain("OpenAI", vectorstore=vector_store, api_key=api_key)
        response = chain({"question": user_question})
        response_output = response['chat_history'][-1].content
    elif model_name == "Gemini":
        new_db = FAISS.load_local("faiss_index", GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=api_key), allow_dangerous_deserialization=True)
        docs = new_db.similarity_search(user_question)
        chain = get_conversational_chain("Gemini", vectorstore=new_db, api_key=api_key)
        response = chain({"input_documents": docs, "question": user_question}, return_only_outputs=True)
        response_output = response['output_text']

    # Sauvegarder l'historique
    conversation_history.append((user_question_output, response_output, model_name, datetime.now().strftime('%Y-%m-%d %H:%M:%S')))

    # Affichage dans l'interface
    st.markdown(f"""
    <div style='padding:10px; margin-bottom:10px; background-color:#2b313e; color:white; border-radius:8px;'>
    <b>User:</b> {user_question_output}
    </div>
    <div style='padding:10px; margin-bottom:10px; background-color:#475063; color:white; border-radius:8px;'>
    <b>Bot ({model_name}):</b> {response_output}
    </div>
    """, unsafe_allow_html=True)

    # Export CSV
    if conversation_history:
        df = pd.DataFrame(conversation_history, columns=["Question","Answer","Model","Timestamp"])
        csv = df.to_csv(index=False)
        b64 = base64.b64encode(csv.encode()).decode()
        href = f'<a href="data:file/csv;base64,{b64}" download="conversation_history.csv"><button>Download CSV</button></a>'
        st.sidebar.markdown(href, unsafe_allow_html=True)

# === Interface principale ===
def main():
    st.set_page_config(page_title="Chat with multiple files", page_icon=":books:")
    st.header("Chat with multiple files (PDF, DOCX, TXT)")

    if 'conversation_history' not in st.session_state:
        st.session_state.conversation_history = []

    # Sidebar : choix du modèle et clé API
    model_name = st.sidebar.radio("Select Model:", ("OpenAI", "Gemini"))
    api_key = None
    if model_name == "OpenAI":
        api_key = st.sidebar.text_input("OpenAI API Key:", type="password")
    elif model_name == "Gemini":
        api_key = st.sidebar.text_input("Gemini API Key:", type="password")

    # Upload files
    pdf_docs = st.file_uploader("Upload PDF Files", accept_multiple_files=True, type=["pdf"])
    docx_docs = st.file_uploader("Upload DOCX Files", accept_multiple_files=True, type=["docx"])
    txt_docs = st.file_uploader("Upload TXT Files", accept_multiple_files=True, type=["txt"])

    user_question = st.text_input("Ask a question:")

    if user_question:
        user_input(user_question, model_name, api_key, pdf_docs, docx_docs, txt_docs, st.session_state.conversation_history)

if __name__ == "__main__":
    main()
