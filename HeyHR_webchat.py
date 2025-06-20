import os
import streamlit as st
hide_streamlit_style = """
    <style>
    #MainMenu {visibility: hidden;}
    header {visibility: hidden;}
    footer {visibility: hidden;}

    /* Override bottom bar (Streamlit branding) */
    .block-container > div:nth-child(n) > div:nth-child(n) > div:nth-child(n) > div:nth-child(n) > footer {display: none !important;}

    /* Remove all elements with data-testid footer branding */
    [data-testid="stStatusWidget"] {display: none !important;}
    [data-testid="stDecoration"] {display: none !important;}
    [class*="stDeployButton"] {display: none !important;}
    .st-emotion-cache-z5fcl4 {display: none !important;} /* mobile footer container */
    </style>
"""
st.markdown(hide_streamlit_style, unsafe_allow_html=True)
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Streamlit app layout
st.set_page_config(page_title="HeyHR", page_icon="✨")

# Get OpenAI API key securely from secrets
openai_api_key = st.secrets["OPENAI_API_KEY"]

# Load and process documents
@st.cache_resource
def load_qa():
    folder_path = "."
    docs = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            try:
                loader = TextLoader(os.path.join(folder_path, filename))
                docs.extend(loader.load())
            except Exception as e:
                st.error(f"❌ Could not load {filename}: {e}")

    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)
    vector_store = FAISS.from_documents(split_docs, embeddings)

    chat_model = ChatOpenAI(temperature=0, openai_api_key=openai_api_key)
    return RetrievalQA.from_chain_type(llm=chat_model, retriever=vector_store.as_retriever())

# Load the QA chain
qa_chain = load_qa()

st.markdown("""
<style>
    body { background-color: #F9F9F9; font-family: 'Arial', sans-serif; }
    .stTextInput > div > div > input { background-color: #fff; color: #333; }
</style>
""", unsafe_allow_html=True)

st.image("logo.png", width=300)
st.title("Type your query below✨")
st.caption("This is the HeyHR bot on behalf of TEST CLIENT, you can ask us anything relating to the TEST handbook, policies and processes.")

# Capture chat input
query = st.text_input("Type your HR question here:")

if query:
    with st.spinner("Thinking..."):
        response = qa_chain.run(query)
        st.success(response)