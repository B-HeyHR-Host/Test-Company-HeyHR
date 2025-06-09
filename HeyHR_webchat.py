
import os
import streamlit as st
st.set_page_config(page_title="HeyHR", page_icon="✨")
st.markdown("""
    <style>
    body {
        background-color: #F9F9F9;
        font-family: 'Arial', sans-serif;
    }
    .stTextInput > div > div > input {
        background-color: #ffffff;
        color: #333;
    }
    </style>
""", unsafe_allow_html=True)
from langchain.document_loaders import TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA

# Set your OpenAI API key
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

@st.cache_resource
def load_qa():
    # Load and prepare documents
    folder_path = "."
    docs = []

    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            try:
                st.write(f"Loading: {filename}")
                loader = TextLoader(os.path.join(folder_path, filename))
                docs.extend(loader.load())
            except Exception as e:
                st.error(f"❌ Could not load {filename}: {e}")


    # Split into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    split_docs = splitter.split_documents(docs)

    # Turn into vector embeddings
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(split_docs, embeddings)

    # Create the QA chain
    chat_model = ChatOpenAI(model="gpt-4", temperature=0)
    return RetrievalQA.from_chain_type(llm=chat_model, retriever=vector_store.as_retriever())


# Load once and reuse
qa_chain = load_qa()

# Streamlit web interface
st.image("logo.png", width=300)  # Adjust width as needed
st.write("We are here to support with your queries.")

if "history" not in st.session_state:
    st.session_state.history = []

user_input = st.text_input("Your question:", key="input")

if user_input:
    answer = qa_chain.run(user_input)
    st.session_state.history.append((user_input, answer))

# Show chat history
for q, a in reversed(st.session_state.history):
    st.markdown(f"""
    <div style="background-color: #f0f0f5; padding: 10px; border-radius: 8px; margin-bottom: 10px;">
        <strong>You:</strong> {q}
    </div>
    <div style="background-color: #e0f7fa; padding: 10px; border-radius: 8px; margin-bottom: 20px;">
        <strong>HeyHR:</strong> {a}
    </div>
    """, unsafe_allow_html=True)


    import os
import streamlit as st
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
                st.write(f"Loading: {filename}")
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
st.title("Ask HeyHR✨")
st.caption("Your instant HR assistant. Ask anything based on your company handbook, policies, and processes.")

# Capture chat input
query = st.text_input("Type your HR question here:")

if query:
    with st.spinner("Thinking..."):
        response = qa_chain.run(query)
        st.success(response)