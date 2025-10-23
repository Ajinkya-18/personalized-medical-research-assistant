import streamlit as st
from langchain_community.llms import Ollama
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv
import os


load_dotenv()
os.environ["GOOGLE_API_KEY"] = str(os.getenv("GOOGLE_API_KEY"))


@st.cache_resource
def get_models():
    # llm = Ollama(model="gemma3:1b")
    llm = ChatGoogleGenerativeAI(model='gemini-2.0-flash')

    model_name = "sentence-transformers/all-MiniLM-L6-v2"

    embeddings = HuggingFaceEmbeddings(
        model_name=model_name, 
        model_kwargs = {'device': 'cpu'}
    )
    
    return llm, embeddings


@st.cache_resource
def load_vector_db(_embeddings):
    db = FAISS.load_local("faiss_index", _embeddings, allow_dangerous_deserialization=True)
    return db


def create_rag_chain(llm, retriever):
    prompt_template = """
    You are an expert medical research assistant specializing in Neuroscience. 
    Answer the user's question based *only* on the following context. 
    If the context does not contain the answer, state that you don't know. However
    try to answer the user's query if possible by first staing that the provided text 
    doesn't provide any direct information regarding user's query but you would try to 
    answer the user's query using your own reasoning and intelligence since they 
    asked it.

    Context:
    {context}

    Question:
    {input}
    """

    prompt = ChatPromptTemplate.from_template(prompt_template)

    combine_docs_chain = create_stuff_documents_chain(llm, prompt)

    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

    return retrieval_chain


st.set_page_config(page_title="Medical Research Assistant", layout='wide')
st.title("Specialized Neuroscience Research Assistant")
st.write("Ask me questions about the neuroscience documents you provided.")

llm, embeddings = get_models()
db = load_vector_db(embeddings)

retriever = db.as_retriever(search_kwargs={"K":10})

rag_chain = create_rag_chain(llm, retriever)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if prompt := st.chat_input("What is your question?"):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.spinner("Thinking..."):
        response = rag_chain.invoke({"input": prompt})

        answer = response["answer"]

    with st.chat_message("assistant"):
        st.markdown(answer)

    st.session_state.messages.append({"role":"assistant", "content": answer})



