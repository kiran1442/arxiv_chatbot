import os
import getpass
import streamlit as st
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
import faiss
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain import hub
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_community.document_loaders import WebBaseLoader
from transformers import pipeline
from collections import Counter
import pandas as pd
from typing_extensions import List, TypedDict

# Initialize API Key
os.environ["GROQ_API_KEY"] = getpass.getpass("Enter Your Key : ")

df = pd.read_parquet("hf://datasets/ashish-chouhan/arxiv_cs_papers/data/train-00000-of-00001-bf80d7e563046673.parquet")
print(df['url'])

# LLM and Embedding Setup
llm = ChatGroq(model="gemma2-9b-it")
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

dummy_text = "This is a test."
embedding_vector = embeddings.embed_query(dummy_text)
dimension = len(embedding_vector)

# Initialize FAISS Components
index = faiss.IndexFlatL2(dimension)
vector_store = FAISS(embedding_function=embeddings,
                    index=index,
                    docstore=InMemoryDocstore(),
                    index_to_docstore_id={}
)
# Define Prompt for Question-Answering
prompt = hub.pull("rlm/rag-prompt")

# Summarization Pipeline
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

def summarize_context(context: str) -> str:
    summary = summarizer(context, max_length=100, do_sample=False)
    return summary[0]["summary_text"]

# State Definition
class State(TypedDict):
    question: str
    context: List[Document]
    answer: str
    history: List[str]

# Application Steps
def retrieve(state: State):
    retrieved_docs = vector_store.similarity_search(state["question"])
    return {"context": retrieved_docs}

def generate(state: State):
    docs_content = "\n\n".join(doc.page_content for doc in state["context"])
    messages = prompt.invoke({"question": state["question"], "context": docs_content})
    response = llm.invoke(messages)
    return {"answer": response.content}

# def visualize_terms(context: List[Document]):
#     words = [word for doc in context for word in doc.page_content.split()]
#     freq = Counter(words)
#     df = pd.DataFrame(freq.most_common(20), columns=["Word", "Frequency"])
#     st.bar_chart(df.set_index("Word"))

def load_web_documents_from_url(url: str):
    loader = WebBaseLoader(web_paths=[url])
    #print(loader)
    return loader.load()