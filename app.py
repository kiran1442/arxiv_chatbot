import streamlit as st
from main import *
from langchain_text_splitters import RecursiveCharacterTextSplitter
st.title("ArchiBot: Your Research Assistant")
st.write("Query research papers and explore complex concepts with explanations and summaries.")

# Sidebar for Upload Options
st.sidebar.title("Upload Options")
# file_uploaded = st.sidebar.file_uploader("Upload a research paper (TXT format):", type=["txt"])
url_input = st.sidebar.text_input("Enter a paper URL (HTML format):")

# Document Loading and Processing
if url_input:
    # if file_uploaded:
    #     st.sidebar.write("Processing uploaded document...")
    #     arxiv_docs = load_single_arxiv_paper(file_uploaded)
    if url_input:
        st.sidebar.write("Processing document from URL...")
        arxiv_docs = load_web_documents_from_url(url_input)
    else:
        st.write("Please provide a valid input.")

    # Split Document into Chunks
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    all_splits = text_splitter.split_documents(arxiv_docs)

    # Index Chunks
    _ = vector_store.add_documents(documents=all_splits)

    # Question Input
    question = st.text_input("Ask a question about the document:")

    if question:
        state = State(question=question, context=[], answer="", history=[])

        with st.spinner("Retrieving relevant context..."):
            retrieval_result = retrieve(state)
            state["context"] = retrieval_result["context"]
            state["summary"] = [summarize_context(doc.page_content) for doc in state["context"]]
        
        st.write("### Retrieved Context")
        if state["context"]:
            # Combine all context into a single paragraph
            combined_context = "\n\n".join(doc.page_content for doc in state["context"])
            st.write(combined_context)
        else:
            st.write("No relevant context found.")

        with st.spinner("Generating answer..."):
            if state["context"]:
                generation_result = generate(state)
                state["answer"] = generation_result["answer"]

        st.write("### Answer")
        if state["answer"]:
            st.write(state["answer"])
            # visualize_terms(state["context"])
        else:
            st.write("No answer could be generated.")

