# ArchiBot: Research Assistant Chatbot

## Overview

ArchiBot is an intelligent chatbot designed to assist users in querying research papers and exploring complex concepts. Built using advanced NLP techniques, ArchiBot leverages large language models (LLMs) to generate accurate and concise answers to user queries based on the content of research papers. It also includes a summarization feature for better understanding.

## Features

- Paper Querying: Upload or provide the URL of a research paper in HTML format and ask questions about its content.

- Summarization: Automatically summarizes relevant sections of the paper for easier understanding.

- Contextual Answers: Generates answers based on retrieved content from the document.

- Streamlit Interface: User-friendly web interface for seamless interaction.

## Tech Stack

- Python: Core programming language.

- Streamlit: Web framework for creating the user interface.

- LangChain: Framework for building applications powered by LLMs.

- FAISS: Vector search library for similarity-based retrieval.

- Transformers: Used for text summarization with models like BART.

- GROQ LLM API: Provides access to the language model for generating answers.

## Installation

### Prerequisites

- Python 3.8+

- Access to GROQ API (Get your API key from GROQ).

<!--start code-->
### Install dependencies:

    pip install -r requirements.txt

<!--end code-->

- Set the GROQ API key

<!--start code-->

### Run the application:

    streamlit run app.py

<!--end code-->

## Usage

1. Provide a Research Paper:

Enter the URL of a research paper in the sidebar input box. The paper must be in HTML format.

2. Ask Questions:

Use the text input box to ask specific questions about the paper’s content.

3. View Context and Answer:

The chatbot will retrieve and summarize relevant sections from the paper and display the answer.

## Workflow

1. Launch the application using Streamlit.

2. Enter the URL of a research paper (e.g., from arXiv).

3. Ask a question like:

    - "What is the main contribution of this paper?"

    - "Explain the methodology used in this research."

4. View the retrieved context, summary, and generated answer.

## File Structure

- app.py                 # Streamlit interface file
- main.py                # Core logic for document processing and chatbot functionality
- requirements.txt       # Python dependencies
- README.md              # Documentation
