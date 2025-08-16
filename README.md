# Llama3 RAG with Groq

This project is a retrieval-augmented generation (RAG) application that uses Streamlit for the user interface, Groq AI for fast inference with the Llama3 model, and LangChain for building the RAG pipeline. The application allows users to ask questions about a PDF document and receive answers based on the document's content.

## Features

- **Fast Inference:** Utilizes the Groq API for near real-time responses from the Llama3 language model.
- **Document Q&A:** Ask questions about the content of a PDF document (`esp32_datasheet_en.pdf` is used as an example).
- **Local Embeddings:** Uses Ollama with `nomic-embed-text` to generate embeddings locally.
- **Vector Store:** Employs FAISS for efficient similarity searches on document embeddings.
- **Web Interface:** A simple and interactive web interface built with Streamlit.

## How it Works

1.  **Document Loading:** The application loads all PDF documents from the `docs` directory using `PyPDFLoader`.
2.  **Text Splitting:** The documents are split into smaller chunks using `RecursiveCharacterTextSplitter`.
3.  **Embedding:** The text chunks are converted into vector embeddings using a local Ollama model.
4.  **Vector Storage:** The embeddings are stored in a FAISS vector store for quick retrieval.
5.  **User Query:** The user enters a question through the Streamlit interface.
6.  **Retrieval:** The application retrieves the most relevant document chunks from the vector store based on the user's query.
7.  **Generation:** The retrieved chunks and the user's question are passed to the Llama3 model via the Groq API to generate an answer.
8.  **Display:** The answer and the source document chunks are displayed in the Streamlit interface.

## Setup and Installation

### Prerequisites

-   Python 3.8+
-   [Ollama](https://ollama.com/) installed and running.
-   A Groq API key.

### 1. Clone the repository

```bash
git clone https://github.com/your-username/RAG_with_GroqAI.git
cd RAG_with_GroqAI
```

### 2. Create a virtual environment and install dependencies

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
pip install -r requirements.txt
```

### 3. Set up environment variables

Create a `.env` file in the root directory and add your Groq API key:

```
GROQ_API_KEY="your_groq_api_key"
```

You can optionally set the `OLLAMA_BASE_URL` if it's different from the default:
```
OLLAMA_BASE_URL="http://localhost:11434"
```

### 4. Download the embedding model

Pull the required embedding model using Ollama:
```bash
ollama pull nomic-embed-text
```

### 5. Prepare your document

Place the PDF files you want to query in the `docs` directory. The application will process all `.pdf` files found in this directory.

## Usage

1.  Run the Streamlit application:

    ```bash
    streamlit run app.py
    ```

2.  Open your web browser and go to the local URL provided by Streamlit (usually `http://localhost:8501`).

3.  Click the "Documents Embedding" button to process and embed the PDF file.

4.  Once the embedding is complete, enter your question in the text box and press Enter. The answer will be displayed on the page.

## Project Structure

```
.
├── .env              # Environment variables
├── .gitignore
├── app.py            # Main Streamlit application
├── docs/             # Directory for your PDF documents
│   └── example.pdf
├── README.md
└── requirements.txt  # Python dependencies
```
