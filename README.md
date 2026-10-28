# RAG Customer Support Chatbot

A Retrieval-Augmented Generation (RAG) chatbot built with Python that helps answer customer queries about payment, refund, and delivery issues. The application retrieves relevant information from uploaded text documents and uses a locally running Large Language Model (LLM) to generate responses.

## Overview

This project explores how semantic search and large language models can work together to build a document-grounded customer support assistant.

Rather than relying exclusively on a language model's existing knowledge, the chatbot searches uploaded documents for relevant information and includes the retrieved content in the prompt sent to the model.

The application uses Streamlit to provide a simple interface for uploading documents, submitting questions, viewing answers, and inspecting retrieved context.

## Features

- **Document Upload:** Upload multiple `.txt` files through the Streamlit interface.
- **Smart Chunking:** Split documents into smaller text chunks to support retrieval.
- **Overlapping Chunks:** Preserve contextual continuity by including overlapping words between adjacent chunks.
- **Semantic Search:** Use text embeddings and cosine similarity to identify relevant document chunks.
- **Similarity Threshold:** Filter retrieved chunks using a configurable similarity threshold.
- **LLM-Powered Responses:** Generate answers using Mistral through Ollama.
- **Context Inspection:** View the retrieved chunks, source filenames, and similarity scores.
- **Local Inference:** Run the language model locally through Ollama.

## How It Works

The application follows a Retrieval-Augmented Generation pipeline.

### 1. Document Ingestion

Users upload text documents containing information relevant to customer support queries.

The application reads the uploaded documents and records their filenames and text.

### 2. Smart Chunking

The documents are divided into smaller chunks using the `smartChunking()` function.

The default configuration uses:

- Chunk size: 300 words
- Overlap: 80 words

The overlap helps preserve context between neighbouring chunks.

### 3. Embedding Generation

The application uses the following Sentence Transformers model:

`BAAI/bge-small-en-v1.5`

The model converts document chunks and the user's query into numerical vector representations called embeddings.

### 4. Semantic Retrieval

The application calculates cosine similarity between the query embedding and document embeddings.

Chunks meeting the configured similarity threshold are selected as relevant context.

The default threshold is `0.55`.

### 5. Prompt Construction

The retrieved chunks are combined with the user's question to construct a prompt for the language model.

The prompt instructs the model to answer customer queries clearly and avoid mixing unrelated information.

### 6. Response Generation

The prompt is sent to the Mistral model through Ollama.

The generated response is then displayed in the Streamlit application, alongside information about the retrieved context.

## Technology Stack

- **Python** — Application logic and RAG pipeline
- **Streamlit** — User interface
- **Sentence Transformers** — Text embedding generation
- **BAAI/bge-small-en-v1.5** — Embedding model
- **Ollama** — Local LLM runtime
- **Mistral** — Language model used for response generation
- **PyTorch** — Underlying tensor operations used by the embedding libraries

## Project Structure

```text
rag-customer-support-chatbot/
├── App/
│   └── app.py
├── rag.py
├── requirements.txt
├── .gitignore
└── README.md
```

- `App/app.py` — Streamlit interface for document uploads, user queries, and displaying responses.
- `rag.py` — Document processing, text chunking, semantic search, prompt construction, and LLM response generation.
- `requirements.txt` — Python dependencies.
- `.gitignore` — Files and directories excluded from Git tracking.
- `README.md` — Project documentation.

## Installation and Setup

### Prerequisites

Install the following before running the application:

- Python 3.10 or a compatible Python version
- Git
- Ollama

### 1. Clone the Repository

```bash
git clone https://github.com/tainwala-ayush/rag-customer-support-chatbot.git
cd rag-customer-support-chatbot
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

On Windows:

```bash
.venv\Scripts\activate
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Install and Prepare the Language Model

Install Ollama from:

https://ollama.com/

Download the Mistral model:

```bash
ollama pull mistral
```

Ensure Ollama is running and the model is available before using the chatbot.

### 5. Run the Application

From the project root, run:

```bash
streamlit run App/app.py
```

Streamlit will provide a local URL at which the application can be accessed.

## Usage

1. Launch the Streamlit application.
2. Upload one or more `.txt` files containing customer support information.
3. Enter a question in the query field.
4. Click **Ask**.
5. Read the generated response.
6. Expand the context section to inspect the retrieved chunks, filenames, and similarity scores.

The quality of the response depends on the relevance and quality of the uploaded documents, the retrieval threshold, and the language model's generation.

## Configuration

The main retrieval and generation parameters can be adjusted in the source code.

| Parameter | Default | Purpose |
|---|---|---|
| Chunk size | 300 words | Controls the approximate size of each text chunk |
| Chunk overlap | 80 words | Preserves shared context between adjacent chunks |
| Similarity threshold | 0.55 | Filters document chunks by cosine similarity |
| Embedding model | `BAAI/bge-small-en-v1.5` | Converts text into embeddings |
| LLM | `mistral` | Generates responses through Ollama |

## Limitations

- The application currently accepts plain-text `.txt` files.
- Retrieval quality depends on document content and the similarity threshold.
- A similarity score does not guarantee that a retrieved chunk contains the correct answer.
- The language model may generate inaccurate responses, even when relevant context is provided.
- The language model and embedding model require suitable local computing resources.
- The current implementation uses a straightforward in-memory retrieval approach rather than a persistent vector database.

## Future Improvements

Potential improvements include:

- Supporting PDF and DOCX document ingestion.
- Adding a persistent vector database for larger document collections.
- Improving chunking strategies and retrieval ranking.
- Adding conversation history and multi-turn interactions.
- Implementing stronger grounding and answer verification.
- Adding automated retrieval and response evaluations.
- Improving error handling and user feedback.
- Providing configurable model and retrieval settings through the interface.

## Author

**Ayush Tainwala**

GitHub: [@tainwala-ayush](https://github.com/tainwala-ayush)
