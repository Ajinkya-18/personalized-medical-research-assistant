# Personalized Medical Research Assistant

This repository contains a small, focused project that builds a retrieval-augmented generation (RAG) style medical research assistant specialized for neuroscience-related documents. The application uses local PDF research papers (in `knowledge-base/`), embeds them with a pre-trained sentence-transformer, stores vectors in a local FAISS index, and serves a Streamlit chat-style UI that answers user questions grounded strictly in the provided documents.

## Streamlit App
[Personalized Neuroscience Research Assistant](https://personalized-neuroscience-research-assistant.streamlit.app/)

## Contents

- `app/` - Streamlit application that exposes a chat UI and a RAG chain using an Ollama LLM and the FAISS vector store.
- `ingest.py` - Script to load PDFs from `knowledge-base/`, split them into chunks, embed them and build the FAISS index saved to `faiss_index/`.
- `knowledge-base/` - Folder of PDF research papers used as the knowledge source. Example filenames are listed below.
- `faiss_index/` - Directory where the FAISS vector store is saved and loaded by the app.

## Why this project
The assistant is intended as a reproducible, local-first research helper: it keeps PDF sources under user control, uses open embedding models, stores vector indices locally, and restricts LLM answers to the supplied documents so responses stay evidence-based and traceable.

## Quick architecture summary

- **Document ingestion:** `ingest.py` uses `PyPDFDirectoryLoader` to load all PDFs from `knowledge-base/`, splits text with `RecursiveCharacterTextSplitter` (chunk_size=1000, overlap=150), embeds chunks with `sentence-transformers/all-MiniLM-L6-v2`, and persists a FAISS index under `faiss_index/`.
- **Runtime app:** `app/app.py` is a Streamlit app that loads an Ollama LLM (`gemma3:1b`) and the same Hugging Face embedding model, loads the FAISS index, creates a retriever (top K=5), builds a simple RAG chain with a prompt that instructs the LLM to answer only from the provided context, and exposes a chat interface.

## Requirements / Libraries

- streamlit
- langchain (and related community connectors used in the code: `langchain_community`, `langchain_huggingface`, `langchain_core`, etc.)
- sentence-transformers / Hugging Face embeddings
- FAISS (via the `langchain_community.vectorstores.FAISS` wrapper)
- Ollama for the local LLM (`langchain_community.llms.Ollama`).

## Installation (suggested)

1. Create and activate a Python virtual environment (example using PowerShell on Windows):

```powershell
python -m venv .venv; .\.venv\Scripts\Activate.ps1
```

1. Install project dependencies. The project does not include a pinned `requirements.txt` by default; install the libraries used in the code. Example (adjust versions as needed):

```powershell
pip install streamlit langchain langchain-community langchain-huggingface sentence-transformers faiss-cpu ollama
```

## Notes:

- On Windows, installing FAISS can sometimes be problematic; `faiss-cpu` is the recommended pip package for CPU-only usage. If you need GPU support, follow FAISS's platform-specific installation instructions.
- `ollama` may require a running Ollama server or local runtime depending on how you use it. If you don't have Ollama available, you can swap the LLM in `app/app.py` with another LangChain-compatible LLM.

### How to build the FAISS index (ingest PDFs)

1. Put your PDFs in the `knowledge-base/` directory. The repository already contains example PDFs covering topics around altered states of consciousness, near-death experiences, terminal lucidity, and related neuroscience/psychology literature.

2. Run the ingestion script to create the vector store:

```powershell
python ingest.py
```

#### This will:

- Load all PDFs from `knowledge-base/` using `PyPDFDirectoryLoader`.
- Split documents into overlapping text chunks (1000 tokens with 150 overlap).
- Compute embeddings for each chunk using `sentence-transformers/all-MiniLM-L6-v2`.
- Create and save a FAISS index under `faiss_index/`.

## Running the app (Streamlit)

1. Ensure the FAISS index is present (created by `ingest.py`).

2. Start the Streamlit app from the project root:

```powershell
streamlit run app\\app.py
```

3. Interact with the chat UI. The app will:

- Load the Ollama LLM (`gemma3:1b`) and embedding model.
- Load the FAISS index and create a retriever (top K=5).
- Build a RAG chain that restricts answers to the retrieved context and respond to user prompts in a chat interface.

### Prompting and behavior

- The assistant explicitly uses a system prompt that says: "Answer the user's question based *only* on the following context. If the context does not contain the answer, state that you don't know." This helps keep answers grounded in the provided literature rather than hallucinating.
- The retriever returns up to 10 relevant chunks for each query; the chain then stitches them together and passes them with the question to the LLM.


## Security, privacy, and reproducibility notes

- All PDFs are stored locally. No external APIs are called for retrieval or storage — embeddings and FAISS index are local (though some embedding and LLM backends may call external services depending on configuration).
- The LLM backend in `app/app.py` uses `Ollama`. If you configure a cloud LLM, be aware of data sent to third-party services.

## Extending or swapping components

- Swap LLM: Replace the `Ollama` instantiation in `app/app.py` with any LangChain-compatible LLM.
- Swap embeddings: `HuggingFaceEmbeddings` is used with `all-MiniLM-L6-v2`. You can choose a different embedding model but keep consistent embedding dimensions when reusing an index.
- Increase retrieval K: modify `db.as_retriever(search_kwargs={"K":5})` in `app/app.py`.

## Troubleshooting

- If the app fails to load the FAISS index, verify `faiss_index` exists and contains the saved index files. If missing, run `python ingest.py`.
- If embeddings fail to load, ensure the `sentence-transformers` model is installed and that you have network access to download model weights (or pre-download them).

## License

See the repository LICENSE file for licensing details.

### Contributing

This project is a personal project. If you plan to extend it, prefer small, focused PRs that add tests and documentation.

-----------------------------------

