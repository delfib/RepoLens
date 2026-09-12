# RepoLens 🔍

**RepoLens** is an intelligent Retrieval-Augmented Generation (RAG) tool designed to help developers explore, understand, and question a GitHub codebases. It clones a given repository, indexes its source code files using Gemini embeddings, and provides an interactive CLI chat session powered by Google's Gemini models.

---

## Features

* Clones public GitHub repositories into a local environment automatically.
* Filters and splits common source code and configuration extensions (`.py`, `.js`, `.ts`, `.java`, `.go`, `.json`, etc.) into manageable context chunks.
* Generates semantic embeddings and stores them in a local vector database.
* Answers queries strictly based on the retrieved code snippets.

---

## Tech Stack

* **Language**: Python 3.10+
* **LLM & Embeddings**: Google Gemini API via `langchain-google-genai`
* **Orchestration**: LangChain Expression Language (LCEL)
* **Vector Store**: ChromaDB (`langchain-chroma`)
* **Git Operations**: GitPython

---

## Prerequisites

* **Python**: Version 3.10 or higher installed.
* **Git**: Installed and accessible from your system PATH.
* **Google Gemini API Key**: Obtainable from [Google AI Studio](https://aistudio.google.com/).

---

## Installation & Setup

### Set up a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate 
```

### Install dependencies
```bash
pip install -r requirements.txt
```

### Configure environment variables
Create a .env file in the root directory and add your Google API key:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

---

## Usage
### Launch the interactive CLI

```bash
python3 src/app.py
```

When prompted, enter a valid public GitHub repository URL. Ask questions about the repository.

Type `exit` or `quit` to end the session and automatically clean up temporary files.