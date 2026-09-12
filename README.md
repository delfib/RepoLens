# RepoLens

**RepoLens** is a RAG-powered developer assistant designed to help developers explore, understand, and question GitHub codebases. It clones a given repository, indexes its source code files using Gemini embeddings, and provides an interactive chat session.

---

### Features

- **GitHub Repository Analysis:** clone and index a repository directly from its GitHub URL.
- **Codebase Q&A:** ask natural-language questions about the indexed repository.
- **RAG-powered Retrieval:** retrieve relevant code fragments before generating an answer.
- **Adjustable Retrieval:** experiment with the number of code fragments retrieved for each question.
- **Web Interface:** interactive Streamlit interface for indexing repositories and chatting with RepoLens.

---

## Tech Stack

- [Python](https://www.python.org/)
- [Streamlit](https://streamlit.io/)
- [LangChain](https://www.langchain.com/)
- [Google Gemini](https://ai.google.dev/)
- [ChromaDB](https://www.trychroma.com/)
- [GitPython](https://gitpython.readthedocs.io/)

---

## Getting Started

### Prerequisites

* **Python**: Version 3.10 or higher installed.
* **Git**
* **Google Gemini API Key**: Obtainable from [Google AI Studio](https://aistudio.google.com/).

---

### Installation

1. Set up a virtual environment
```bash
python3 -m venv venv
source venv/bin/activate 
```

2. Install dependencies
```bash
pip install -r requirements.txt
```

3. Create a .env file in the root directory and add your Google API key:
```bash
GOOGLE_API_KEY=your_gemini_api_key_here
```

4. Run the Application
```bash
streamlit run src/app.py
```
Streamlit will provide a local URL where you can access the RepoLens web interface.

---

### Usage
1. Enter a public GitHub repository URL in the sidebar.
2. Wait for the repository to be processed and indexed.
3. Ask questions about the repository using the chat interface.
4. Use the Number of code fragments slider to experiment with retrieval.


When prompted, enter a valid public GitHub repository URL. Ask questions about the repository.

Type `exit` or `quit` to end the session and automatically clean up temporary files.

---

### Experimenting with Retrieval
RepoLens allows you to change the number of code fragments retrieved for each question using the `k` parameter.
The Streamlit interface allows you to select a value between 1 and 10.
Changing `k` changes how much relevant code is provided to Gemini when answering a question:

* Lower values provide less code context.
* Higher values provide more code context.
* Different values may produce different answers depending on the question and repository.

Try asking the same question with different k values to compare the results.


