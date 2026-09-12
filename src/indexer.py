import os
import shutil
import git
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from dotenv import load_dotenv

load_dotenv()

TEMP_REPO_DIR = "./temp_repos"
CHROMA_DB_DIR = "./.chroma_db"

def clone_repository(repo_url: str) -> str:
    """Clones a GitHub repository to a local temporary directory"""
    if os.path.exists(TEMP_REPO_DIR):
        shutil.rmtree(TEMP_REPO_DIR)
    
    print(f" Cloning {repo_url}...")
    git.Repo.clone_from(repo_url, TEMP_REPO_DIR)
    print(" Repository cloned successfully.")
    return TEMP_REPO_DIR


def load_and_split_documents(repo_path: str):
    """Loads text/code files from the repo and splits them into chunks"""

    print(" Parsing repository codebase...")

    supported_extensions = {
        ".py", ".js", ".jsx", ".ts", ".tsx",
        ".java", ".md", ".go", ".rb", ".c",
        ".cpp", ".cs", ".json", ".yaml", ".yml",
        ".txt",
    }

    documents = []

    for root, _, files in os.walk(repo_path):
        for filename in files:
            extension = os.path.splitext(filename)[1].lower()

            if extension not in supported_extensions:
                continue

            file_path = os.path.join(root, filename)

            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                documents.append(
                    Document(
                        page_content=content, metadata={"source": file_path, "file_name": filename},
                    )
                )

            except (UnicodeDecodeError, OSError):
                continue

    print(f" Found {len(documents)} source files.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150)
    chunks = text_splitter.split_documents(documents)

    print(f" Generated {len(chunks)} text chunks.")

    return chunks


def build_vector_store(chunks):
    """Generates embeddings via Gemini and indexes chunks into ChromaDB."""

    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-001")

    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR
    )
    
    return vector_store


def clear_session_data():
    """Cleans up cloned repo and ChromaDB directory after exit."""
    if os.path.exists(TEMP_REPO_DIR):
        shutil.rmtree(TEMP_REPO_DIR)
        
    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)