import os
import shutil
import git
from langchain_community.document_loaders import GenericLoader
from langchain_community.document_loaders.parsers import LanguageParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

# Directories for temporary repo storage and Chroma DB
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
    """Loads code/markdown files from the repo and splits them into chunks"""
    print(" Parsing repository codebase...")
    
    # Parse common programming languages and markdown
    loader = GenericLoader.from_filesystem(
        repo_path,
        glob="**/*",
        suffixes=[".py", ".js", ".jsx", ".ts", ".tsx", ".java", ".md", ".go", ".rb", ".c", ".cpp", ".cs", ".json", ".yaml", ".txt"],
        parser=LanguageParser()
    )
    documents = loader.load()
    print(f" Found {len(documents)} source files.")

    # Split documents into ~500 character chunks with 50 character overlap
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(documents)

    print(f" Generated {len(chunks)} text chunks.")

    return chunks


def build_vector_store(chunks):
    """Generates embeddings via Gemini and indexes chunks into ChromaDB."""
    
    embeddings = GoogleGenerativeAIEmbeddings(model="gemini-embedding-2-preview")

    if os.path.exists(CHROMA_DB_DIR):
        shutil.rmtree(CHROMA_DB_DIR)

    # Index chunks in local ChromaDB
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