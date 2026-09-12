import streamlit as st
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from indexer import clone_repository, load_and_split_documents, build_vector_store

load_dotenv()

st.set_page_config(page_title="RepoLens", page_icon="🔍", layout="wide")

def format_docs(docs):
    """Formats retrieved code chunks for the LLM."""
    formatted = []

    for doc in docs:
        source = doc.metadata.get("source", "unknown file")

        formatted.append(
            f"--- Code Snippet ---\n"
            f"File: {source}\n\n"
            f"{doc.page_content}"
        )

    return "\n\n".join(formatted)


def build_rag_chain(vector_store, k):
    """Builds the RAG chain using the selected number of chunks."""

    retriever = vector_store.as_retriever(search_kwargs={"k": k})

    llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

    template = """Answer the question based ONLY on the following retrieved code snippets. 
    If you do not know the answer or if it isn't in the code context, state that you cannot find it in the codebase.
    
    Retrieved Code Context: {context}

    Question: {question}

    Answer:"""

    prompt = ChatPromptTemplate.from_template(template)

    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# Session state
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None

if "indexed_repo" not in st.session_state:
    st.session_state.indexed_repo = None

if "messages" not in st.session_state:
    st.session_state.messages = []


# Header
st.title("RepoLens")
st.caption("A RAG-powered developer assistant for querying GitHub repositories.")


# Sidebar
with st.sidebar:

    st.header("Repository")
    repo_url = st.text_input("GitHub Repository URL", placeholder="https://github.com/user/repository.git")
    index_button = st.button("Index Codebase", type="primary", use_container_width=True)
    st.divider()
    st.header("Retrieval Settings")

    k = st.slider(
        "Number of code fragments",
        min_value=1,
        max_value=10,
        value=4,
        help="Number of relevant code fragments retrieved for each question asked."
    )

    st.caption(f"RepoLens will retrieve {k} code fragment(s) for each question.")


# Repository indexing
if index_button:
    if not repo_url.strip():
        st.error("Please enter a GitHub repository URL.")
    else:
        try:
            with st.spinner("Cloning repository and building code index..."):
                repo_path = clone_repository(repo_url)
                chunks = load_and_split_documents(repo_path)
                vector_store = build_vector_store(chunks)

                st.session_state.vector_store = vector_store
                st.session_state.indexed_repo = repo_url
                st.session_state.messages = []

            st.success("Codebase indexed successfully!")
        except Exception as e:
            st.error(f"Error while indexing repository: {e}")


# Active repository
if st.session_state.indexed_repo:
    st.info(f"Active repository: {st.session_state.indexed_repo}")

# Chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
user_query = st.chat_input("Ask a question about the repository...")

if user_query:
    if st.session_state.vector_store is None:
        st.warning("Please index a GitHub repository first.")
    else:
        st.session_state.messages.append({"role": "user", "content": user_query})
        
        with st.chat_message("user"):
            st.markdown(user_query)

        with st.chat_message("assistant"):
            with st.spinner("Searching the codebase..."):
                try:
                    rag_chain = build_rag_chain(st.session_state.vector_store, k)
                    response = rag_chain.invoke(user_query)
                    st.markdown(response)

                    st.session_state.messages.append({"role": "assistant", "content": response})

                except Exception as e:
                    st.error(f"Error generating response: {e}")