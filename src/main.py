from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from ingest import clone_repository, load_and_split_documents, build_vector_store, clear_session_data

load_dotenv()

def main():
    print("=" * 60)
    print(" Welcome to RepoLens ")
    print("=" * 60)

    try:
        repo_url = input("\nEnter GitHub Repository URL: ").strip()
        if not repo_url:
            print("No URL provided. Exiting.")
            return

        # Ingest repo and populate vector database
        repo_path = clone_repository(repo_url)
        chunks = load_and_split_documents(repo_path)
        vector_store = build_vector_store(chunks)

        retriever = vector_store.as_retriever(search_kwargs={"k": 4})
        llm = ChatGoogleGenerativeAI(model="gemini-3.6-flash")

        template = """Answer the question based ONLY on the following retrieved code snippets.
        If you do not know the answer or if it isn't in the code context, state that you cannot find it in the codebase.

        Retrieved Code Context: {context}

        Question: {question}

        Answer:"""

        prompt = ChatPromptTemplate.from_template(template)

        def format_docs(docs):
            formatted = []

            for doc in docs:
                source = doc.metadata.get("source", "unknown file")

                formatted.append(
                    f"--- Code Snippet ---\n"
                    f"File: {source}\n\n"
                    f"{doc.page_content}"
                )

            return "\n\n".join(formatted)

        # Build LangChain RAG Chain
        rag_chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | llm
            | StrOutputParser()
        )

        print("\n" + "=" * 60)
        print(" RepoLens is ready! Ask any question about this codebase.")
        print(" Type 'exit' or 'quit' to end session.")
        print("=" * 60 + "\n")

        while True:
            user_query = input("\nRepoLens > ").strip()
            if user_query.lower() in ["exit", "quit"]:
                break
            if not user_query:
                continue

            print("\nSearching codebase & generating response...\n")
            response = rag_chain.invoke(user_query)
            print(f"Answer: {response}")

    except KeyboardInterrupt:
        print("\nSession interrupted by user.")
    except Exception as e:
        print(f"\n An error occurred: {e}")
    finally:
        clear_session_data()

if __name__ == "__main__":
    main()