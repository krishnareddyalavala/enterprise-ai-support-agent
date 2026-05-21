
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
import os
load_dotenv()
def load_documents():
    
    documents = []

    for file in os.listdir("data"):

        file_path = f"data/{file}"

        if file.endswith(".txt"):
            loader = TextLoader(file_path)
            loaded_docs = loader.load()
            documents.extend(loaded_docs)

        elif file.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            loaded_docs = loader.load()
            documents.extend(loaded_docs)

    return documents
def split_documents(documents):

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=300,
        chunk_overlap=50
    )

    chunks = text_splitter.split_documents(documents)

    return chunks

def create_vector_store(chunks):

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    vector_store = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory="chroma_db"
    )

    return vector_store
def retrieve_documents(vector_store, query):

    retriever = vector_store.as_retriever()

    retrieved_docs = retriever.invoke(query)

    return retrieved_docs
def generate_response(query, retrieved_docs):

    llm = ChatOpenAI(model="gpt-3.5-turbo")

    context = "\n".join([doc.page_content for doc in retrieved_docs])

    prompt = f"""
    Answer the question using ONLY the context below.

    Context:
    {context}

    Question:
    {query}
    """

    response = llm.invoke(prompt)

    return response.content
if __name__ == "__main__":

    docs = load_documents()

    chunks = split_documents(docs)

    vector_store = create_vector_store(chunks)

    query = "How do employees reset VPN passwords?"

    retrieved_docs = retrieve_documents(vector_store, query)

    response = generate_response(query, retrieved_docs)

    print("\nQUESTION:")
    print(query)

    print("\nRETRIEVED DOCUMENTS:")
    for doc in retrieved_docs:
        print(doc.page_content)

    print("\nAI RESPONSE:")
    print(response)
