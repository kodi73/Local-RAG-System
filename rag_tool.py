import os
from langchain_community.document_loaders import PyPDFLoader, UnstructuredMarkdownLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

os.environ["TOKENIZERS_PARALLELISM"] = "false"

load_dotenv()

PRIVATE_DATA_DIR = "private_data"
CHROMA_DIR = "chroma_store"

def load_documents():
    documents = []
    for file in os.listdir(PRIVATE_DATA_DIR):
        path = os.path.join(PRIVATE_DATA_DIR, file)
        if file.endswith(".pdf"):
            loader = PyPDFLoader(path)
        elif file.endswith(".md"):
            loader = UnstructuredMarkdownLoader(path)
        else:
            continue
        documents.extend(loader.load())
    return documents

def build_vector_store(docs):
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = splitter.split_documents(docs)
    
    embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    
    vectordb = Chroma.from_documents(chunks, embedding_model, persist_directory=CHROMA_DIR)
    vectordb.persist()
    return vectordb

def create_qa_chain(vectordb):
    retriever = vectordb.as_retriever(search_type="similarity", search_kwargs={"k": 3})

    llm = ChatOpenAI(
        openai_api_base="https://api.groq.com/openai/v1",
        openai_api_key=os.getenv("GROQ_API_KEY"),
        model_name="llama-3.3-70b-versatile",
        temperature=0
    )
    qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
    return qa_chain

def main():
    print("Loading documents...")
    docs = load_documents()
    
    print("Creating vector store...")
    vectordb = build_vector_store(docs)
    
    print("Building RAG QA system...")
    qa = create_qa_chain(vectordb)
    
    print("\n✅ Ready to take questions from your private data!")
    while True:
        query = input("\nAsk a question (or 'exit'): ")
        if query.lower() in ["exit", "quit"]:
            break
        result = qa.invoke({"query": query})
        print("\nAnswer:", result)

if __name__ == "__main__":
    main()