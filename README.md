## Local RAG System with Llama 3.3

A simple Retrieval-Augmented Generation (RAG) system that uses local documents and a large language model (LLM) to answer questions. This project leverages LangChain, ChromaDB, and HuggingFace for a powerful, local question-answering tool.

### Features

    Document Loading: Loads and processes documents from PDF and Markdown files.

    Local Vector Store: Creates and persists a vector store using ChromaDB, allowing for fast, semantic search.

    HuggingFace Embeddings: Uses the all-MiniLM-L6-v2 model for efficient text embeddings.

    Groq API Integration: Connects to the Groq API to use a powerful LLM like Llama 3.3 for generating answers.

    RetrievalQA Chain: Utilizes LangChain's RetrievalQA chain to combine document retrieval with the LLM's generative capabilities.

### Prerequisites

Before running this project, you need to have the following installed:

    Python 3.8+

    pip package manager

You will also need an API key from Groq to access their models.

## Setup

Follow these steps to get the project running on your local machine.

1. Clone the repository

```
git clone <repository_url>
cd <repository_name>
```

2. Install dependencies

Install the required Python libraries using pip.
```
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Set up your environment variables

Create a .env file in the root directory of the project and add your Groq API key.
```
GROQ_API_KEY="your_groq_api_key_here"
```

4. Add your private data

Create a directory named private_data in the root of the project. Place your .pdf and .md files inside this directory. This is the data the RAG system will use to answer your questions.
```
/
├── private_data/
│   ├── document1.pdf
│   └── notes.md
├── .env
├── main.py
└── requirements.txt
```

### Usage

To start the RAG system, simply run the main.py script.
Bash
```
python rag_tool.py
```

The script will perform the following actions:

    Load documents from the private_data folder.

    Split the documents into manageable chunks and create a persistent vector store in a chroma_store directory.

    Set up the RetrievalQA chain.

Once the system is ready, you will be prompted to ask a question.
```
✅ Ready to take questions from your private data!

Ask a question (or 'exit'): What is the main topic of the PDF document?

Answer: ...

You can continue asking questions until you type exit or quit.
```
### How it works

The system follows a standard RAG pipeline:

    Document Loading: Documents are loaded using PyPDFLoader and UnstructuredMarkdownLoader.

    Chunking and Embedding: The documents are split into smaller chunks, and each chunk is converted into a vector representation using a HuggingFace embedding model. This process allows the system to understand the semantic meaning of the text.

    Vector Store: These vectors are stored in a ChromaDB vector database, which is a high-performance, on-disk storage solution.

    Retrieval: When you ask a question, the system converts your query into a vector and searches the vector store for the top 3 most semantically similar chunks.

    Generation: The retrieved chunks are then passed to the Llama 3.3 model via the Groq API, along with your original query. The LLM synthesizes this information to generate a comprehensive and accurate answer.

### Contributing

Feel free to open issues or submit pull requests.