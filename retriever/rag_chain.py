import os
from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Pinecone as PineconeStore
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.retrieval_qa.base import RetrievalQA
from langchain_openai import OpenAIEmbeddings, OpenAI
from pinecone import Pinecone
# Load API Key
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
PINECONE_ENVIRONMENT = os.getenv("PINECONE_ENVIRONMENT")
PINECONE_INDEX_NAME = os.getenv("PINECONE_INDEX_NAME")
if not api_key:
    raise ValueError("OPENAI_API_KEY is not found. Make sure it's set in your .env file.")
print("API Key Loaded Successfully.")  # Debugging step
if not PINECONE_API_KEY:
    raise ValueError("PINECONE_API_KEY is not found. Make sure it's set in your .env file.")
print("PINECONE_API_KEY Loaded Successfully.")  # Debugging step
if not PINECONE_ENVIRONMENT:
    raise ValueError("PINECONE_ENVIRONMENT is not found. Make sure it's set in your .env file.")
print("PINECONE_ENVIRONMENT Loaded Successfully.")  # Debugging step
if not PINECONE_INDEX_NAME:
    raise ValueError("PINECONE_INDEX_NAME is not found. Make sure it's set in your .env file.")
print("PINECONE_INDEX_NAME Loaded Successfully.")  # Debugging step

def create_qa_chain():

    # Step 1: Load Document
    current_dir = os.getcwd()
    file_path = os.path.join(current_dir, r"retriever\sample.txt")

    loader = TextLoader(file_path) # Simple .txt file with your knowledge
    documents = loader.load()

    # Step 2: Split Document
    text_splitter = CharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    texts = text_splitter.split_documents(documents)

    # Step 3: Create Embeddings
    embeddings = OpenAIEmbeddings()
    # Step 3.1: Initialize Pinecone client
    pc = Pinecone(api_key=PINECONE_API_KEY, environment=PINECONE_ENVIRONMENT)
    
    # Check if index exists, else create
    index_name = ''.join(c.lower() for c in PINECONE_INDEX_NAME if c.isalnum() or c.isspace()).replace(' ', '-')
    if index_name not in pc.list_indexes().names():
        pc.create_index(
            name=index_name,
            dimension=1536,  # For OpenAI embeddings
            metric='cosine',
            spec={"serverless": {"cloud": "gcp", "region": "asia-southeast1-gcp"}}
        )
        print(f"Created new index: {index_name}")
    
    # Get the index instance
    index = pc.Index(index_name)

    # Step 4: Create vector store
    vectorstore = PineconeStore.from_documents(texts, embeddings, index_name=index_name)

    # Step 5: Set up Retriever
    retriever = vectorstore.as_retriever()

    # Step 6: Set up LLM + RetrievalQA Chain
    llm = OpenAI(temperature=0)
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

    return qa_chain
