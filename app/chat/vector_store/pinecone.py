import os
import pinecone
from langchain.vectorstores.pinecone import Pinecone
from app.chat.embeddings.openai import embeddings
from dotenv import load_dotenv
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables from the root directory
load_dotenv(os.path.join(os.path.dirname(__file__), '..', '..', '..', '..', '.env'))
logger.info("Pinecone API key set in environment variables")

# Initialize Pinecone with the class-based approach
try:
    pc = pinecone.Pinecone(os.environ["PINECONE_API_KEY"])
    logger.info("Pinecone client initialized successfully")
except Exception as e:
    logger.error(f"Error initializing Pinecone client: {str(e)}", exc_info=True)
    raise

# Get the index name
index_name = os.environ["PINECONE_INDEX_NAME"]
logger.info(f"Using Pinecone index: {index_name}")

# Create vector store using the existing index
try:
    vector_store = Pinecone.from_existing_index(
        index_name=index_name,
        embedding=embeddings,
    )
    logger.info("Vector store initialized successfully")
except Exception as e:
    logger.error(f"Error initializing vector store: {str(e)}", exc_info=True)
    raise

def build_retriever(chat_args, k=10):
    search_kwargs = {
        "filter": { "pdf_id": chat_args.pdf_id },
        "k": k
    }
    return vector_store.as_retriever(
        search_kwargs=search_kwargs
    )
