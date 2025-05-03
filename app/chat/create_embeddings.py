from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from app.chat.vector_store.pinecone import vector_store
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_embeddings_for_pdf(pdf_id: str, pdf_path: str):
    """
    Generate and store embeddings for the given pdf

    1. Extract text from the specified PDF.
    2. Divide the extracted text into manageable chunks.
    3. Generate an embedding for each chunk.
    4. Persist the generated embeddings.

    :param pdf_id: The unique identifier for the PDF.
    :param pdf_path: The file path to the PDF.

    Example Usage:

    create_embeddings_for_pdf('123456', '/path/to/pdf')
    """
    try:
        logger.info(f"Starting to process PDF: {pdf_path} with ID: {pdf_id}")
        
        # Load the PDF
        loader = PyPDFLoader(pdf_path)
        logger.info(f"PDF loaded successfully: {pdf_path}")

        # Split the text into chunks
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
        docs = loader.load_and_split(text_splitter)
        logger.info(f"PDF split into {len(docs)} chunks")

        for doc in docs:
            doc.metadata = {
                "page": doc.metadata["page"],
                "text": doc.page_content,
                "pdf_id": pdf_id,
            }
        
        # Add documents to the vector store
        logger.info(f"Adding {len(docs)} documents to vector store")
        vector_store.add_documents(docs)
        logger.info(f"Successfully added {len(docs)} documents to vector store")
        
    except Exception as e:
        logger.error(f"Error processing PDF {pdf_path}: {str(e)}", exc_info=True)
        raise
