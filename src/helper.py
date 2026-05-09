from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from typing import List
from langchain_core.documents import Document


# Extract text from PDF files(json, csv, txt, etc. can also be loaded using TextLoader)
def load_pdf_files(data):
    loader = DirectoryLoader(
        data,
        glob="*.pdf",  # all pdf loaded
        loader_cls=PyPDFLoader
    )

    documents = loader.load()
    return documents


# Backward compatibility alias
load_pdf_file = load_pdf_files


def filter_to_minimal_docs(docs: List[Document]) -> List[Document]:
    """
    Given a list of Document objects, return a new list of Document objects
    containing only 'source' in metadata and the original page_content.
    """
    minimal_docs: List[Document] = []
    for doc in docs:
        src = doc.metadata.get("source")
        minimal_docs.append(
            Document(
                page_content=doc.page_content,
                metadata={"source": src}
            )
        )
    return minimal_docs


# Split the Data into Text Chunks
def text_split(minimal_docs):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,  # Adjust the chunk size as needed
        chunk_overlap=20  # Adjust the chunk overlap as needed
    )
    texts_chunk = text_splitter.split_documents(minimal_docs)
    return texts_chunk


# Download the Embeddings from HuggingFace
def download_embeddings():
    """
    Download and return the HuggingFace embeddings model.
    """
    model_name = "sentence-transformers/all-MiniLM-L6-v2"  # 384 dimensions
    embeddings = HuggingFaceEmbeddings(
        model_name=model_name
    )
    return embeddings


# Backward compatibility alias
download_hugging_face_embeddings = download_embeddings
