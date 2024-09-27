import os
import shutil
from typing import List
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
from fonction.embedding_function import get_embedding_function
from langchain_community.vectorstores import Chroma
from langchain_pinecone import PineconeVectorStore
from pinecone.grpc import PineconeGRPC as Pinecone
from pinecone import ServerlessSpec

CHROMA_PATH = "c:/Users/chatt/Desktop/Nouveau dossier/Stage enova/chroma"
DATA_PATH = "C:/Users/DELL/Documents/GitHub/EnovaGPT/data"

def main(reset=False):
    if reset:
        print("✨ Clearing Database")
        clear_database()

    documents = load_documents()
    chunks = split_documents(documents)
    add_to_chroma(chunks)
    
def load_documents():
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    document_loader = PyPDFDirectoryLoader(DATA_PATH)
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "enova"
    if index_name not in pc.list_indexes():
        pc.create_index(
            name=index_name,
            dimension=384,
            metric="cosine",
            spec=ServerlessSpec(
                cloud="aws",
                region="us-east-1"
            ),
        )
    return document_loader.load()

def split_documents(documents: List[Document]):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=750,
        chunk_overlap=100,
        length_function=len,
        is_separator_regex=False,
    )
    return text_splitter.split_documents(documents)

def add_to_chroma(chunks: List[Document]):
        # Obtenir la fonction d'embedding à utiliser pour la vectorisation.
    embedding_function = get_embedding_function()
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    
        # Initialiser le stockage vectoriel Chroma.
    
    vectorstore = PineconeVectorStore(
    pinecone_api_key = PINECONE_API_KEY,
    embedding=embedding_function,
    index_name='enova'
    )
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index = pc.Index("enova")
    # Attribuer des identifiants uniques aux fragments et vérifier les doublons dans la base de données.
    chunks_with_ids = calculate_chunk_ids(chunks)
    query_result = index.query(
        vector=[0]*384,  # Dummy vector with the correct dimensionality
        top_k=10000,     # Set top_k large enough to retrieve many existing IDs
        include_metadata=True
    )
    existing_ids = set([match['metadata']['id'] for match in query_result['matches']])
    print(f"Number of existing documents in DB: {len(existing_ids)}")
    # Filtrer les fragments déjà présents dans la base de données.
    new_chunks = []
    for chunk in chunks_with_ids:
        if chunk.metadata["id"] not in existing_ids:
            new_chunks.append(chunk)
            
    # Ajouter les nouveaux fragments à la base de données et sauvegarder les changements.

    if len(new_chunks):
        print(f"👉 Adding new documents: {len(new_chunks)}")
        new_chunk_ids = [chunk.metadata["id"] for chunk in new_chunks]
        vectorstore.add_texts(
            texts=[chunk.page_content for chunk in new_chunks],
            metadatas=[chunk.metadata for chunk in new_chunks],
            ids=new_chunk_ids
        )
        
    else:
        print("✅ No new documents to add")

def calculate_chunk_ids(chunks: List[Document]):
    last_page_id = None
    current_chunk_index = 0

    for chunk in chunks:
        source = chunk.metadata.get("source")
        page = chunk.metadata.get("page")
        current_page_id = f"{source}:{page}"

        if current_page_id == last_page_id:
            current_chunk_index += 1
        else:
            current_chunk_index = 0

        chunk_id = f"{current_page_id}:{current_chunk_index}"
        last_page_id = current_page_id

        chunk.metadata["id"] = chunk_id

    return chunks

def clear_database():
    if os.path.exists(DATA_PATH):
        shutil.rmtree(DATA_PATH)
        os.makedirs(DATA_PATH)
    
    if check_index_exists("enova"):
        PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
        pc = Pinecone(api_key=PINECONE_API_KEY)
    try:
        # Delete the index
        pc.delete_index("enova")
        print(f"✅ Pinecone index deleted successfully.")
    except Exception as e:
        print(f"❌ Failed to delete index : {str(e)}")

def check_index_exists(index_name: str) -> bool:
    """Check if the specified index exists in Pinecone."""
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    pc = Pinecone(api_key=PINECONE_API_KEY)

    existing_indexes = pc.list_indexes()
    if index_name in existing_indexes:
        return True
    else:
        return False
if __name__ == "__main__":
    main()
