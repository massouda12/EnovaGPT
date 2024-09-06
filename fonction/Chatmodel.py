from langchain_core.prompts import ChatPromptTemplate
from operator import itemgetter
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
import os 
import re 
from dotenv import load_dotenv
import PyPDF2
from pdfminer.high_level import extract_text
import csv
from langchain_community.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document

load_dotenv()

def initialize_services():
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
    index_name = "cv"

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=PINECONE_API_KEY)
    vectorstore = PineconeVectorStore(
    pinecone_api_key = PINECONE_API_KEY,
    embedding=embeddings,
    index_name='cv'
    )
    llm_groq = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name='llama3-8b-8192',
        temperature=0.1,
    )

    return docsearch, llm_groq,vectorstore ,embeddings,index_name

# Function to set up the retrieval-augmented generation chain
def setup_rag_chain(docsearch, llm_groq):
    RAG_PROMPT = """\
                    The user will give you a person name and i want you to answer like this : 
                    "phone number" : phone number ,
                    "email " : email , 
                    "diplome " : diplome 

                    If you cannot answer the question, please respond with 'I don't know'.
                    {context} 

                    Question:
                    {question}
                    Answer : 
"""
    retriever = docsearch.as_retriever()
    rag_prompt = ChatPromptTemplate.from_template(RAG_PROMPT)

    retrieval_augmented_generation_chain = (
           {"context": itemgetter("question") 
        | retriever, "question": itemgetter("question")}
        | RunnablePassthrough.assign(context=itemgetter("context"))
        | {"response": rag_prompt | llm_groq, "context": itemgetter("context")}
    )

    return retrieval_augmented_generation_chain

# Function to extract the response from the chain output
def extract_answer(response_dict):
    return response_dict.get('response').content

# Function to get the trimmed response using the RAG chain
async def get_trimmed_response(chain, question):
    response_dict = await chain.ainvoke({"question": question})
    return extract_answer(response_dict)

# Main function to run the query and get the response
async def chat(user_question):
    docsearch, llm_groq,vectorstore ,embeddings,index_name = initialize_services()
    retrieval_augmented_generation_chain = setup_rag_chain(docsearch, llm_groq)
    response_dict = await retrieval_augmented_generation_chain.ainvoke({"question": user_question})
    return extract_answer(response_dict)


def CSV(input_file):
    text = extract_text(input_file)
    lines = text.split('\n')
    for line in lines :
        phone_pattern = r'\b(?:\d\s?){8,11}\b'
        phone = re.findall(phone_pattern, text)
        email_pattern = r'[a-zA-Z0-9_.+-]+@[a-zA-Z0-9-]+\.[a-zA-Z0-9-.]+'
        email = re.findall(email_pattern, text)

    file_path = 'extracted_data.csv'

    with open(file_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Email', 'Phone Number'])

        max_len = max(len(email), len(phone))
        for i in range(max_len):
            row = []
            row.append(email[i] if i < len(email) else '')
            row.append(phone[i] if i < len(phone) else '')
            writer.writerow(row)
    
    return file_path


def extract_cv(file_path,vectorstore):
    
    loader = CSVLoader(file_path=file_path, encoding="utf-8", csv_args={'delimiter': ','})
    data = loader.load()
    data = loader.load()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=10)
    text_chunks = text_splitter.split_documents(data)
   
    docsearch= vectorstore.add_texts(texts=[t.page_content for t in text_chunks])
    os.remove(file_path)

    return docsearch


def extract_pdf(input_file,vectorstore,index_name,embeddings,PINECONE_API_KEY):
    pdf = PyPDF2.PdfReader(input_file)
    pdf_text = ""
    for page in pdf.pages:
        pdf_text += page.extract_text()
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    docs = text_splitter.split_text(pdf_text)
    documents = [Document(page_content=text) for text in docs]
    docsearch = vectorstore.add_texts(texts=[doc.page_content for doc in documents])
    
    docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings,pinecone_api_key=PINECONE_API_KEY)
    
    return docsearch

