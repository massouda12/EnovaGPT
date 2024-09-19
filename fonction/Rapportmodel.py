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
import pdfplumber


def initialize_services2():
    PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
    GROQ_API_KEY = os.environ.get('GROQ_API_KEY')
    index_name = "rapport"

    embeddings = HuggingFaceEmbeddings(model_name='sentence-transformers/all-MiniLM-L6-v2')
    docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings, pinecone_api_key=PINECONE_API_KEY)
    vectorstore = PineconeVectorStore(
    pinecone_api_key = PINECONE_API_KEY,
    embedding=embeddings,
    index_name='rapport'
    )
    llm_groq = ChatGroq(
        groq_api_key=GROQ_API_KEY,
        model_name='mixtral-8x7b-32768',
        temperature=0.1,
    )

    return docsearch, llm_groq,vectorstore ,embeddings,index_name


def setup_rag_chain(docsearch, llm_groq):
    RAG_PROMPT = """\
                    You are an assistant that give a general introduction about a Project. Given the content below, You will analyse the data that you have and give the user the most closest Project to his question and the Link to that Project.

                        {content}
                        Question:
                        {question}
                        If you cannot answer the question, please respond with 'I don't know'.

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


def extract_answer(response_dict):
    # Extract the 'content' field from the 'response' key
    return response_dict.get('response').content

# Invoke the chain and get the trimmed response
async def get_trimmed_response(chain, question):
    response_dict = await chain.ainvoke({"question": question})
    return extract_answer(response_dict)

# Main function to run the query and get the response
async def Rapportchat(user_question):
    docsearch, llm_groq,vectorstore ,embeddings,index_name = initialize_services()
    retrieval_augmented_generation_chain = setup_rag_chain(docsearch, llm_groq)
    response_dict = await retrieval_augmented_generation_chain.ainvoke({"question": user_question})
    return extract_answer(response_dict)


def process_pdf(input_file, vectorstore,index_name,embeddings,PINECONE_API_KEY):
    combined_text = ""

    # Open and process the PDF file
    with pdfplumber.open(input_file) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                lines = page_text.split('\n')
                text_after_matches = []
                capture = False
                general_conclusion_pattern = re.compile(r"(conclusion générale|general conclusion)", re.IGNORECASE)
    
                for line in lines:
                    # Look for various sections to start capturing text
                    if general_conclusion_pattern.search(line) or re.search(r"Abstract", line) or \
                       re.search(r"Conclusion and Perspectives", line)or re.search(r"CONCLUSION GÉNÉRALE & PERSPECTIVES") or \
                       re.search(r"Résumé", line) or\
                       re.search(r"Resume", line):
                        capture = True
                        text_after_matches.append(line)
                    elif capture:
                        line = line.strip()
                        if line:
                            text_after_matches.append(line)

                # Join the captured lines into a full text block
                full_text_after_matches = '\n'.join(text_after_matches)
                combined_text += full_text_after_matches

    # Add the file path as a clickable link
    current_directory = "file:///C:/Users/DELL/Desktop/Rapport%20enova/"
    file_path = current_directory + input_file
    combined_text += f"\nLink To Project: [{file_path}]"

    # Split the combined text for further processing
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    docs = text_splitter.split_text(combined_text)
    documents = [Document(page_content=text) for text in docs]

    # Add the processed text to the vector store
    docsearch = vectorstore.add_texts(texts=[doc.page_content for doc in documents])
    docsearch = PineconeVectorStore(index_name=index_name, embedding=embeddings,pinecone_api_key=PINECONE_API_KEY)


    return docsearch