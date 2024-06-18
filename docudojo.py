from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
import google.generativeai as genai
from langchain_community.vectorstores import FAISS
from langchain_google_genai.chat_models import ChatGoogleGenerativeAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import PromptTemplate
from dotenv import load_dotenv

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))


def get_pdf_text(pdf_docs):
    txt=""
    for pdf in pdf_docs:
        reader=PdfReader(pdf)
        for page in reader.pages:
            txt+=page.extract_text()
    return txt

def get_text_chunks(text):
    txt_splitter=RecursiveCharacterTextSplitter(chunk_size=10000,chunk_overlap=1000)
    chunks=txt_splitter.split_text(text)
    return chunks

def get_vec_store(chunks):
    embeddings=GoogleGenerativeAIEmbeddings(model="models/embedding-001")
    vec_store=FAISS.from_texts(chunks, embedding=embeddings)
    vec_store.save_local("index")


def get_conv_chain():
    prompt_temp="""
        Give me a very detailed answer as possible, from the provided context, make sure to provide all the details, if the answer is not in
        provided context just say, "Answer is not available in the provided context", don't provide the wrong answers\n\n
        Context:\n {context}?\n
        Question: \n{question}\n

        Answer:
    """
    model=ChatGoogleGenerativeAI(model="gemini-pro", temperature=0.1)
    prompt=PromptTemplate(template=prompt_temp, input_variables = ["context", "question"])
    chain=load_qa_chain(model, chain_type="stuff", prompt=prompt)

    return chain

def user_input(query):
    embeddings = GoogleGenerativeAIEmbeddings(model = "models/embedding-001")
    db=FAISS.load_local("index", embeddings, allow_dangerous_deserialization=True)
    docs=db.similarity_search(query)
    chain=get_conv_chain()
    res=chain(
        {"input_documents":docs, "question": query}
        , return_only_outputs=True
    )
    return res

