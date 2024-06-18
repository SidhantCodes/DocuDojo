from docudojo import get_pdf_text, get_text_chunks, get_vec_store, get_conv_chain, user_input
from dotenv import load_dotenv
import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
import shutil
from fastapi.middleware.cors import CORSMiddleware

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app=FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post('/upload-file')
async def upload_file(files: list[UploadFile]):
    try:
        pdf_docs=[]
        for file in files:
            with open(file.filename, 'wb') as buffer:
                shutil.copyfileobj(file.file, buffer)
            pdf_docs.append(open(file.filename, "rb")) 
            file.file.close()
        txt=get_pdf_text(pdf_docs)
        chunks=get_text_chunks(txt)
        get_vec_store(chunks)
        
        for pdf in pdf_docs:
            pdf.close()

        for file in files:
            os.remove(file.filename)

        return JSONResponse({"message": "Files processed and vector store created!"}, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/query')
async def user_query(query: str=Form(...)):
    answer=user_input(query)
    return JSONResponse(content={"response":answer})

