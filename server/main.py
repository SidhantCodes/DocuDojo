from docudojo import get_pdf_text, get_text_chunks, get_vec_store, get_conv_chain, user_input
from dotenv import load_dotenv
import os
import google.generativeai as genai
from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import JSONResponse
import shutil

load_dotenv()
genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

app=FastAPI()
