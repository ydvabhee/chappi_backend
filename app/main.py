
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.messages.base import BaseMessage

from app.embeddings import get_sentence_transformer_embeddings
from app.pinecone import get_pinecone
from app.text_splitter import get_text_splitter
from app.llm import get_groq_llm
from app.routers import rag


load_dotenv()




@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.llm = get_groq_llm()
    app.state.text_splitter = get_text_splitter()
    app.state.pinecone = get_pinecone()
    app.state.embedding = get_sentence_transformer_embeddings()
     
    yield

app = FastAPI(lifespan=lifespan)


# CORS 
origins = [
    "http://localhost.tiangolo.com",
    "https://localhost.tiangolo.com",
    "http://localhost",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



app.include_router(rag.router)

@app.get("/")
def greet_json(request: Request):

    query = 'This is a test api, greet the user'
    r: BaseMessage = request.app.state.llm.invoke(query)
    return r.content
