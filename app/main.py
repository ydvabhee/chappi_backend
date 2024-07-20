
from contextlib import asynccontextmanager
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
from fastapi import FastAPI
from langchain_core.messages.base import BaseMessage

from app.llm import get_groq_llm




load_dotenv()



llm = None
vectordb = None
text_splitter = None 
@asynccontextmanager
async def lifespan(app: FastAPI):

    llm = get_groq_llm()
    app.state.llm = llm
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





app.state.test = 'test'

    

@app.get("/")
def greet_json(request: Request):

    query = 'This is a test api, greet the user'
    r: BaseMessage = request.app.state.llm.invoke(query)
    return r.content
