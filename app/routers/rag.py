from http import client
from re import template
from fastapi import APIRouter, Request
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter

from langchain_community.vectorstores import Pinecone

from regex import P 
from app import llm
from app.types.rag import ContextCreationData, ContextCreationType, ContextQueryBody, ContextUpdateBodyType
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.retrievers import RetrieverOutputLike
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableParallel
import uuid
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain.chains import RetrievalQA
from langchain import hub
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableLambda
import time
from langchain_qdrant import QdrantVectorStore, Qdrant
import os
from fastapi.responses import StreamingResponse
from qdrant_client import QdrantClient


QDRANT_API_KEY = os.environ.get('QDRANT_API_KEY')
QDRANT_URL = os.environ.get('QDRANT_API_URL')





router = APIRouter(prefix="/rag", tags=["rag"])


def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def inspect(state):
    """Print the state passed between Runnables in a langchain and pass it on"""
    print(state)
    return state

@router.post("/context/" )
async def context(req : Request, body: ContextCreationData):

    print(">> ", QDRANT_URL)
    print(" >> " ,QDRANT_API_KEY)

    data = body.data
    type = body.type

    

    texr_splitter: RecursiveCharacterTextSplitter = req.app.state.text_splitter
    embedding = req.app.state.embedding
    llm = req.app.state.llm

    if type == ContextCreationType.TEXT:
        # split the text
        texts = texr_splitter.split_text(data)
        # print("texts >>>>>>>>>>>>", texts)
    elif type == ContextCreationType.FILE:
        # read the file
        pass

        
        
    index_name = str(uuid.uuid4())
   
    vectordb: QdrantVectorStore = QdrantVectorStore.from_texts(
    texts=texts, embedding=embedding, url=QDRANT_URL, api_key=QDRANT_API_KEY, collection_name=index_name
)
    # get retriever
    retriever = vectordb.as_retriever()

    response_schemas = [
    ResponseSchema(name="questions", description="list of the user's question"),
    ]
    output_parser: StructuredOutputParser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()
    template= '''
    SYSTEM
    You are a precise, autoregressive question-answering system.

    HUMAN
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    If you don't know the answer, do nothing. Use five sentences maximum and keep the answer concise.
    Question: {question}

    Context: {context}

    Question: {question}


    give the output as best as possible.\n{format_instructions}

    '''

    prompt = PromptTemplate(
    template=template,
    input_variables=["question", 'context'],
    partial_variables={"format_instructions": format_instructions},
    )



    query = f'you have a innertext from a webpage as context, please analyze the context and give three questions based on the context'


    # RAG pipeline
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | RunnableLambda(inspect)  # Add the inspector here to print the intermediate results
        | prompt
        | RunnableLambda(inspect)  # Add the inspector here to print the intermediate results
        | llm
        | RunnableLambda(inspect)  # Add the inspector here to print the intermediate results
        | output_parser
    )
    a = chain.invoke(query)

    
    print(a)
    return  {"questions":a['questions'], "id":index_name}



async def context_query_stream(req : Request, body: ContextQueryBody):

    query = body.query
    context_id = body.context_id

    llm = req.app.state.llm
    embedding = req.app.state.embedding

    vectordb = QdrantVectorStore.from_existing_collection(
    embedding=embedding,
    collection_name=context_id,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY,
)
    # get retriever
    retriever = vectordb.as_retriever()

    output_parser = StrOutputParser()
    template= '''
    SYSTEM
    You are a precise, autoregressive question-answering system.

    HUMAN
    You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question.
    you have a innertext data from a webpage as context, please analyze the context and give an answer based on the context.
    If you don't know the answer, do nothing.
    Context: {context}

    Question: {question}


    '''

    prompt = PromptTemplate(
    template=template,
    input_variables=["question", 'context'],
    )

    # RAG pipeline
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
        | output_parser
    )
    # answer = chain.invoke(query)
    for answer in chain.stream(query):
        yield answer


@router.post("/context/query/")
async def context_query(req : Request, body: ContextQueryBody):
    return StreamingResponse(context_query_stream(req, body))


@router.put('/context/update')
async def context_update(req : Request, body: ContextUpdateBodyType):

    data = body.data
    type = body.type
    context_id = body.context_id


    embedding = req.app.state.embedding
    texr_splitter: RecursiveCharacterTextSplitter = req.app.state.text_splitter

    if type == ContextCreationType.TEXT:
        # split the text
        texts = texr_splitter.split_text(data)
    elif type == ContextCreationType.FILE:
        # read the file
        pass


    # Qdrand Client


    
    client = QdrantClient(
        url= QDRANT_URL,
        api_key= QDRANT_API_KEY
    )
    collection_name = context_id
    qdrant   = Qdrant(client, collection_name, embeddings=embedding)
    qdrant.add_texts(texts=texts)
   
    
    

