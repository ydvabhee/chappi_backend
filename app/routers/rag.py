from fastapi import APIRouter, Request
from langchain_core.vectorstores.base import VectorStoreRetriever
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pinecone import ServerlessSpec
from pinecone.grpc import PineconeGRPC 
from langchain.vectorstores.pinecone import Pinecone
from pinecone.grpc.index_grpc import GRPCIndex 
from app import llm
from app.types.rag import ContextCreationData, ContextCreationType
from app.utills.rag import get_pinecone_index
from langchain.chains import RetrievalQA
from langchain_pinecone import PineconeVectorStore


router = APIRouter(prefix="/rag", tags=["rag"])


@router.post("/context/" )
async def context(req : Request, data: str, type: ContextCreationType):

    texr_splitter: RecursiveCharacterTextSplitter = req.app.state.text_splitter
    pc: PineconeGRPC = req.app.state.pinecone
    embedding = req.app.state.embedding
    llm = req.app.state.llm

    if type == ContextCreationType.TEXT:
        # split the text
        texts = texr_splitter.split_text(data)
    elif type == ContextCreationType.FILE:
        # read the file
        pass
        
    index_name = "chappi"
    # create the pinecone index
    index = get_pinecone_index(pc, index_name)


    # create the embedding
   

    
    vectordb: PineconeVectorStore  = PineconeVectorStore.from_texts(texts, embedding, index_name=index_name)

    # get retriever
    retriever: VectorStoreRetriever = vectordb.as_retriever()
    query = 'give three question that could be asked '

    # make a chain
    

    chain = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever, return_source_documents=True)
    result = chain.invoke(query)

    q = result['result']
    a = result['result']
    source = result['source_documents']
    
    return  {"questions ": a}
