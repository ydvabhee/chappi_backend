from pinecone import Pinecone
import os




def get_pinecone() -> Pinecone:
  api_key = os.getenv("PINECONE_API_KEY")
  pc =  Pinecone(api_key=api_key)
  return pc