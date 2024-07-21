
from pinecone import ServerlessSpec, Pinecone
from pinecone.grpc.index_grpc import GRPCIndex


def get_pinecone_index(pinecone: Pinecone, index_name: str) -> GRPCIndex:
  if index_name not in pinecone.list_indexes().names():
        # if does not exist, create index
        pinecone.create_index(
            index_name,
            dimension=384,  # dimensionality of text-embedding-ada-002
            metric='cosine',
            spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
    ) 
        )
    # connect to index
  index: GRPCIndex = pinecone.Index(index_name)
  return index
