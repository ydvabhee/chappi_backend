from langchain_huggingface import HuggingFaceEmbeddings




MODEL_NAME = "all-MiniLM-L6-v2"
def get_sentence_transformer_embeddings() -> HuggingFaceEmbeddings:

 embedding =  HuggingFaceEmbeddings(model_name=MODEL_NAME,)
 return embedding
