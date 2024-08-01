from langchain_text_splitters import RecursiveCharacterTextSplitter


def get_text_splitter() -> RecursiveCharacterTextSplitter:
  ts= RecursiveCharacterTextSplitter(
    chunk_size=1000, chunk_overlap=0
  )
  return ts