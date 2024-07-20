
from langchain_groq import ChatGroq


def get_groq_llm() -> ChatGroq:

  groq = ChatGroq(
  temperature=0,
  model="llama3-8b-8192", streaming=True,
  )
  return groq