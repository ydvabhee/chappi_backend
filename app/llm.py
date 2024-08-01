
from langchain_groq import ChatGroq

from app.embeddings import MODEL_NAME

# Chat Completion
# ID	Requests per Minute	Requests per Day	Tokens per Minute	Tokens per Day
# gemma-7b-it	30	14,400	15,000	(No limit)
# gemma2-9b-it	30	14,400	15,000	(No limit)
# llama-3.1-405b-reasoning	30	14,400	131,072	131,072
# llama-3.1-70b-versatile	30	14,400	131,072	131,072
# llama-3.1-8b-instant	30	14,400	131,072	131,072
# llama3-groq-70b-8192-tool-use-preview	30	14,400	15,000	(No limit)
# llama3-groq-8b-8192-tool-use-preview	30	14,400	15,000	(No limit)
# mixtral-8x7b-32768	30	14,400	5,000	(No limit)



TEMPERATURE = 0
MODEL_NAME  = ['llama3-70b-8192','llama3-70b-8192', 'llama-3.1-8b-instant','llama-3.1-70b-versatile', 'llama3-groq-70b-8192-tool-use-preview']


def get_groq_llm(temperature=TEMPERATURE, model_name=MODEL_NAME[4]) -> ChatGroq:

  groq = ChatGroq(
  temperature=temperature,
  model=model_name, 
  streaming=True,
  )
  return groq