
from typing import List
from langchain_core.documents.base import Document

from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.chains import create_history_aware_retriever
from langchain_core.retrievers import RetrieverOutputLike
import time




CONTEXTUALIZE_SYSTEM_PROMPT = """Given a chat history and the latest user question \
which might reference context in the chat history, formulate a standalone question \
which can be understood without the chat history. Do NOT answer the question, \
just reformulate it if needed and otherwise return it as is."""

SYSTEM_PROMPT = (
    "You are a helpful assistant. "
    "You are an assistant for question-answering tasks. "
    "Given a chat history and the latest user question \
     which might reference context in the chat history"
    "Use the following pieces of retrieved context to answer the question. If you don't know the answer, say Thank you, I don't know the answer."
    "\n\n"
    "{context}"
)

def history_aware_retriever(
        llm,
        retriever,
        contextualize_q_system_prompt = CONTEXTUALIZE_SYSTEM_PROMPT
        )    -> RetrieverOutputLike :
 
  contextualize_q_prompt: ChatPromptTemplate = ChatPromptTemplate.from_messages(
      [
          ("system", contextualize_q_system_prompt),
          MessagesPlaceholder("chat_history"),
          ("human", "{input}"),
      ]
  )
  history_aware_retriever:RetrieverOutputLike = create_history_aware_retriever(
      llm, retriever, contextualize_q_prompt
  )
  return history_aware_retriever