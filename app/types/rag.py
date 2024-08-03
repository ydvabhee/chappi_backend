from enum import Enum
from pydantic import BaseModel



class ContextCreationType(str, Enum):
  TEXT = "text"
  FILE = "file"
  HTML = "html"




class ContextCreationData(BaseModel):
  data: str
  type: str


class ContextQueryBody(BaseModel):
  query: str
  context_id : str


class ContextUpdateBodyType(BaseModel):
  data: str
  type: str
  context_id :str