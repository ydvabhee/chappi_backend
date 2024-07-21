from enum import Enum
from pydantic import BaseModel



class ContextCreationType(str, Enum):
  TEXT = "text"
  FILE = "file"




class ContextCreationData(BaseModel):
  data: str
