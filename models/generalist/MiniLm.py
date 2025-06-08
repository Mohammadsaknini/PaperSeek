from ..base import BaseModel

class MiniLm(BaseModel):
    
    def __init__(self):
        super().__init__("sentence-transformers/all-MiniLM-L6-v2")