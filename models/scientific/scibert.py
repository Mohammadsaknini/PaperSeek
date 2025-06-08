from ..base import BaseModel

class SciBert(BaseModel):
    
    def __init__(self):
        super().__init__("jordyvl/scibert_scivocab_uncased_sentence_transformer")