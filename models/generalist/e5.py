from ..base import BaseModel

class E5small(BaseModel):
    
    def __init__(self):
        super().__init__("intfloat/multilingual-e5-small")

    def encode_query(self, text, **kwargs):
        return super().encode_query(f"query: {text}", **kwargs)
    
    def encode(self, texts, **kwargs):
        texts = [f"passage: {text}" for text in texts]
        return super().encode(texts, **kwargs)