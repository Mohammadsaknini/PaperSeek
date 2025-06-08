from ..base import BaseModel

class Linq(BaseModel):
    
    def __init__(self):
        super().__init__("Linq-AI-Research/Linq-Embed-Mistral", model_kwargs={"torch_dtype":"bfloat16"})

    def encode_query(self, query):
        query = f"""Instruct: Given a research question, retrieve passages that answer the question.\nQuery:{query}"""
        return self.model.encode(query)