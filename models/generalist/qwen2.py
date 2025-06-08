from ..base import BaseModel

class Qwen2(BaseModel):
    
    def __init__(self):
        super().__init__("Alibaba-NLP/gte-Qwen2-7B-instruct", model_kwargs={"torch_dtype":"bfloat16"})

    def encode_query(self, query):
        query = f"""Instruct: Given a research question, retrieve passages that answer the question.\nQuery:{query}"""
        return self.model.encode(query)


