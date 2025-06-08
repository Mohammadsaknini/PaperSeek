from ..base import BaseModel
from openai import OpenAI
from tqdm import tqdm
import numpy as np
import time

class OpenAi_small(BaseModel):
    def __init__(self, **kwargs):
        self.client = OpenAI()

    def encode_query(self, text):
        response = self.client.embeddings.create(
            input=text, model="text-embedding-3-small"
        )
        return np.array(response.data[0].embedding)


    def encode(self, texts, **kwargs):
        embeddings = []
        chunk_size = 2000
        texts = [text[:6000] for text in texts]
        for i in tqdm(range(0, len(texts), chunk_size), desc="Requesting Embeddings"):
            response = self.client.embeddings.create(
                input=texts[i : i + chunk_size], model="text-embedding-3-small"
            )
            embeddings.extend([item.embedding for item in response.data])
        return np.vstack(embeddings)
