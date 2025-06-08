from sentence_transformers import models, SentenceTransformer
from transformers import AutoTokenizer
from ..base import BaseModel

class Specter2(BaseModel):

    def __init__(self):
        transformer = models.Transformer('allenai/specter2_base')
        tokenizer = AutoTokenizer.from_pretrained('allenai/specter2_base')
        transformer.tokenizer = tokenizer
        pooling = models.Pooling(transformer.get_word_embedding_dimension(), pooling_mode="mean")
        normalize = models.Normalize()
        model = SentenceTransformer(modules=[transformer, pooling, normalize])
        super().__init__(model=model)


    

