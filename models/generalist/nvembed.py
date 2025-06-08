from ..base import BaseModel
import warnings
warnings.filterwarnings("ignore")

class NvEmbed(BaseModel):
    
    def __init__(self):
        super().__init__('nvidia/NV-Embed-v2')
        self.model.max_seq_length = 4096
        self.model.tokenizer.padding_side="right"
        self.task = "Given a title and an abstract, retrieve the passage that is the most relevant to the title and abstract."
    
    def _add_eos_token(self, texts):
        if isinstance(texts, str):
            texts = [texts]
        return [text + self.model.tokenizer.eos_token for text in texts]
    
    def encode_query(self, texts):
        texts = self._add_eos_token(texts)
        query_prefix = "Instruct: "+ self.task +"\nQuery: "
        return super().encode(texts, prompt=query_prefix, normalize_embeddings=True)
    
    def encode_corpus(self, texts):
        texts = self._add_eos_token(texts)
        return super().encode(texts, normalize_embeddings=True)

    def encode(self, texts, query=True):
        raise NotImplementedError("Please use encode_query or encode_corpus instead.")

    
