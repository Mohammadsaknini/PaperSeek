from adapters import AutoAdapterModel
from transformers import AutoTokenizer
from ..base import BaseModel
from ..scientific import Specter2
import numpy as np

class Specter2Adhoc(BaseModel):

    def __init__(self):
        self.base_model = Specter2()
        self.model = AutoAdapterModel.from_pretrained('allenai/specter2_base')
        self.tokenizer = AutoTokenizer.from_pretrained("allenai/specter2_base")
        self.model.load_adapter(
            "allenai/specter2_adhoc_query",
            source="hf",
            load_as="specter2_adhoc_query",
            set_active=True,
        )
    
    def encode_query(self, text: list[str], **kwargs) -> np.ndarray:
        inputs = self.tokenizer(text, padding=True, truncation=True,
                                        return_tensors="pt", return_token_type_ids=False, max_length=512)
        output = self.model(**inputs, **kwargs)
        embeddings = output.last_hidden_state[:, 0, :]

        return embeddings.detach().numpy().flatten()
    
    def encode(self, text: str, **kwargs) -> np.ndarray:
        return self.base_model.encode(text, **kwargs)