from ..base import BaseModel
from typing import Literal


class Stella(BaseModel):
    def __init__(self, **kwargs):
        super().__init__(
            "NovaSearch/stella_en_1.5B_v5",
            model_kwargs={"torch_dtype": "bfloat16"},
            **kwargs,
        )

    def encode_query(
        self, text, prompt_name: Literal["s2p_query", "s2s_query"] = "s2p_query"
    ):
        return super().encode(text, prompt_name=prompt_name)

    def encode(self, texts, **kwargs):
        return super().encode(texts, **kwargs)
