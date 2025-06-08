from sentence_transformers import SentenceTransformer
import numpy as np


class BaseModel:

    def __init__(self, model_name = None, model: SentenceTransformer = None, **kwargs):
        if model_name is None and model is None:
            raise ValueError("Either model_name or model should be provided.")

        if model_name is not None and model is not None:
            raise ValueError("Only one of model_name or model should be provided.")

        if model is not None:
            self.model = model
        else:
            self.model = SentenceTransformer(model_name, trust_remote_code=True, **kwargs).cuda()

    def encode(self, text, **kwargs) -> np.ndarray:
        
        if "show_progress_bar" in kwargs:
            show_progress_bar = kwargs.pop("show_progress_bar")
        else:
            show_progress_bar = True

        if isinstance(text, str) or (isinstance(text, list) and len(text) == 1):
            return self.model.encode(text, convert_to_numpy=True, **kwargs)
        
        return self.model.encode(text, show_progress_bar=show_progress_bar, convert_to_numpy=True, **kwargs)
    
    def encode_query(self, text, **kwargs) -> np.ndarray:
        return self.encode(text, **kwargs)

    def encode_parallel(self, text, devices: list, **kwargs) -> np.ndarray:
        if len(devices) == 1:
            return self.encode(text, **kwargs)
        
        pool = self.model.start_multi_process_pool(devices)
        if isinstance(text, str) or (isinstance(text, list) and len(text) == 1):
            encodings = self.model.encode_multi_process(text, pool, **kwargs)
        else:
            encodings = self.model.encode_multi_process(text, pool, show_progress_bar=True, **kwargs)
        
        self.model.stop_multi_process_pool(pool)
        
        return encodings



    
class BaseReranker:
    
    def __init__(self, **kwargs):
        pass

    def rerank(self, query: str, documents: dict[str, str], **kwargs) -> dict[str, float]:
        """
        Reranks the documents based on the query.
        
        paramters
        ---------
        query : str
            The query to rerank the documents.
        documents : dict[str, str]
            The documents to rerank with the document id as key and the document text as value.

        returns
        -------
        dict[str, float]
            A dictionary with the document id as key and the score as value.
        
        """
        raise NotImplementedError("This method should be implemented in the derived class.")