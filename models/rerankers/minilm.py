from sentence_transformers import CrossEncoder
from models.base import BaseReranker

class MiniLmCE(BaseReranker):
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.model = CrossEncoder('cross-encoder/ms-marco-MiniLM-L-6-v2')
    
    def rerank(self, query, documents):
        inputs = [[query, doc] for doc in documents.values()]
        scores = self.model.predict(inputs, show_progress_bar=False)
        results = {doc_id: score for doc_id, score in zip(documents.keys(), scores)}
        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
