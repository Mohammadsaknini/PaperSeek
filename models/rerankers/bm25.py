from sklearn.feature_extraction import _stop_words
from rank_bm25 import BM25Plus
from models.base import BaseReranker
import string

class BM25Reranker(BaseReranker):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def _tokenize(self, text: str):
        tokenized_doc = []
        for token in text.lower().split():
            token = token.strip(string.punctuation)
            if len(token) > 0 and token not in _stop_words.ENGLISH_STOP_WORDS:
                tokenized_doc.append(token)
        return tokenized_doc

    def rerank(self, query : str, documents: dict[str, str]) -> dict[str, float]:
        tokenized_query = self._tokenize(query)
        tokenized_documents = [self._tokenize(doc) for doc in documents.values()]
        bm25 = BM25Plus(tokenized_documents)
        scores = bm25.get_scores(tokenized_query)
        results = {doc_id: score for doc_id, score in zip(documents.keys(), scores)}
        return dict(sorted(results.items(), key=lambda x: x[1], reverse=True))
    