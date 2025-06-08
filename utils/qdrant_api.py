from qdrant_client import QdrantClient, models
from qdrant_client.models import QueryResponse
from models import Stella
from typing import Union
from numpy import ndarray


class QdrantReader:
    def __init__(self):
        self.collection_name = "OpenAlex"
        self.timeout = 2000
        self.client = QdrantClient(
            url="localhost:6333",
            timeout=self.timeout,
            prefer_grpc=True,
        )
        self.model = Stella()

    def _fetch_point(self, query: Union[str, ndarray], n_hits=10000, **kwargs) -> QueryResponse | list[QueryResponse]:
        if isinstance(query, str):
            query = self.model.encode(query)

        hits = self.client.query_points(
            self.collection_name,
            query=query,
            with_payload=True,
            limit=n_hits,
            timeout=self.timeout,
            **kwargs,
        )

        return [hits]

    def fetch(self, query: Union[str, list[str], ndarray], n_hits=10000, **kwargs) -> QueryResponse | list[QueryResponse]:
        if not all(isinstance(q, str) for q in query) and not all(
            isinstance(q, (ndarray | list)) for q in query
        ):
            raise ValueError("All the queries must be of the same type")

        if isinstance(query, str):
            return self._fetch_point(query, n_hits=n_hits, **kwargs)

        if isinstance(query, list) and len(query) == 1:
            return self._fetch_point(query[0], n_hits=n_hits, **kwargs)

        if isinstance(query, ndarray) and query.ndim == 1:
            return self._fetch_point(query, n_hits=n_hits, **kwargs)
        
        
        query = self.model.encode(query, show_progress_bar=False)
        requests = [
            models.QueryRequest(
                query=q, with_payload=True, limit=n_hits
            )
            for q in query
        ]

        return self.client.query_batch_points(
            collection_name=self.collection_name,
            requests=requests,
            timeout=self.timeout,
            **kwargs,
        )

# Add Contrastive learning for SLRs, in which whereby we increment the positive samples (core pubs) iteratively
# and observe the change in scores for the positive and negative samples.