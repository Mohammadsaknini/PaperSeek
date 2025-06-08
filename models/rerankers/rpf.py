from collections import defaultdict


class ReciprocalRankFusion:
    def rerank(self, results: list[dict[str, float]] | list[list]):
        ranks = defaultdict(float)
        for result in results:
            if isinstance(result, dict):
                # Make sure is sorted
                result = dict(sorted(result.items(), key=lambda x: x[1], reverse=True))
                for rank, doc in enumerate(result.keys()):
                    ranks[doc] += 1 / (rank + 60)

            elif isinstance(result, list):
                # We assume they are already sorted
                for rank, doc in enumerate(result):
                    ranks[doc] += 1 / (rank + 60)

        return dict(sorted(ranks.items(), key=lambda x: x[1], reverse=True))
