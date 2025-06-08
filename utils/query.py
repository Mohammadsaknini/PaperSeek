import json


class Query:
    def __init__(self, topic: str):
        self._data = json.load(open("data/evaluation_data.json", encoding="utf-8"))
        self._data = self._data[topic]

    @property
    def research_questions(self):
        return self._data["RQs"]

    @property
    def slr_query(self):
        return self._data["query"]

    def format(
        self, rq_count=1, include_sub_rq=False, supporting_texts: list[str] | str = None
    ):
        rq_count = min(rq_count, len(self.research_questions))
        rq_query = ""
        for i in range(1, rq_count + 1):
            rq = self.research_questions[f"RQ{i}"]
            rq_query += f"RQ{i}: {rq['main']}\n"

            if include_sub_rq:
                for j, sub_q in enumerate(rq["sub"]):
                    rq_query += f"RQ{i}.{j + 1}: {sub_q}\n"

        template = "Research Questions:\n{rqs} [SEP] Supporting Documents:\n {text}"

        if supporting_texts is None:
            return rq_query
        
        if isinstance(supporting_texts, list):
            supporting_texts = [template.format(rqs=rq_query, text=text) for text in supporting_texts]
        else:
            supporting_texts = template.format(rqs=rq_query, text=supporting_texts)
            
        return supporting_texts
