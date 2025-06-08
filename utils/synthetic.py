from openai import OpenAI, NOT_GIVEN
from pydantic import BaseModel


class Publication(BaseModel):
    title: str
    abstract: str


class HyResearch:
    """
    Hypothetical Research Embeddings (HyResearch) is a unified class that generates similar documents,
    queries, and publications for a given research input.
    """

    def __init__(self):
        self._client = OpenAI()
        self._agent = self._client.beta.chat.completions
        self._seeds = [i for i in range(42, 100)]
        self._template = """Title: {title}[SEP]\n 
        Abstract: {abstract}
        """

    def _generate_response(
        self,
        prompt: str,
        user_input: str,
        model: str = "o3-mini",
        response_format=NOT_GIVEN,
        seed=42,
    ):
        """
        Helper method to interact with the OpenAI agent.

        Parameters
        ----------
        prompt : str
            The system prompt to guide the AI response.
        user_input : str
            The user-provided input (document or question).
        model : str
            The model to use for generating the response.
        response_format : type
            The expected response format (e.g., Publication).

        Returns
        -------
        dict or str
            The parsed response from the AI agent.
        """
        completion = self._agent.parse(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": user_input},
            ],
            model=model,
            response_format=response_format,
            seed=seed,
        )
        if response_format:
            return eval(completion.choices[0].message.content)
        return completion.choices[0].message.content

    def generate_similar_doc(self, document: str, field: str, seed = 42) -> dict:
        prompt = f"""You are an expert in the field of {field}. Given a scientific publication, generate the abstract \
        of an expert-level research paper that targets the same problem. Stick to a maximum length of 250 tokens and return \
        just the text of the title and abstract, as a JSON in the following format:
        {{
            "title": "Title of the paper",
            "abstract": "Abstract of the paper"
        }}
        Use research-specific jargon."""

        pub = self._generate_response(prompt, document, response_format=Publication)
        return self._template.format(title=pub["title"], abstract=pub["abstract"], seed=seed)

    def generate_query(self, document: str, field: str, seed = 42) -> str: #TODO: Fix... why is ther a document? use it you monkey? maybe use context instead?
        prompt = f"""You are an expert in the field of {field}. Given a research question and a scientific publication, generate a keywords based query that \
        can be used to retrieve similar publications. Stick to a maximum length of 200 tokens and return just the text of the query. \
        Use research-specific jargon. Must have the following format:
        Keyword1, Keyword2, Keyword3, ...
        
        Input Data: {document}
        """

        return self._generate_response(prompt, document, seed=seed)

    def generate_core_pub(self, question: str, field: str, seed =42) -> dict:
        prompt = f"""You are an expert in the field of {field}. Given a research question, generate the title and abstract \
        of an expert-level research paper that targets the same problem. Stick to a maximum length of 350 tokens and return \
        just the text of the title and abstract, as a JSON in the following format:
        {{
            "title": "Title of the paper",
            "abstract": "Abstract of the paper"
        }}
        Use research-specific jargon."""

        pub = self._generate_response(prompt, question, response_format=Publication, seed=seed)
        return self._template.format(title=pub["title"], abstract=pub["abstract"])

    def generate_n_docs(self, document: str, field: str, n: int) -> list[dict]:
        return [self.generate_similar_doc(document, field, self._seeds[i]) for i in range(n)]

    def generate_n_queries(self, document: str, field: str, n: int) -> list[str]:
        return [self.generate_query(document, field, self._seeds[i]) for i in range(n)]

    def generate_n_pubs(self, question: str, field: str, n: int) -> list[dict]:
        return [self.generate_core_pub(question, field, self._seeds[i]) for i in range(n)]