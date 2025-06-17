# Abstract

The rapid expansion of scientific publications has rendered classical keyword
based literature search methods increasingly inadequate, as they often overlook
semantically relevant works or return irrelevant results with matching keywords.
In this thesis, we introduce PaperSeek, a semantic search framework designed to
enhance large-scale literature discovery by using research questions, domain-rel-
evant context, synthetic document generation, and active learning-based rerank-
ing. Starting from an OpenAlex snapshot of over 100 million English-language
works with abstracts, we benchmark nine embedding models for literature re-
trieval and select Stella as the most suitable model for this task. We curate an
evaluation dataset comprising 30 systematic literature reviews, which we use as a
benchmark for literature retrieval. Our experiments demonstrate that using five
targeted research questions, along with the title and abstract of a known relevant
paper answering these questions, and synthetically generated context, can boost
recall by 8.1 percentage points (p.p.). Furthermore, for literature screening, our
active learning-based reranking approach, which incorporates human feedback,
yields up to a 15 p.p. improvement over traditional query-based methods. This
work contributes to literature retrieval by introducing a scalable semantic search
framework that enhances recall through question-driven queries, contextual aug-
mentation, and active learning-based reranking, particularly within the context
of systematic literature review workflows.
