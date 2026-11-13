"""Lacuna: an agentic literature-review assistant over a scholarly knowledge graph.

Package map (see docs/blueprint.md section 12):
    settings   runtime configuration from the environment / .env
    models     shared Pydantic records (PaperRecord, AuthorRecord, SearchHit)
    db         Postgres connection, idempotent schema, usage ledgers
    cache      key/value caches (memory for tests, Postgres for real runs)
    ratelimit  token-bucket limiter shared by every outbound client
    llm        provider layer: free-tier chain, cache, quotas, spend gate
    ingest     OpenAlex / Semantic Scholar / arXiv clients
    graph      Neo4j repository and Cypher templates
    scout      seed search -> snowball -> ranked paper set
    deepread   FAMM-RAG: full-text, table and figure QA over ingested PDFs
    evals      E1 ... E6 runners and their datasets
"""
