"""Entity extraction over paper chunks: schema, LLM and host extraction, resolution, graph writes.

Heavy modules (embeddings, Neo4j) are imported lazily; `from lacuna.extract import schemas` is cheap.
"""
