"""Every Cypher statement the graph layer runs, as named parameterised constants.

Blueprint 10: the graph path uses parameterised templates only; nothing here is ever built with
string formatting from data. Blueprint 7.1: `MERGE`, never `CREATE`, so re-ingest is idempotent.
Blueprint 6 (AuraDB rule): `Paper` nodes exist only for papers in a topic set, so `LINK_CITATIONS`
and `UPSERT_INCLUDES` `MATCH` their endpoints and never create stubs for outside references.

Batched statements take `$rows` and are fed in chunks by `GraphRepo`.
"""

# ---------------------------------------------------------------- schema

ENSURE_CONSTRAINTS = (
    "CREATE CONSTRAINT paper_paper_id IF NOT EXISTS FOR (p:Paper) REQUIRE p.paper_id IS UNIQUE",
    "CREATE CONSTRAINT author_author_id IF NOT EXISTS FOR (a:Author) REQUIRE a.author_id IS UNIQUE",
    "CREATE CONSTRAINT topic_topic_id IF NOT EXISTS FOR (t:Topic) REQUIRE t.topic_id IS UNIQUE",
)

# ---------------------------------------------------------------- writes

# rows: [{paper_id, props: {title, doi?, arxiv_id?, year?, venue?, citation_count?, ...extra},
#         authors: [{author_id, position, props: {name, ...extra}}]}]
# `+=` only touches keys present in `props`; GraphRepo drops None values so a sparser re-ingest
# never erases a property a richer source already set.
UPSERT_PAPERS = """
UNWIND $rows AS row
MERGE (p:Paper {paper_id: row.paper_id})
SET p += row.props
WITH p, row
UNWIND row.authors AS a
MERGE (au:Author {author_id: a.author_id})
SET au += a.props
MERGE (au)-[r:AUTHORED]->(p)
SET r.position = a.position
"""

UPSERT_TOPIC = """
MERGE (t:Topic {topic_id: $topic_id})
SET t.question = $question
SET t += $props
"""

# rows: [{paper_id, role, rank, score}]. Papers not yet in the graph are skipped (MATCH), so the
# caller upserts papers before memberships. Returns how many memberships were set.
UPSERT_INCLUDES = """
MATCH (t:Topic {topic_id: $topic_id})
UNWIND $rows AS row
MATCH (p:Paper {paper_id: row.paper_id})
MERGE (t)-[r:INCLUDES]->(p)
SET r.role = row.role, r.rank = row.rank, r.score = row.score
RETURN count(r) AS n
"""

# rows: [{src, dst, source}]. Both endpoints must already exist; an edge is created at most once.
LINK_CITATIONS = """
UNWIND $rows AS row
MATCH (src:Paper {paper_id: row.src})
MATCH (dst:Paper {paper_id: row.dst})
MERGE (src)-[r:CITES]->(dst)
ON CREATE SET r.source = row.source
RETURN count(r) AS n
"""

DELETE_BY_TEST_RUN = """
MATCH (n {test_run: $test_run})
DETACH DELETE n
RETURN count(n) AS n
"""

RESET_ALL = """
MATCH (n)
DETACH DELETE n
RETURN count(n) AS n
"""

# ---------------------------------------------------------------- reads

PAPER_IDS = "MATCH (p:Paper) RETURN p.paper_id AS paper_id"

NODE_COUNTS = "MATCH (n) RETURN labels(n)[0] AS label, count(*) AS n ORDER BY label"

RELATIONSHIP_COUNTS = "MATCH ()-[r]->() RETURN type(r) AS type, count(*) AS n ORDER BY type"

NEIGHBOURHOOD = """
MATCH (p:Paper {paper_id: $paper_id})
OPTIONAL MATCH (p)-[:CITES]->(c:Paper)
WITH p, collect(DISTINCT c {.paper_id, .title, .year}) AS cites
OPTIONAL MATCH (p)<-[:CITES]-(b:Paper)
WITH p, cites, collect(DISTINCT b {.paper_id, .title, .year}) AS cited_by
OPTIONAL MATCH (a:Author)-[r:AUTHORED]->(p)
WITH p, cites, cited_by, a, r
ORDER BY r.position
RETURN p {.*} AS paper, cites, cited_by,
       collect(a {.author_id, .name, position: r.position}) AS authors
"""

TOPIC_PAPERS = """
MATCH (t:Topic {topic_id: $topic_id})-[r:INCLUDES]->(p:Paper)
RETURN p.paper_id AS paper_id, p.title AS title, p.year AS year,
       r.role AS role, r.rank AS rank, r.score AS score
ORDER BY r.rank, p.paper_id
"""

TOPIC_COUNTS = """
MATCH (t:Topic {topic_id: $topic_id})
OPTIONAL MATCH (t)-[:INCLUDES]->(p:Paper)
WITH t, count(p) AS n_papers
OPTIONAL MATCH (t)-[:INCLUDES]->(a:Paper)-[:CITES]->(b:Paper)<-[:INCLUDES]-(t)
RETURN t.question AS question, n_papers, count(b) AS n_cites_within
"""

# Ranked by citations received from other members of the same topic, then by global count.
TOP_CITED_IN_TOPIC = """
MATCH (t:Topic {topic_id: $topic_id})-[:INCLUDES]->(p:Paper)
OPTIONAL MATCH (t)-[:INCLUDES]->(q:Paper)-[:CITES]->(p)
WITH p, count(q) AS in_set_citations
RETURN p.paper_id AS paper_id, p.title AS title, p.year AS year,
       p.citation_count AS citation_count, in_set_citations
ORDER BY in_set_citations DESC, citation_count DESC, paper_id
LIMIT $n
"""
