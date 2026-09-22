CREATE EXTENSION IF NOT EXISTS vector;

-- No ANN index: ~2k chunks is small enough for exact search.
CREATE TABLE IF NOT EXISTS chunks (
    id        text PRIMARY KEY,
    paper_id  text NOT NULL,
    page      int  NOT NULL,
    section   text,
    text      text NOT NULL,
    embedding vector(768) NOT NULL
);

-- Which config produced the rows in `chunks`; run_eval refuses to score a stale index.
CREATE TABLE IF NOT EXISTS index_meta (
    key   text PRIMARY KEY,
    value text NOT NULL
);
