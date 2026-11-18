# API notes (verified against official docs)

Read 2026-11-14 (docs last updated 2026-11-03 for OpenAlex). Re-verify before relying on numbers.

## OpenAlex

Sources: https://help.openalex.org/api/authentication/ , https://help.openalex.org/access/pricing/ ,
https://help.openalex.org/access/example-costs/ , https://help.openalex.org/api/get-single-entities/ ,
https://help.openalex.org/api/filtering/

- Base URL `https://api.openalex.org`. Plain GET.
- Key: free, from https://openalex.org/settings/api-key (needs an account), or `POST /api-keys` with the
  owner's real email (a link is emailed; key is inert until clicked, `403 key_not_activated` before).
  Never invent an email. Send as query param `api_key=...` or header `Authorization: Bearer ...`.
  Keep in `.env` (`OPENALEX_API_KEY`), never in files.
- Budget: no key = shared **$0.10/day per IP** (stated in the API's own `meta.notice`; docs say a free key
  is 10x); free key = **$1/day**; resets midnight UTC. Prepaid in $1 steps; annual plans from $5,000/yr.
- Prices per 1,000 calls: single-entity lookup free; list + filter $0.10; keyword search $1; semantic
  search $1; rerank +$1; content (PDF) download $10.
- Rate: 10 req/s no key, 30 req/s free key. Semantic search 1 req/s. A per-second 429 costs nothing.
  Response headers: `X-RateLimit-Limit/Remaining/Credits-Used/Reset`; `GET /rate-limit?api_key=` for usage.
- Limits: **100 OR values per filter** (`filter=openalex_id:W1|W2|...`), `per_page` max **100**, basic
  paging 10,000 results; cursor paging (`cursor=*`) needs a key. `sample` max 10,000.
- Lookup forms (free): `/works/W123`, `/works/doi:10.x/y` (or full doi.org URL), `/works/arxiv:2305.18290`
  (also an arxiv.org URL or `doi:10.48550/arxiv.2305.18290`), `/works/pmid:...`, `/works/mag:...`.
  An arXiv id resolves to the work holding arXiv's DOI, which may be the merged published article.
- `filter=cites:W123` (works citing W123) works in practice (checked 2026-11-14, returns results); the doc
  pages read did not describe it, so treat as observed, not documented. `filter=cited_by_count:>1`,
  `from_publication_date` / `to_publication_date` are documented.
- Use `select=` to trim fields; the `referenced_works` field is a list of full W-URLs.

## Semantic Scholar Academic Graph API

Sources: https://api.semanticscholar.org/api-docs/ , https://www.semanticscholar.org/product/api

- Base `https://api.semanticscholar.org/graph/v1`. Key header is `x-api-key` (case-sensitive).
- Endpoints: `GET /paper/{id}`, `POST /paper/batch` (body `{"ids":[...]}`, `fields=` query param),
  `GET /paper/search` (relevance), `GET /paper/search/bulk` (token paging, up to 10,000,000 papers),
  `GET /paper/{id}/references`, `GET /paper/{id}/citations` (`limit` <= 1000, `offset`),
  `GET /author/{id}`, `POST /author/batch`. `GET /paper/search` `limit` <= 100.
- ID forms for `{id}`: S2 sha, `CorpusId:`, `DOI:`, `ARXIV:2106.15928`, and others listed per endpoint.
- **Batch limit: 500 ids per call**, 10 MB response cap, 9,999 citations per call.
- Rate limits (overview page): unauthenticated endpoints share **1,000 req/s among all unauthenticated
  users**, may be throttled in heavy periods; introductory limit for a key is **1 RPS on all endpoints**.
  Certain endpoints need a key. Docs recommend sending the key on every request.
- Key: request through the form at https://www.semanticscholar.org/product/api#api-key-form ; the key is
  emailed. Not requested yet for this project (no email supplied).

## arXiv API

Sources: https://info.arxiv.org/help/api/user-manual.html , https://info.arxiv.org/help/api/tou.html

- Query: `https://export.arxiv.org/api/query?search_query=...&id_list=...&start=0&max_results=N` (Atom XML).
  The manual shows `http://`; `https://` works (checked 2026-11-14). `id_list` is comma-delimited;
  with both given, it filters. Use `id_list` (not `id:`) and append `vN` for a version.
- Fields: `ti au abs co jr cat rn all`; `AND OR ANDNOT`; `submittedDate:[YYYYMMDDTTTT TO YYYYMMDDTTTT]` (GMT).
- Politeness: ToU says **no more than one request every 3 seconds, one connection at a time**, across all
  your machines. Manual: cache; results only change when new papers are added.
- `max_results` up to 2000 per slice, 30,000 total; refine queries over 1,000 results; use OAI-PMH for bulk.
- PDF: each entry has a `<link title="pdf" rel="related">` (e.g. `http://arxiv.org/pdf/<id>v<n>`);
  abs page `https://arxiv.org/abs/<id>`.
- Terms: descriptive metadata is CC0 and free to store and share. E-print content may be retrieved and
  stored for personal or research use; must not store and serve PDFs or sources from our servers unless
  the licence permits; do not imply arXiv endorsement; do not circumvent rate limits.
