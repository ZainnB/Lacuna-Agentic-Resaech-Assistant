"""Shared HTTP layer for the paper-API clients: cache, rate limit, retry and cost ledger.

Order of operations for every request: cache lookup (a hit touches nothing else), budget guard,
rate-limiter token, send, retry on 429/5xx/transport errors, then ledger bump and cache write.
Errors are never cached.
"""

from __future__ import annotations

import time
from datetime import date
from pathlib import Path

import httpx

from lacuna import db
from lacuna.cache import Cache, cache_key
from lacuna.ratelimit import RateLimiter

MAX_RETRIES = 4
BACKOFF_BASE_S = 1.0
BACKOFF_CAP_S = 60.0


class BudgetExceeded(RuntimeError):
    """Raised before a request is sent when it would push today's spend over the daily budget."""


def user_agent(contact_email: str | None) -> str:
    return f"lacuna/0.1 (mailto:{contact_email})" if contact_email else "lacuna/0.1"


class CachedClient:
    def __init__(
        self,
        source: str,
        cache: Cache,
        limiter: RateLimiter,
        conn=None,
        headers: dict | None = None,
        *,
        budget_usd: float | None = None,
        transport: httpx.BaseTransport | None = None,
        sleep=time.sleep,
        max_retries: int = MAX_RETRIES,
        timeout: float = 30.0,
    ) -> None:
        self.source = source
        self.cache = cache
        self.limiter = limiter
        self.conn = conn
        self.budget_usd = budget_usd
        self.sleep = sleep
        self.max_retries = max_retries
        self.http = httpx.Client(
            headers={"User-Agent": "lacuna/0.1", **(headers or {})},
            timeout=timeout,
            transport=transport,
            follow_redirects=True,
        )
        # In-memory ledger, used when no connection is given: {day: {calls, cost_usd, rate_limited}}.
        self._ledger: dict[date, dict] = {}
        self.network_calls = 0  # requests actually sent, for tests and the CLI

    # -- ledger ---------------------------------------------------------------------------------

    def usage_today(self) -> dict:
        if self.conn is not None:
            row = db.usage_today(self.conn, "api_usage", source=self.source)
            return {
                k: float(v) if k == "cost_usd" else int(v)
                for k, v in row.items()
                if k in ("calls", "cost_usd", "rate_limited")
            }
        return dict(
            self._ledger.get(db.quota_day(self.source), {"calls": 0, "cost_usd": 0.0, "rate_limited": 0})
        )

    def spent_today(self) -> float:
        return float(self.usage_today().get("cost_usd", 0.0))

    def _bump(self, **inc) -> None:
        if self.conn is not None:
            db.bump_usage(
                self.conn, "api_usage", {"day": db.quota_day(self.source), "source": self.source}, **inc
            )
            return
        row = self._ledger.setdefault(
            db.quota_day(self.source), {"calls": 0, "cost_usd": 0.0, "rate_limited": 0}
        )
        for k, v in inc.items():
            row[k] += v

    def _guard_budget(self, cost: float) -> None:
        if self.budget_usd is not None and cost > 0 and self.spent_today() + cost > self.budget_usd + 1e-9:
            raise BudgetExceeded(
                f"{self.source}: spent ${self.spent_today():.4f} today, a ${cost:.4f} call would exceed "
                f"the ${self.budget_usd:.2f} daily budget"
            )

    # -- requests -------------------------------------------------------------------------------

    def get(self, url, params=None, *, cost=0.0, as_text=False, auth_params=None):
        return self.request("GET", url, params=params, cost=cost, as_text=as_text, auth_params=auth_params)

    def post(self, url, params=None, json=None, *, cost=0.0, auth_params=None):
        return self.request("POST", url, params=params, json=json, cost=cost, auth_params=auth_params)

    def request(self, method, url, *, params=None, json=None, cost=0.0, as_text=False, auth_params=None):
        """Return parsed JSON (or text). `auth_params` are sent but kept out of the cache key."""
        key = cache_key(self.source, method, url, params or {}, json)
        hit = self.cache.get(key)
        if hit is not None:
            return hit["_text"] if "_text" in hit else hit.get("_json", hit)

        self._guard_budget(cost)
        send_params = {**(params or {}), **(auth_params or {})}
        resp = None
        for attempt in range(self.max_retries + 1):
            self.limiter.acquire()
            self.network_calls += 1
            try:
                resp = self.http.request(method, url, params=send_params or None, json=json)
            except httpx.TransportError:
                self._bump(calls=1)
                if attempt == self.max_retries:
                    raise
                self.sleep(self._backoff(attempt, None))
                continue
            if resp.status_code == 429:
                wait = self._backoff(attempt, resp.headers.get("Retry-After"))
                self._bump(calls=1, rate_limited=1)
                self.limiter.penalise(wait)
                if attempt == self.max_retries:
                    break
                self.sleep(wait)
                continue
            if resp.status_code >= 500:
                self._bump(calls=1)
                if attempt == self.max_retries:
                    break
                self.sleep(self._backoff(attempt, resp.headers.get("Retry-After")))
                continue
            break
        resp.raise_for_status()  # a 429/5xx that survived the retries, or a 4xx
        self._bump(calls=1, **({"cost_usd": cost} if cost else {}))
        if as_text:
            self.cache.set(key, self.source, {"_text": resp.text})
            return resp.text
        data = resp.json()
        self.cache.set(key, self.source, data if isinstance(data, dict) else {"_json": data})
        return data

    @staticmethod
    def _backoff(attempt: int, retry_after: str | None) -> float:
        if retry_after:
            try:
                return min(max(float(retry_after), 0.0), BACKOFF_CAP_S)
            except ValueError:
                pass
        return min(BACKOFF_BASE_S * 2**attempt, BACKOFF_CAP_S)

    def download(self, url: str, dest: Path) -> Path:
        """Fetch a binary file (not cached in the DB; an existing file is kept)."""
        dest = Path(dest)
        if dest.exists() and dest.stat().st_size > 0:
            return dest
        self.limiter.acquire()
        self.network_calls += 1
        resp = self.http.get(url)
        self._bump(calls=1)
        resp.raise_for_status()
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(resp.content)
        return dest

    def close(self) -> None:
        self.http.close()
