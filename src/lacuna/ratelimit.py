"""Token-bucket rate limiter shared by every outbound client.

Limits are enforced in code, never in prompts (blueprint section 0). Blocking and single-process:
the codebase is synchronous until the API server arrives in Phase 2.
"""

from __future__ import annotations

import threading
import time


class RateLimiter:
    def __init__(self, rate_per_sec: float, burst: int = 1, clock=time.monotonic, sleep=time.sleep):
        if rate_per_sec <= 0:
            raise ValueError("rate_per_sec must be positive")
        self.rate = rate_per_sec
        self.capacity = max(1, burst)
        self.tokens = float(self.capacity)
        self.updated = clock()
        self.clock = clock
        self.sleep = sleep
        self.lock = threading.Lock()

    def _refill(self) -> None:
        now = self.clock()
        self.tokens = min(self.capacity, self.tokens + (now - self.updated) * self.rate)
        self.updated = now

    def acquire(self) -> float:
        """Block until a token is available. Returns the seconds waited."""
        waited = 0.0
        with self.lock:
            while True:
                self._refill()
                if self.tokens >= 1:
                    self.tokens -= 1
                    return waited
                need = (1 - self.tokens) / self.rate
                self.sleep(need)
                waited += need

    def penalise(self, seconds: float) -> None:
        """After a 429: empty the bucket and push the next token `seconds` away."""
        with self.lock:
            self.tokens = 0.0
            self.updated = self.clock() + seconds - (1 / self.rate)
