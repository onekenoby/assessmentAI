"""Bounded concurrency and backpressure for RAG requests.

The limiter is intentionally independent from FastAPI.  A limiter instance is
owned by each ASGI application so tests and multi-worker deployments do not
share event-loop primitives accidentally.
"""

from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import asdict, dataclass
from typing import Any


class CapacityError(RuntimeError):
    """Base error for the request-capacity gate."""


class CapacityRejectedError(CapacityError):
    """The request cannot enter the bounded execution queue."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = str(reason or "capacity_rejected")


@dataclass(frozen=True, slots=True)
class CapacitySnapshot:
    max_concurrent: int
    max_queued: int
    active: int
    queued: int
    available_slots: int
    queue_available: int
    accepting: bool
    saturated: bool
    closed: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class RequestCapacityLimiter:
    """FIFO-like bounded gate for expensive Hybrid-RAG requests.

    The gate bounds both active work and waiting callers.  Waiting is also
    time-bounded, preventing a slow Ollama call from creating an unbounded
    backlog.  Acquisition and release are cancellation-safe.
    """

    def __init__(
        self,
        *,
        max_concurrent: int,
        max_queued: int,
        acquire_timeout_seconds: float,
    ) -> None:
        if int(max_concurrent) <= 0:
            raise ValueError("max_concurrent deve essere maggiore di zero")
        if int(max_queued) < 0:
            raise ValueError("max_queued non può essere negativo")
        if float(acquire_timeout_seconds) <= 0:
            raise ValueError("acquire_timeout_seconds deve essere maggiore di zero")

        self._max_concurrent = int(max_concurrent)
        self._max_queued = int(max_queued)
        self._acquire_timeout_seconds = float(acquire_timeout_seconds)
        self._condition = asyncio.Condition()
        self._active = 0
        self._queued = 0
        self._closed = False

    @property
    def max_concurrent(self) -> int:
        return self._max_concurrent

    @property
    def max_queued(self) -> int:
        return self._max_queued

    @property
    def acquire_timeout_seconds(self) -> float:
        return self._acquire_timeout_seconds

    def snapshot(self) -> CapacitySnapshot:
        available_slots = max(0, self._max_concurrent - self._active)
        queue_available = max(0, self._max_queued - self._queued)
        accepting = (not self._closed) and (
            available_slots > 0 or queue_available > 0
        )
        saturated = (not self._closed) and available_slots == 0 and queue_available == 0
        return CapacitySnapshot(
            max_concurrent=self._max_concurrent,
            max_queued=self._max_queued,
            active=self._active,
            queued=self._queued,
            available_slots=available_slots,
            queue_available=queue_available,
            accepting=accepting,
            saturated=saturated,
            closed=self._closed,
        )

    async def acquire(self) -> None:
        async with self._condition:
            if self._closed:
                raise CapacityRejectedError("closed")

            if self._active < self._max_concurrent:
                self._active += 1
                return

            if self._queued >= self._max_queued:
                raise CapacityRejectedError("queue_full")

            self._queued += 1
            try:
                try:
                    await asyncio.wait_for(
                        self._condition.wait_for(
                            lambda: self._closed
                            or self._active < self._max_concurrent
                        ),
                        timeout=self._acquire_timeout_seconds,
                    )
                except TimeoutError as exc:
                    raise CapacityRejectedError("queue_timeout") from exc

                if self._closed:
                    raise CapacityRejectedError("closed")

                self._active += 1
            finally:
                self._queued -= 1

    async def release(self) -> None:
        async with self._condition:
            if self._active > 0:
                self._active -= 1
            self._condition.notify(1)

    @asynccontextmanager
    async def slot(self) -> AsyncIterator[CapacitySnapshot]:
        await self.acquire()
        try:
            yield self.snapshot()
        finally:
            await self.release()

    async def close(self) -> None:
        """Stop accepting new work and wake queued callers."""

        async with self._condition:
            self._closed = True
            self._condition.notify_all()


__all__ = [
    "CapacityError",
    "CapacityRejectedError",
    "CapacitySnapshot",
    "RequestCapacityLimiter",
]
