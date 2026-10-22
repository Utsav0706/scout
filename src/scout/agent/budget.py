"""Run-wide limits on steps, money and time.

One RunBudget object is shared by every node of a run, including researchers running
in parallel. Keeping it outside the graph state matters: parallel branches each get a
snapshot of the state, so a counter kept in state would let every branch spend the
whole budget. The shared object sees all spending as it happens.

Trade-off: a check happens before each call, so parallel branches that all pass the
check at once can overshoot by at most one call each. Each call is itself capped by
max_tokens, so the overshoot is bounded.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field


class BudgetExceeded(Exception):
    pass


@dataclass
class RunBudget:
    max_steps: int
    max_cost_gbp: float
    timeout_s: float
    steps: int = 0
    cost_gbp: float = 0.0
    stop_reason: str | None = None
    _start: float = field(default_factory=time.monotonic)

    @property
    def elapsed_s(self) -> float:
        return time.monotonic() - self._start

    def check(self, reserve_gbp: float = 0.0) -> None:
        """Raise BudgetExceeded if another model call shouldn't start.

        reserve_gbp keeps money back for a later step (the writer), so research
        stops early instead of spending everything and leaving no report.
        """
        reason = None
        if self.steps >= self.max_steps:
            reason = f"step limit reached ({self.max_steps} model calls)"
        elif self.cost_gbp + reserve_gbp >= self.max_cost_gbp:
            reason = f"cost limit reached (£{self.cost_gbp:.3f} of £{self.max_cost_gbp:.2f})"
        elif self.elapsed_s >= self.timeout_s:
            reason = f"time limit reached ({self.timeout_s:.0f}s)"
        if reason:
            self.stop_reason = self.stop_reason or reason
            raise BudgetExceeded(reason)

    def has_room(self, gbp: float) -> bool:
        """Is there at least `gbp` of budget left (and steps and time)? Unlike check(),
        a "no" here is a decision (e.g. skip the retry round), not a stop: it doesn't
        set stop_reason or mark the report partial."""
        return (
            self.steps < self.max_steps
            and self.cost_gbp + gbp < self.max_cost_gbp
            and self.elapsed_s < self.timeout_s
        )

    def charge(self, cost_gbp: float, steps: int = 1) -> None:
        self.cost_gbp += cost_gbp
        self.steps += steps
