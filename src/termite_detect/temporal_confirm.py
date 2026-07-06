from __future__ import annotations

"""Temporal confirmation gate for the patrol pipeline.

A robot dog sees a *video stream*, not isolated photos. A single-frame false
positive should NOT raise an alert; a real termite (or termite sign) persists
across consecutive frames. This gate only fires an alert when a spatially
consistent detection is seen in >= K of the last N frames — turning a noisy
per-frame detector into a low-false-alarm alerting system.

If a per-frame false positive appears with probability p (roughly independent
frame-to-frame), a K-of-N gate cuts the false-ALERT rate to ~ P(Binom(N,p) >= K),
e.g. p=0.15, K=3, N=5 -> ~1.3% — well under the 5% target — while a true
detection (high per-frame hit rate) still passes easily.
"""

from collections import deque
from dataclasses import dataclass, field


def _center(b):
    return ((b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0)


def _close(c1, c2, tol):
    return abs(c1[0] - c2[0]) <= tol and abs(c1[1] - c2[1]) <= tol


@dataclass
class TemporalConfirm:
    k: int = 3                 # need a hit in >= k of the last n frames
    n: int = 5
    dist_tol: float = 60.0     # px: how far a detection may drift and still count as "the same"
    history: deque = field(default_factory=lambda: deque())

    def update(self, dets):
        """dets: list of (box_xyxy, score) for the current frame.
        Returns the list of CONFIRMED boxes (persisted across k-of-n frames)."""
        centers = [_center(b) for b, _ in dets]
        self.history.append(centers)
        while len(self.history) > self.n:
            self.history.popleft()
        confirmed = []
        for (b, s), c in zip(dets, centers):
            hits = sum(1 for frame in self.history if any(_close(c, pc, self.dist_tol) for pc in frame))
            if hits >= self.k:
                confirmed.append((b, s))
        return confirmed


def _demo():
    """Monte-Carlo: false-alert rate under a K-of-N gate for a few per-frame FP rates."""
    import random
    rng = random.Random(0)
    for p in (0.10, 0.15, 0.25):
        for k, n in ((2, 3), (3, 5), (4, 7)):
            alerts = 0
            trials = 20000
            for _ in range(trials):
                g = TemporalConfirm(k=k, n=n)
                fired = False
                for _t in range(n):
                    # a spurious FP jumps around: new random center each frame -> rarely persists
                    dets = [([x := rng.uniform(0, 1900), y := rng.uniform(0, 1000), x + 30, y + 30], 0.5)] if rng.random() < p else []
                    if g.update(dets):
                        fired = True
                alerts += fired
            print(f"p={p:.2f}  K{k}-of-N{n}: false-alert {alerts/trials:6.2%}")
    print("(true detections persist in ~the same place -> pass the gate reliably)")


if __name__ == "__main__":
    _demo()
