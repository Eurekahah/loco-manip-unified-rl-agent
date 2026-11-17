"""Small, dependency-free runtime profiler for teleoperation and collection."""

from __future__ import annotations

import json
import time
from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any


class RuntimeProfiler:
    """Collect rolling timing samples and write compact JSON snapshots."""

    def __init__(self, output_path: str | None = None, *, interval_s: float = 5.0) -> None:
        self.output_path = Path(output_path).expanduser() if output_path else None
        self.interval_s = float(interval_s)
        self._samples: dict[str, deque[float]] = defaultdict(lambda: deque(maxlen=2048))
        self._counters: dict[str, float] = {}
        self._started_at = time.perf_counter()
        self._last_write = self._started_at
        self._last_loop = self._started_at
        self._frames = 0
        self._fps = 0.0

    @contextmanager
    def measure(self, name: str) -> Iterator[None]:
        start = time.perf_counter()
        try:
            yield
        finally:
            self.record(name, (time.perf_counter() - start) * 1000.0)

    def record(self, name: str, milliseconds: float) -> None:
        self._samples[name].append(max(float(milliseconds), 0.0))

    def set_counter(self, name: str, value: float) -> None:
        self._counters[name] = float(value)

    def increment(self, name: str, amount: float = 1.0) -> None:
        self._counters[name] = self._counters.get(name, 0.0) + float(amount)

    def tick(self) -> None:
        now = time.perf_counter()
        self._frames += 1
        elapsed = now - self._last_loop
        self.record("loop_ms", elapsed * 1000.0)
        self._last_loop = now
        if elapsed > 0.0:
            self._fps = 0.9 * self._fps + 0.1 * (1.0 / elapsed) if self._fps else 1.0 / elapsed
        if self.output_path is not None and now - self._last_write >= self.interval_s:
            self.write()
            self._last_write = now

    def reset_clock(self) -> None:
        """Exclude environment startup time from the first loop timing sample."""
        now = time.perf_counter()
        self._started_at = now
        self._last_write = now
        self._last_loop = now
        self._frames = 0
        self._fps = 0.0

    @staticmethod
    def _percentile(values: list[float], fraction: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        index = min(max(int(round((len(ordered) - 1) * fraction)), 0), len(ordered) - 1)
        return ordered[index]

    def snapshot(self) -> dict[str, Any]:
        timings = {}
        for name, samples in self._samples.items():
            values = list(samples)
            timings[name] = {
                "count": len(values),
                "mean_ms": sum(values) / len(values) if values else 0.0,
                "p95_ms": self._percentile(values, 0.95),
                "max_ms": max(values, default=0.0),
            }
        return {
            "elapsed_s": time.perf_counter() - self._started_at,
            "frames": self._frames,
            "fps": self._fps,
            "timings": timings,
            "counters": dict(self._counters),
        }

    def write(self, *, force: bool = False) -> None:
        if self.output_path is None:
            return
        self.output_path.parent.mkdir(parents=True, exist_ok=True)
        temporary_path = self.output_path.with_suffix(self.output_path.suffix + ".tmp")
        temporary_path.write_text(
            json.dumps(self.snapshot(), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        temporary_path.replace(self.output_path)

    def print_summary(self) -> None:
        snapshot = self.snapshot()
        print(
            f"[Profile] frames={snapshot['frames']} fps={snapshot['fps']:.2f} "
            f"loop_ms={snapshot['timings'].get('loop_ms', {}).get('mean_ms', 0.0):.2f}"
        )
        for name, stats in snapshot["timings"].items():
            if name == "loop_ms":
                continue
            print(
                f"[Profile] {name}: mean={stats['mean_ms']:.2f} ms "
                f"p95={stats['p95_ms']:.2f} ms max={stats['max_ms']:.2f} ms"
            )
        if snapshot["counters"]:
            print(f"[Profile] counters={snapshot['counters']}")
