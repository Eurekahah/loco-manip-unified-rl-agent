"""Pure helpers for resolving collection episode outcomes."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np


def resolve_episode_outcome(
    last_outcome: dict,
    manual_outcome: bool | None,
    termination_names: Sequence[str],
) -> tuple[bool, str, bool, list[str]]:
    """Resolve success, label, automatic success, and fired termination terms."""
    flags = np.asarray(
        last_outcome.get(
            "termination_flags",
            np.zeros(len(termination_names), dtype=bool),
        ),
        dtype=bool,
    )
    fired = [
        name for name, value in zip(termination_names, flags) if bool(value)
    ]
    auto_success = bool(last_outcome.get("success", False))
    success = auto_success if manual_outcome is None else bool(manual_outcome)
    if manual_outcome is True:
        label = "manual_success"
    elif manual_outcome is False:
        label = "manual_failure"
    elif fired:
        label = "|".join(fired)
    else:
        label = "manual_reset"
    return success, label, auto_success, fired
