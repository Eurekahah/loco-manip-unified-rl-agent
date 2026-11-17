"""Termination helpers for the generic collection loop."""

from __future__ import annotations


def validate_success_termination_term(termination_manager, success_term_name: str) -> None:
    """Require a configured success term to exist and be non-timeout."""
    if not success_term_name:
        return
    active_terms = tuple(termination_manager.active_terms)
    if success_term_name not in active_terms:
        raise ValueError(
            f"Collection success term '{success_term_name}' is not an active termination term. "
            f"Available terms: {active_terms}"
        )
    term_cfg = termination_manager.get_term_cfg(success_term_name)
    if bool(term_cfg.time_out):
        raise ValueError(
            f"Collection success term '{success_term_name}' must not be a timeout termination."
        )


def compute_fast_termination_status(termination_manager):
    """Evaluate termination terms and snapshot the reset status for Fast stepping."""
    reset = termination_manager.compute().clone()
    terminated = termination_manager.terminated.clone()
    truncated = termination_manager.time_outs.clone()
    return reset, terminated, truncated
