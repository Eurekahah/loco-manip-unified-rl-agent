"""Registry and task-isolated configuration for demonstration collection."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


CameraFactory = Callable[..., Any]
EnvConfigurator = Callable[[Any], None]
ActionConfigurator = Callable[[Any], None]


@dataclass(frozen=True)
class CollectionTaskProfile:
    """Everything the generic collection loop needs to know about one task."""

    key: str
    display_name: str
    task_ids: tuple[str, ...]
    output_dir: str
    action_term_name: str
    camera_factories: Mapping[str, CameraFactory]
    tracked_entity_names: tuple[str, ...] = ("object", "target_pad")
    primary_object_name: str = "object"
    primary_target_name: str = "target_pad"
    success_term_name: str = "place_success"
    command_dim: int = 13
    keyboard_absolute_slices: tuple[tuple[int, int], ...] = ((3, 6), (6, 9), (9, 12))
    keyboard_limits: tuple[tuple[float, float], ...] = ()
    ee_body_name: str = "gripper_base"
    base_link_name: str = "base_link"
    gripper_joint_names: tuple[str, ...] = ("gripper_joint1", "gripper_joint2")
    instruction_attr: str = "language_instruction"
    recorder_attr_name: str = "hdf5_recorder"
    legacy_recorder_attr_name: str | None = None
    configure_env: EnvConfigurator | None = None
    configure_action_term: ActionConfigurator | None = None
    extra_metadata: Mapping[str, Any] = field(default_factory=dict)

    def matches_task(self, task_id: str) -> bool:
        return task_id in self.task_ids


_PROFILES: dict[str, CollectionTaskProfile] = {}


def register_collection_profile(profile: CollectionTaskProfile) -> None:
    """Register or replace a task profile by key."""
    if not profile.key:
        raise ValueError("CollectionTaskProfile.key must not be empty")
    _PROFILES[profile.key] = profile


def resolve_profile(
    task_id: str,
    *,
    profile_key: str = "auto",
    profile_modules: Sequence[str] = (),
) -> CollectionTaskProfile:
    """Resolve a task profile, importing optional extension modules first."""
    import importlib

    for module_name in profile_modules:
        importlib.import_module(module_name)

    if profile_key != "auto":
        if profile_key in _PROFILES:
            return _PROFILES[profile_key]
        from . import builtin_profiles  # noqa: F401

        if profile_key in _PROFILES:
            return _PROFILES[profile_key]
        raise ValueError(
            f"Unknown collection profile '{profile_key}'. "
            f"Available profiles: {sorted(_PROFILES)}"
        )

    from . import builtin_profiles  # noqa: F401
    matches = [profile for profile in _PROFILES.values() if profile.matches_task(task_id)]
    if not matches:
        raise ValueError(
            f"No collection profile matches task '{task_id}'. "
            f"Register one with register_collection_profile() and pass "
            f"--profile-module, or select an available profile: {sorted(_PROFILES)}"
        )
    if len(matches) > 1:
        raise ValueError(
            f"Task '{task_id}' matches multiple profiles: {[profile.key for profile in matches]}. "
            "Pass --profile explicitly."
        )
    return matches[0]
