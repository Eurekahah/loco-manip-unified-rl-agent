from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace


DATA_DIR = Path(__file__).resolve().parents[2] / "scripts" / "data"
sys.path.insert(0, str(DATA_DIR))

from collection.profiles import (  # noqa: E402
    CollectionTaskProfile,
    register_collection_profile,
    resolve_profile,
)
from collection.cli import apply_rendering_default, build_collection_parser  # noqa: E402


def test_explicit_collection_profile_is_task_isolated() -> None:
    profile = CollectionTaskProfile(
        key="unit_test_task",
        display_name="Unit test task",
        task_ids=("Isaac-M20-Piper-UnitTest-Collect-v0",),
        output_dir="unit_test",
        action_term_name="pre_trained_pick_action",
        camera_factories={},
        tracked_entity_names=("object", "target_pad"),
    )
    register_collection_profile(profile)

    resolved = resolve_profile(
        "Isaac-M20-Piper-UnitTest-Collect-v0",
        profile_key="unit_test_task",
    )

    assert resolved is profile
    assert resolved.matches_task("Isaac-M20-Piper-UnitTest-Collect-v0")
    assert not resolved.matches_task("Isaac-M20-Piper-Other-Collect-v0")


def test_collection_fast_and_ik_defaults() -> None:
    parser = build_collection_parser(
        description="test",
        default_task="Isaac-M20-Piper-UnitTest-Collect-v0",
    )

    args = parser.parse_args([])

    assert args.fast_stepping
    assert args.ik_update_period == 0.01
    assert args.camera_update_period == 0.0
    assert args.fisheye_update_period == 0.0
    assert args.width == 1280
    assert args.height == 720


def test_collection_rendering_defaults_to_balanced(monkeypatch) -> None:
    monkeypatch.setattr(sys, "argv", ["collect_demonstrations.py"])
    args = SimpleNamespace(
        num_envs=1,
        device="cuda:0",
        fisheye_rig="none",
        rendering_mode="performance",
    )

    apply_rendering_default(args)

    assert args.device == "cpu"
    assert args.rendering_mode == "balanced"
