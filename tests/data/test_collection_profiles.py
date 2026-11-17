from __future__ import annotations

import sys
from pathlib import Path


DATA_DIR = Path(__file__).resolve().parents[2] / "scripts" / "data"
sys.path.insert(0, str(DATA_DIR))

from collection.profiles import (  # noqa: E402
    CollectionTaskProfile,
    register_collection_profile,
    resolve_profile,
)
from collection.cli import build_collection_parser  # noqa: E402


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


def test_collection_ik_update_period_defaults_to_100_hz() -> None:
    parser = build_collection_parser(
        description="test",
        default_task="Isaac-M20-Piper-UnitTest-Collect-v0",
    )

    args = parser.parse_args([])

    assert args.ik_update_period == 0.01
