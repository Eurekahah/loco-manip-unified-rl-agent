"""Built-in collection profiles for tasks shipped in this repository."""

from __future__ import annotations

from rl_training.tasks.manager_based.locomotion.highlevel.config.low_shelf_place.scene_cfg import (
    make_top_camera as make_low_shelf_top_camera,
    make_workcell_camera as make_low_shelf_workcell_camera,
    make_wrist_camera as make_low_shelf_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.two_cylinder_place.scene_cfg import (
    make_two_cylinder_top_camera,
    make_two_cylinder_workcell_camera,
    make_two_cylinder_wrist_camera,
)

from .profiles import CollectionTaskProfile, register_collection_profile


register_collection_profile(
    CollectionTaskProfile(
        key="low_shelf",
        display_name="Low-shelf pick-and-place",
        task_ids=("Isaac-M20-Piper-LowShelf-Collect-v0",),
        output_dir="low_shelf_place",
        action_term_name="pre_trained_pick_action",
        camera_factories={
            "top_camera": make_low_shelf_top_camera,
            "wrist_camera": make_low_shelf_wrist_camera,
            "workcell_camera": make_low_shelf_workcell_camera,
        },
    )
)


register_collection_profile(
    CollectionTaskProfile(
        key="two_cylinder",
        display_name="Two-cylinder pick-and-place",
        task_ids=("Isaac-M20-Piper-TwoCylinder-Collect-v0",),
        output_dir="two_cylinder_place",
        action_term_name="pre_trained_pick_action",
        camera_factories={
            "top_camera": make_two_cylinder_top_camera,
            "wrist_camera": make_two_cylinder_wrist_camera,
            "workcell_camera": make_two_cylinder_workcell_camera,
        },
    )
)
