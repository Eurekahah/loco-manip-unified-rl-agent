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
from rl_training.tasks.manager_based.locomotion.highlevel.config.rear_pick.scene_cfg import (
    make_rear_pick_front_camera,
    make_rear_pick_top_camera,
    make_rear_pick_workcell_camera,
    make_rear_pick_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.peek_over.scene_cfg import (
    make_peek_over_front_camera,
    make_peek_over_top_camera,
    make_peek_over_workcell_camera,
    make_peek_over_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.blind_intercept.scene_cfg import (
    make_blind_intercept_front_camera,
    make_blind_intercept_top_camera,
    make_blind_intercept_workcell_camera,
    make_blind_intercept_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.drawer_ladder.scene_cfg import (
    make_drawer_ladder_front_camera,
    make_drawer_ladder_top_camera,
    make_drawer_ladder_workcell_camera,
    make_drawer_ladder_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.door_brace.scene_cfg import (
    make_door_brace_front_camera,
    make_door_brace_top_camera,
    make_door_brace_workcell_camera,
    make_door_brace_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.high_low_serve.scene_cfg import (
    make_high_low_front_camera,
    make_high_low_top_camera,
    make_high_low_workcell_camera,
    make_high_low_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.home_mission.scene_cfg import (
    make_home_mission_front_camera,
    make_home_mission_top_camera,
    make_home_mission_workcell_camera,
    make_home_mission_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.front_desk_pickup.scene_cfg import (
    make_front_desk_pickup_front_camera,
    make_front_desk_pickup_top_camera,
    make_front_desk_pickup_workcell_camera,
    make_front_desk_pickup_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.ward_delivery.scene_cfg import (
    make_ward_delivery_front_camera,
    make_ward_delivery_top_camera,
    make_ward_delivery_workcell_camera,
    make_ward_delivery_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.kitchen_drawer_serve.scene_cfg import (
    make_kitchen_drawer_serve_front_camera,
    make_kitchen_drawer_serve_top_camera,
    make_kitchen_drawer_serve_workcell_camera,
    make_kitchen_drawer_serve_wrist_camera,
)
from rl_training.tasks.manager_based.locomotion.highlevel.config.supply_restock.scene_cfg import (
    make_supply_restock_front_camera,
    make_supply_restock_top_camera,
    make_supply_restock_workcell_camera,
    make_supply_restock_wrist_camera,
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


register_collection_profile(
    CollectionTaskProfile(
        key="rear_pick",
        display_name="P1 RearPick",
        task_ids=("Isaac-M20-Piper-RearPick-Collect-v0",),
        output_dir="rear_pick",
        action_term_name="pre_trained_pick_action",
        camera_factories={
            "top_camera": make_rear_pick_top_camera,
            "wrist_camera": make_rear_pick_wrist_camera,
            "workcell_camera": make_rear_pick_workcell_camera,
            "front_camera": make_rear_pick_front_camera,
        },
        # The rear table and tray are kinematic scene objects. Fast stepping
        # must not try to write linear/angular velocity into them.
        fast_robot_only_write=True,
    )
)


register_collection_profile(
    CollectionTaskProfile(
        key="peek_over",
        display_name="P3 PeekOver",
        task_ids=("Isaac-M20-Piper-PeekOver-Collect-v0",),
        output_dir="peek_over",
        action_term_name="pre_trained_pick_action",
        camera_factories={
            "top_camera": make_peek_over_top_camera,
            "wrist_camera": make_peek_over_wrist_camera,
            "workcell_camera": make_peek_over_workcell_camera,
            "front_camera": make_peek_over_front_camera,
        },
        fast_robot_only_write=True,
        fisheye_mount_pos=(-0.44, 0.0, 0.58),
    )
)


register_collection_profile(
    CollectionTaskProfile(
        key="blind_intercept",
        display_name="P2 BlindIntercept",
        task_ids=("Isaac-M20-Piper-BlindIntercept-Collect-v0",),
        output_dir="blind_intercept",
        action_term_name="pre_trained_pick_action",
        camera_factories={
            "top_camera": make_blind_intercept_top_camera,
            "wrist_camera": make_blind_intercept_wrist_camera,
            "workcell_camera": make_blind_intercept_workcell_camera,
            "front_camera": make_blind_intercept_front_camera,
        },
        fast_robot_only_write=True,
    )
)


register_collection_profile(
    CollectionTaskProfile(
        key="drawer_ladder",
        display_name="W1 DrawerLadder",
        task_ids=("Isaac-M20-Piper-DrawerLadder-Collect-v0",),
        output_dir="drawer_ladder",
        action_term_name="pre_trained_pick_action",
        camera_factories={
            "top_camera": make_drawer_ladder_top_camera,
            "wrist_camera": make_drawer_ladder_wrist_camera,
            "workcell_camera": make_drawer_ladder_workcell_camera,
            "front_camera": make_drawer_ladder_front_camera,
        },
        fast_robot_only_write=True,
    )
)


register_collection_profile(
    CollectionTaskProfile(
        key="door_brace",
        display_name="W2 DoorBrace",
        task_ids=("Isaac-M20-Piper-DoorBrace-Collect-v0",),
        output_dir="door_brace",
        action_term_name="pre_trained_pick_action",
        camera_factories={
            "top_camera": make_door_brace_top_camera,
            "wrist_camera": make_door_brace_wrist_camera,
            "workcell_camera": make_door_brace_workcell_camera,
            "front_camera": make_door_brace_front_camera,
        },
        fast_robot_only_write=True,
    )
)


register_collection_profile(
    CollectionTaskProfile(
        key="high_low_serve",
        display_name="W5 HighLowServe",
        task_ids=("Isaac-M20-Piper-HighLowServe-Collect-v0",),
        output_dir="high_low_serve",
        action_term_name="pre_trained_pick_action",
        camera_factories={
            "top_camera": make_high_low_top_camera,
            "wrist_camera": make_high_low_wrist_camera,
            "workcell_camera": make_high_low_workcell_camera,
            "front_camera": make_high_low_front_camera,
        },
        tracked_entity_names=("object", "object_high", "target_pad"),
        fast_robot_only_write=True,
    )
)


register_collection_profile(
    CollectionTaskProfile(
        key="home_mission",
        display_name="L1 HomeMission",
        task_ids=("Isaac-M20-Piper-HomeMission-Collect-v0",),
        output_dir="home_mission",
        action_term_name="pre_trained_pick_action",
        camera_factories={
            "top_camera": make_home_mission_top_camera,
            "wrist_camera": make_home_mission_wrist_camera,
            "workcell_camera": make_home_mission_workcell_camera,
            "front_camera": make_home_mission_front_camera,
        },
        fast_robot_only_write=True,
    )
)


register_collection_profile(
    CollectionTaskProfile(
        key="front_desk_pickup",
        display_name="R1 FrontDeskPickup",
        task_ids=("Isaac-M20-Piper-FrontDeskPickup-Collect-v0",),
        output_dir="front_desk_pickup",
        action_term_name="pre_trained_pick_action",
        camera_factories={
            "top_camera": make_front_desk_pickup_top_camera,
            "wrist_camera": make_front_desk_pickup_wrist_camera,
            "workcell_camera": make_front_desk_pickup_workcell_camera,
            "front_camera": make_front_desk_pickup_front_camera,
        },
        tracked_entity_names=("object", "target_pad", "distractor_cracker", "distractor_can"),
        fast_robot_only_write=True,
        default_front_camera=True,
        default_wrist_preview=True,
        default_fisheye_rig="four",
        fisheye_mount_pos=(-0.44, 0.0, 0.58),
        extra_metadata={
            "theme": "office_reception",
            "pano_causal_condition": "reception desk starts behind the robot",
            "body_arm_requirement": "reception-height transfer to visitor-side low table",
        },
    )
)


register_collection_profile(
    CollectionTaskProfile(
        key="ward_delivery",
        display_name="H1 WardDelivery",
        task_ids=("Isaac-M20-Piper-WardDelivery-Collect-v0",),
        output_dir="ward_delivery",
        action_term_name="pre_trained_pick_action",
        camera_factories={
            "top_camera": make_ward_delivery_top_camera,
            "wrist_camera": make_ward_delivery_wrist_camera,
            "workcell_camera": make_ward_delivery_workcell_camera,
            "front_camera": make_ward_delivery_front_camera,
        },
        tracked_entity_names=("object", "target_pad", "distractor_bottle"),
        fast_robot_only_write=True,
        default_front_camera=True,
        default_wrist_preview=True,
        default_fisheye_rig="four",
        fisheye_mount_pos=(-0.44, 0.0, 0.58),
        extra_metadata={
            "theme": "hospital_ward",
            "pano_causal_condition": "medicine shelf starts outside the front view",
            "body_arm_requirement": "high shelf to bedside table with wheelchair aisle",
        },
    )
)


register_collection_profile(
    CollectionTaskProfile(
        key="kitchen_drawer_serve",
        display_name="K1 KitchenDrawerServe",
        task_ids=("Isaac-M20-Piper-KitchenDrawerServe-Collect-v0",),
        output_dir="kitchen_drawer_serve",
        action_term_name="pre_trained_pick_action",
        camera_factories={
            "top_camera": make_kitchen_drawer_serve_top_camera,
            "wrist_camera": make_kitchen_drawer_serve_wrist_camera,
            "workcell_camera": make_kitchen_drawer_serve_workcell_camera,
            "front_camera": make_kitchen_drawer_serve_front_camera,
        },
        tracked_entity_names=("object", "target_pad", "distractor_box"),
        fast_robot_only_write=True,
        default_front_camera=True,
        default_wrist_preview=True,
        default_fisheye_rig="four",
        fisheye_mount_pos=(-0.44, 0.0, 0.58),
        extra_metadata={
            "theme": "kitchen_room",
            "pano_causal_condition": "drawer, counter and upper-door constraint share one 360-degree scene",
            "body_arm_requirement": "low drawer plus partially open overhead cabinet",
        },
    )
)


register_collection_profile(
    CollectionTaskProfile(
        key="supply_restock",
        display_name="H2 SupplyRestock",
        task_ids=("Isaac-M20-Piper-SupplyRestock-Collect-v0",),
        output_dir="supply_restock",
        action_term_name="pre_trained_pick_action",
        camera_factories={
            "top_camera": make_supply_restock_top_camera,
            "wrist_camera": make_supply_restock_wrist_camera,
            "workcell_camera": make_supply_restock_workcell_camera,
            "front_camera": make_supply_restock_front_camera,
        },
        tracked_entity_names=("object", "target_pad", "distractor_bottle"),
        fast_robot_only_write=True,
        default_front_camera=True,
        default_wrist_preview=True,
        default_fisheye_rig="four",
        fisheye_mount_pos=(-0.44, 0.0, 0.58),
        extra_metadata={
            "theme": "hospital_supply_room",
            "pano_causal_condition": "high cabinet starts outside the front view",
            "body_arm_requirement": "1.4 m shelf to floor tray through bed, wheelchair and cart",
            "regrasp_policy": "one recovery attempt is permitted by the task protocol",
        },
    )
)
