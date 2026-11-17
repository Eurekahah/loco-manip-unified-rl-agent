from __future__ import annotations

from types import SimpleNamespace

from scripts.tools.teleop_cli import prefer_cpu_for_single_env_teleop


def test_single_env_teleop_defaults_to_cpu() -> None:
    args = SimpleNamespace(num_envs=1, device="cuda:0")

    changed = prefer_cpu_for_single_env_teleop(args, argv=[])

    assert changed
    assert args.device == "cpu"


def test_explicit_device_overrides_fast_default() -> None:
    args = SimpleNamespace(num_envs=1, device="cuda:0")

    changed = prefer_cpu_for_single_env_teleop(
        args,
        argv=["--device", "cuda:0"],
    )

    assert not changed
    assert args.device == "cuda:0"


def test_multi_env_keeps_device() -> None:
    multi_env = SimpleNamespace(num_envs=2, device="cuda:0")

    assert not prefer_cpu_for_single_env_teleop(multi_env, argv=[])
    assert multi_env.device == "cuda:0"
