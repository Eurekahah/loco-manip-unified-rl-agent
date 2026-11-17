from __future__ import annotations

from types import SimpleNamespace

from scripts.tools.teleop_cli import prefer_cpu_for_single_env_fast_path


def test_fast_single_env_defaults_to_cpu() -> None:
    args = SimpleNamespace(num_envs=1, device="cuda:0")

    changed = prefer_cpu_for_single_env_fast_path(args, enabled=True, argv=[])

    assert changed
    assert args.device == "cpu"


def test_explicit_device_overrides_fast_default() -> None:
    args = SimpleNamespace(num_envs=1, device="cuda:0")

    changed = prefer_cpu_for_single_env_fast_path(
        args,
        enabled=True,
        argv=["--device", "cuda:0"],
    )

    assert not changed
    assert args.device == "cuda:0"


def test_non_fast_or_multi_env_keeps_device() -> None:
    fast_multi_env = SimpleNamespace(num_envs=2, device="cuda:0")
    normal_single_env = SimpleNamespace(num_envs=1, device="cuda:0")

    assert not prefer_cpu_for_single_env_fast_path(
        fast_multi_env, enabled=True, argv=[]
    )
    assert not prefer_cpu_for_single_env_fast_path(
        normal_single_env, enabled=False, argv=[]
    )
    assert fast_multi_env.device == "cuda:0"
    assert normal_single_env.device == "cuda:0"
