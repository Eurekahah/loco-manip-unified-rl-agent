# Copyright (c) 2025 Deep Robotics
# SPDX-License-Identifier: BSD 3-Clause
# 
# # Copyright (c) 2024-2025 Ziqi Fan
# SPDX-License-Identifier: Apache-2.0

"""Installation script for the 'rl_training' python package."""

import os
import toml

from setuptools import find_packages, setup

# Obtain the extension data from the extension.toml file
EXTENSION_PATH = os.path.dirname(os.path.realpath(__file__))
# Read the extension.toml file
EXTENSION_TOML_DATA = toml.load(os.path.join(EXTENSION_PATH, "config", "extension.toml"))

# Minimum dependencies required prior to installation
INSTALL_REQUIRES = [
    # base
    "psutil",
    "colorama",
    "xacrodoc",
    # amp
    "numpy",
    "pandas",
    "pinocchio",
]

# Installation operation
setup(
    name="rl_training",
    # 原来只写 ["rl_training"]：子包（rl_training.tasks / .assets / ...）不会被当成包安装，
    # 换机器 `pip install -e source/rl_training` 后只能靠源码目录兜底（editable 能用，
    # 但 wheel/sdist 会缺文件）。这里改成显式发现所有 rl_training.* 子包。
    # （known_issues 工程债，2026-09-30 修；见 docs/review/DEFECT_LOG_zh.md DEF-034）
    packages=find_packages(include=["rl_training", "rl_training.*"]),
    author=EXTENSION_TOML_DATA["package"]["author"],
    maintainer=EXTENSION_TOML_DATA["package"]["maintainer"],
    url=EXTENSION_TOML_DATA["package"]["repository"],
    version=EXTENSION_TOML_DATA["package"]["version"],
    description=EXTENSION_TOML_DATA["package"]["description"],
    keywords=EXTENSION_TOML_DATA["package"]["keywords"],
    install_requires=INSTALL_REQUIRES,
    license="Apache License 2.0",
    include_package_data=True,
    python_requires=">=3.10",
    classifiers=[
        "Natural Language :: English",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Isaac Sim :: 4.5.0",
        "Isaac Sim :: 5.0.0",
    ],
    zip_safe=False,
)
