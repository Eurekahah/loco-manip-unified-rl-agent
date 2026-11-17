"""Compatibility aliases for the shared HDF5 recorder.

New task configurations should import ``HDF5RecorderManagerCfg`` directly from
``rl_training.recorders`` and declare their own recorder fields.
"""

from rl_training.recorders.hdf5 import (
    HDF5Recorder,
    HDF5RecorderCfg,
    HDF5RecorderManagerCfg,
)


LowShelfHDF5Recorder = HDF5Recorder
LowShelfHDF5RecorderCfg = HDF5RecorderCfg
LowShelfHDF5RecorderManagerCfg = HDF5RecorderManagerCfg

__all__ = [
    "HDF5Recorder",
    "HDF5RecorderCfg",
    "HDF5RecorderManagerCfg",
    "LowShelfHDF5Recorder",
    "LowShelfHDF5RecorderCfg",
    "LowShelfHDF5RecorderManagerCfg",
]
