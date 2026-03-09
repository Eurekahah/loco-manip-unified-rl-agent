
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from rl_training.tasks.manager_based.locomotion.highlevel.high_level_env_cfg import HighLevelEnvCfg
from rl_training.assets.deeprobotics import DEEPROBOTICS_M20_PIPER_CFG


@configclass
class HighLevelRoughEnvCfg(HighLevelEnvCfg):

    camera_link_name: str = "piper_camera/camera_link/arm_camera"
    base_link_name: str = "base_link"

    def __post_init__(self):
        super().__post_init__()
        
        self.scene.robot = DEEPROBOTICS_M20_PIPER_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        self.scene.arm_camera.prim_path = "{ENV_REGEX_NS}/Robot/" + self.camera_link_name
        self.scene.lidar.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        
        self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name
        self.scene.height_scanner_base.prim_path = "{ENV_REGEX_NS}/Robot/" + self.base_link_name

        
