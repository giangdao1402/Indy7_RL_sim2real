import os
import numpy as np
from isaacsim.core.prims import Articulation, XFormPrim
from isaacsim.core.utils.stage import add_reference_to_stage
from isaacsim.core.utils.extensions import get_extension_path_from_name
from isaacsim.core.utils.numpy.rotations import euler_angles_to_quats
from isaacsim.robot_motion.motion_generation import LulaKinematicsSolver, ArticulationKinematicsSolver

class Indy7KinematicsExample():
    def __init__(self):
        self._kinematics_solver = None
        self._articulation_kinematics_solver = None
        self._articulation = None
        self._target = None

    def load_example_assets(self):
        # Thêm robot Indy7 và target
        robot_prim_path = "/Indy7"
        path_to_robot_usd = "/home/apicoo-ai/env_isaaclab/nrmk_isaaclab_public/isaac_neuromeka/assets/model/usd/indy7/indy7.usd"  # thay bằng đường dẫn USD của bạn
        add_reference_to_stage(path_to_robot_usd, robot_prim_path)
        self._articulation = Articulation(robot_prim_path)

        # Thêm target
        add_reference_to_stage("/path/to/frame_prim.usd", "/World/target")
        self._target = XFormPrim("/World/target", scale=[0.04, 0.04, 0.04])
        self._target.set_default_state(np.array([0.5, 0, 0.5]), euler_angles_to_quats([0, 0, 0]))

        return self._articulation, self._target

    def setup(self):
        # Load URDF + YAML cho Indy7
        mg_extension_path = get_extension_path_from_name("isaacsim.robot_motion.motion_generation")
        kinematics_config_dir = os.path.join(mg_extension_path, "motion_policy_configs")

        self._kinematics_solver = LulaKinematicsSolver(
            robot_description_path=os.path.join(kinematics_config_dir, "/home/apicoo-ai/giang_ws/indy_lula.yaml"),
            urdf_path=os.path.join(kinematics_config_dir, "/home/apicoo-ai/env_isaaclab/nrmk_isaaclab_public/isaac_neuromeka/assets/model/urdf/indy7.urdf")
        )

        print("Valid frame names:", self._kinematics_solver.get_all_frame_names())

        end_effector_name = "tcp"  # tên frame end-effector của Indy7 trong URDF
        self._articulation_kinematics_solver = ArticulationKinematicsSolver(
            self._articulation, self._kinematics_solver, end_effector_name
        )

    def update(self, step: float):
        target_position, target_orientation = self._target.get_world_pose()

        # Track robot base
        robot_base_translation, robot_base_orientation = self._articulation.get_world_pose()
        self._kinematics_solver.set_robot_base_pose(robot_base_translation, robot_base_orientation)

        action, success = self._articulation_kinematics_solver.compute_inverse_kinematics(
            target_position, target_orientation
        )

        if success:
            self._articulation.apply_action(action)
        else:
            print("IK did not converge")

    def reset(self):
        pass
