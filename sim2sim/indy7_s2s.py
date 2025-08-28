from typing import Optional

import numpy as np
import omni
import omni.kit.commands
from isaacsim.core.utils.rotations import quat_to_rot_matrix
from isaacsim.core.utils.types import ArticulationAction
from isaacsim.robot.policy.examples.controllers import PolicyController
from pathlib import Path
import omni.client

class Indy7ReachTask(PolicyController):
    """The Indy7 running Reach Task Policy"""

    def __init__(
        self,
        prim_path: str,
        root_path: Optional[str] = None,
        name: str = "Indy7",
        usd_path: Optional[str] = None,
        position: Optional[np.ndarray] = None,
        orientation: Optional[np.ndarray] = None,
    ) -> None:
        """
        Initialize H1 robot and import flat terrain policy.

        Args:
            prim_path (str) -- prim path of the robot on the stage
            root_path (Optional[str]): The path to the articulation root of the robot
            name (str) -- name of the quadruped
            usd_path (str) -- robot usd filepath in the directory
            position (np.ndarray) -- position of the robot
            orientation (np.ndarray) -- orientation of the robot

        """

        super().__init__(name, prim_path, root_path, usd_path, position, orientation)
        # Load model & env config
        repo_root = Path(__file__).resolve().parents[1]
        model_dir = repo_root / "pretrained_models" / "reach"
        self.load_policy(
            str(model_dir / "policy_default.pt"),
            str(model_dir / "env_default.yaml"),
        )
        self._action_scale = 0.2
        self._policy_counter = 00

    def _compute_observation(self, command):
        """
        Compute the observation vector for the policy.

        Argument:
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        Returns:
        np.ndarray -- The observation vector.

        """
        obs = np.zeros(19)
        # Joint states
        current_joint_pos = self.robot.get_joint_positions()
        print("current_joint_pos: ", current_joint_pos)
        current_joint_vel = self.robot.get_joint_velocities()
        print("current_joint_vel: ", current_joint_vel)
        print("default_pos: ", self.default_pos)
        obs[0:6] = current_joint_pos - self.default_pos
        print("current_joint_pos - default_pos: ", obs[0:6])
        obs[6:12] = current_joint_vel
        # Previous Action
        obs[12:18] = command
        return obs

    def forward(self, dt, command):
        """
        Compute the desired articulation action and apply them to the robot articulation.

        Argument:
        dt (float) -- Timestep update in the world.
        command (np.ndarray) -- the robot command (v_x, v_y, w_z)

        """
        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            self.action = self._compute_action(obs)

        action = ArticulationAction(joint_positions=self.default_pos + (self.action * self._action_scale))
        self.robot.apply_action(action)

        self._policy_counter += 1

    def initialize(self):
        result = super().initialize(set_articulation_props=False)
        print("DOF names:", self.robot.dof_names)
        print("Num DOF:", self.robot.num_dof)
        print("Default DOF positions:", self.default_pos)
        return result