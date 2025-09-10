import time
import numpy as np
from pathlib import Path
from neuromeka import IndyDCP3
from controllers.policy_controller import PolicyController
class IndyReachPolicyNoHistory(PolicyController):
    """Policy controller cho Indy Reach KHÔNG dùng history."""

    def __init__(self, robot_ip) -> None:
        super().__init__()

        # Robot
        self.robot = IndyDCP3(robot_ip)

        # DOF names
        self.dof_names = ["joint0", "joint1", "joint2", "joint3", "joint4", "joint5"]
        self.num_joints = len(self.dof_names)

        # Load model & env config
        repo_root = Path(__file__).resolve().parents[1]
        model_dir = repo_root / "pretrained_models" / "reach"
        self.load_policy(
            model_dir / "policy_default.pt",
            model_dir / "env_default.yaml",
        )

        # Parameters
        self._action_scale = 0.2
        self._policy_counter = 0

        # Command pose (xyz + quaternion)
        self.target_command = np.array([0.350, -0.18649, 0.52197, 0, 0, 1, 0], dtype=np.float32)


        # Joint data
        self.has_joint_data = False
        self.current_joint_positions = np.zeros(self.num_joints, dtype=np.float32)
        self.current_joint_velocities = np.zeros(self.num_joints, dtype=np.float32)

    def update_joint_state(self) -> None:
        """Update the current joint state from robot."""
        control_data = self.robot.get_control_data()
        pos_deg = control_data["q"]
        vel_deg = control_data["qdot"]

        pos_rad = np.radians(pos_deg)
        vel_rad = np.radians(vel_deg)

        self.current_joint_positions = pos_rad
        self.current_joint_velocities = vel_rad
        self.has_joint_data = True

    def _compute_observation(self, command: np.ndarray) -> np.ndarray:
        """Build observation vector KHÔNG dùng history."""
        if not self.has_joint_data:
            return None
        print("current_joint_pos: ", self.current_joint_positions)
        joint_pos = self.current_joint_positions - self.default_pos
        print("[current_joint_pos - default_pos]: ", joint_pos)
        joint_vel = self.current_joint_velocities
        print("current_joint_vel: ", joint_vel)
        pose_command = command.astype(np.float32)
        print("command: ", pose_command)
        obs = np.concatenate([
            joint_pos,     # 6
            joint_vel,     # 6
            pose_command,  # 7
        ]).astype(np.float32)

        return obs

    def forward(self, dt: float, command: np.ndarray) -> np.ndarray:
        """Compute the next joint positions based on the policy."""
        if not self.has_joint_data:
            return None

        if self._policy_counter % self._decimation == 0:
            obs = self._compute_observation(command)
            if obs is None:
                return None

            # Lấy action từ policy controller
            self.action = self._compute_action(obs)

            # Debug print
            print("\n=== Policy Step ===")
            print(f"{'Command:':<20} {np.round(command, 4)}")
            print("--- Observation ---")
            print(f"{'Δ Joint Positions:':<20} {np.round(obs[:self.num_joints], 4)}")
            print(f"{'Joint Velocities:':<20} {np.round(obs[self.num_joints:2*self.num_joints], 4)}")
            print(f"{'Command:':<20} {np.round(obs[2*self.num_joints:], 4)}")
            print("--- Action ---")
            print(f"{'Raw Action:':<20} {np.round(self.action, 4)}")
            print(f"{'Processed Action:':<20} {np.round(self.default_pos + (self.action * self._action_scale), 4)}")

        joint_positions = self.default_pos + (self.action * self._action_scale)
        self._policy_counter += 1
        return joint_positions


if __name__ == "__main__":
    ROBOT_IP = "192.168.0.102"
    policy = IndyReachPolicyNoHistory(ROBOT_IP)

    # Move to home
    JOINT_HOME_DEG = [0, 0, -90, 0, -90, 0]
    policy.robot.movej(JOINT_HOME_DEG, vel_ratio=20, acc_ratio=100)
    while policy.robot.get_motion_data()["has_motion"]:
        time.sleep(0.01)

    dt = 0.02
    while True:
        policy.update_joint_state()
        target_pos_rad = policy.forward(dt, policy.target_command)
        if target_pos_rad is None:
            continue
        target_pos_deg = np.degrees(target_pos_rad).tolist()
        
        print("target position in degree: ", target_pos_deg)
        # # Gửi lệnh tới robot nếu muốn
        policy.robot.movej(target_pos_deg, vel_ratio=5, acc_ratio=5)
        while policy.robot.get_motion_data()["has_motion"]:
            time.sleep(0.01)

        time.sleep(dt)
