import pybullet as p
import pybullet_data

# Kết nối PyBullet
p.connect(p.DIRECT)  # Dùng GUI nếu muốn xem robot: p.GUI
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load robot URDF
robot_id = p.loadURDF(
    "/home/apicoo-ai/env_isaaclab/nrmk_isaaclab_public/isaac_neuromeka/assets/model/urdf/indy7.urdf",
    useFixedBase=True
)

# Default joint positions của robot (ví dụ 7 joints)
# default_joint_positions = [0, -0.649, -2.064, 0, 1.11199, 2.356194490192345, 0.0]

# # Reset các joint về default position
# for i, pos in enumerate(default_joint_positions):
#     p.resetJointState(robot_id, i, pos)

current_joint_pos = [ 0.02183251, -0.86905867,  0.230097,    0.24777441, -0.27259284,  0.49371484]

# set joint states tạm thời
for i, q in enumerate(current_joint_pos):
    p.resetJointState(robot_id, i, q)


# Lấy số joint
num_joints = p.getNumJoints(robot_id)
print("Joint states:")
for i in range(num_joints):
    joint_name = p.getJointInfo(robot_id, i)[1].decode('utf-8')
    joint_pos = p.getJointState(robot_id, i)[0]
    print(f"{joint_name}: {joint_pos}")

# Chỉ số link của end-effector
end_effector_index = 6  # thay đổi nếu link khác

ee_state = p.getLinkState(robot_id, end_effector_index)
ee_position = ee_state[0]        # xyz
ee_orientation_quat = ee_state[1]  # quaternion

# Chuyển quaternion sang Euler angles
ee_orientation_euler = p.getEulerFromQuaternion(ee_orientation_quat)

print("\nEnd-Effector pose (FK from default joints):")
print(f"Position: {ee_position}")
print(f"Orientation (Euler angles): {ee_orientation_euler}")
print(f"Orientation (quaternion): {ee_orientation_quat}")


# Giờ pose này có thể gửi cho policy
