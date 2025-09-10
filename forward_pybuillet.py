import pybullet as p
import pybullet_data
import math
p.connect(p.DIRECT) 
p.setAdditionalSearchPath(pybullet_data.getDataPath())

# Load robot URDF
robot_id = p.loadURDF(
    "/home/apicoo-ai/env_isaaclab/nrmk_isaaclab_public/isaac_neuromeka/assets/model/urdf/indy7.urdf",
    useFixedBase=True
)

# joint in degrees
current_joint_pos_degree = [-23.78, -45.00,  -76.04, 64.41, 15.68,  9.64]
current_joint_pos_radians = math.radians(current_joint_pos_degree[0]), math.radians(current_joint_pos_degree[1]), math.radians(current_joint_pos_degree[2]), math.radians(current_joint_pos_degree[3]), math.radians(current_joint_pos_degree[4]), math.radians(current_joint_pos_degree[5])

for i, q in enumerate(current_joint_pos_radians):
    p.resetJointState(robot_id, i, q)


num_joints = p.getNumJoints(robot_id)
print("Joint states:")
for i in range(num_joints):
    joint_name = p.getJointInfo(robot_id, i)[1].decode('utf-8')
    joint_pos = p.getJointState(robot_id, i)[0]
    print(f"{joint_name}: {joint_pos}")

end_effector_index = 6

ee_state = p.getLinkState(robot_id, end_effector_index)
ee_position = ee_state[0]        # xyz
ee_orientation_quat = ee_state[1]  # quaternion

ee_orientation_euler = p.getEulerFromQuaternion(ee_orientation_quat)

print("\nEnd-Effector pose (FK from default joints):")
print(f"Position: {ee_position}")
print(f"Orientation (Euler angles): {ee_orientation_euler}")
print(f"Orientation (quaternion): {ee_orientation_quat}")

