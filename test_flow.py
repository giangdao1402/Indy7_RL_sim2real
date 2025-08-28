from omni.isaac.core.prims import Articulation

# Replace with your robot prim path
robot = Articulation("/World/indy7")

robot.initialize()   # must be initialized after simulation is playing

print("DOF Names:", robot.dof_names)
print("Num DOFs:", robot.num_dof)
print("Default DOF Positions:", robot.get_default_dof_state().positions)