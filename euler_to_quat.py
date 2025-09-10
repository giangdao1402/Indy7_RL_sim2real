import torch

def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    """Convert rotations given as Euler angles in radians to Quaternions.
       Convention: XYZ, return format (w, x, y, z)
    """
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)

    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp

    return torch.stack([qw, qx, qy, qz], dim=-1)

# ---------------- Test ----------------
# pose: [x, y, z, roll, pitch, yaw]
pose = [0.40012,
        -0.40291,
         0.34319,
         3.14,  # roll
         0,                 # pitch
         3.14]   # yaw

x, y, z, roll, pitch, yaw = pose

# chuyển Euler -> Quaternion
roll_t  = torch.tensor([roll])
pitch_t = torch.tensor([pitch])
yaw_t   = torch.tensor([yaw])

quat = quat_from_euler_xyz(roll_t, pitch_t, yaw_t)

print("Position:", (x, y, z))
print("Quaternion (w, x, y, z):", quat.numpy())
