import numpy as np
import torch
from scipy.spatial.transform import Rotation


def quaternion_geodesic_distance(q1, q2):
    q1 = q1 / torch.norm(q1, dim=-1, keepdim=True)
    q2 = q2 / torch.norm(q2, dim=-1, keepdim=True)
    dot_product = torch.sum(q1 * q2, dim=-1)
    dot_product = torch.abs(dot_product)
    dot_product = torch.clamp(dot_product, 0.0, 1.0)
    angle = 2 * torch.acos(dot_product)
    return angle


def quat_to_euler(quat):
    """Convert quaternion to euler angles (roll, pitch, yaw)

    This implementation handles quaternion to Euler angle conversion while
    minimizing sudden jumps in the Euler angles. It ensures continuity
    in the representation by properly handling edge cases.

    Args:
        quat: Quaternion tensor of shape (..., 4) in wxyz format

    Returns:
        Euler angles tensor of shape (..., 3) for roll, pitch, yaw
    """
    # Normalize quaternion to ensure unit length
    quat = quat / torch.norm(quat, dim=-1, keepdim=True)

    # Extract quaternion components
    w = quat[..., 0]
    x = quat[..., 1]
    y = quat[..., 2]
    z = quat[..., 3]

    # Calculate roll (x-axis rotation)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    euler_x = torch.atan2(sinr_cosp, cosr_cosp)

    # Calculate pitch (y-axis rotation)
    # Clamp the value to avoid numerical issues at the poles
    sinp = 2 * (w * y - z * x)
    sinp = torch.clamp(sinp, -1.0, 1.0)
    euler_y = torch.asin(sinp)

    # Calculate yaw (z-axis rotation)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    euler_z = torch.atan2(siny_cosp, cosy_cosp)

    # Stack the euler angles into a single tensor
    euler = torch.stack([euler_x, euler_y, euler_z], dim=-1)

    return euler


def euler_to_quat(euler):
    """Convert Euler angles (roll, pitch, yaw) to quaternion (w, x, y, z)"""
    # Convert Euler angles to quaternion using scipy.spatial.transform.Rotation
    rotation = Rotation.from_euler("xyz", euler, degrees=False)
    quat = rotation.as_quat()  # This returns quaternion in xyzw format
    # Convert from xyzw to wxyz format
    quat_wxyz = np.array([quat[3], quat[0], quat[1], quat[2]])
    return quat_wxyz


def quat_to_axis_angle(quat):
    """Convert quaternion to rotation axis and angle"""
    # Convert quaternion to rotation axis and angle
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    angle = 2 * np.arccos(w)
    axis = np.array([x, y, z]) / np.linalg.norm(np.array([x, y, z]))
    return axis, angle


def euler_to_axis_angle(euler):
    """Convert Euler angles (roll, pitch, yaw) to rotation axis and angle"""
    # Convert Euler angles to quaternion
    quat = euler_to_quat(euler)
    # Convert quaternion to rotation axis and angle
    return quat_to_axis_angle(quat)


def axis_angle_to_quat(axis_angle, angle):
    """Convert rotation axis and angle to quaternion (wxyz format)"""
    # Convert axis-angle to quaternion
    w = np.cos(angle / 2)
    x, y, z = axis_angle / np.linalg.norm(axis_angle) * np.sin(angle / 2)
    return np.array([w, x, y, z])


def matrix_to_quat_trans(matrix):
    if not isinstance(matrix, torch.Tensor):
        matrix = torch.tensor(matrix, dtype=torch.float32)

    # Extract rotation matrix and translation vector
    rot = matrix[..., :3, :3]
    trans = matrix[..., :3, 3]

    # Compute trace
    trace = rot[..., 0, 0] + rot[..., 1, 1] + rot[..., 2, 2]

    # Initialize quaternion tensor
    quat = torch.zeros(
        matrix.shape[:-2] + (4,), dtype=matrix.dtype, device=matrix.device
    )

    # Case 1: trace > 0
    mask_trace = trace > 0
    if torch.any(mask_trace):
        s = torch.sqrt(trace[mask_trace] + 1.0) * 2  # 4 * qw
        quat[mask_trace, 0] = 0.25 * s
        quat[mask_trace, 1] = (rot[mask_trace, 2, 1] - rot[mask_trace, 1, 2]) / s
        quat[mask_trace, 2] = (rot[mask_trace, 0, 2] - rot[mask_trace, 2, 0]) / s
        quat[mask_trace, 3] = (rot[mask_trace, 1, 0] - rot[mask_trace, 0, 1]) / s

    # Case 2: trace <= 0 (find largest diagonal element)
    mask_diag = ~mask_trace
    if torch.any(mask_diag):
        diag_idx = torch.argmax(
            torch.diagonal(rot[mask_diag], dim1=-2, dim2=-1), dim=-1
        )
        s = (
            torch.sqrt(
                1.0
                + rot[mask_diag, diag_idx, diag_idx]
                - rot[mask_diag, (diag_idx + 1) % 3, (diag_idx + 1) % 3]
                - rot[mask_diag, (diag_idx + 2) % 3, (diag_idx + 2) % 3]
            )
            * 2
        )
        quat[mask_diag, 0] = (
            rot[mask_diag, (diag_idx + 2) % 3, (diag_idx + 1) % 3]
            - rot[mask_diag, (diag_idx + 1) % 3, (diag_idx + 2) % 3]
        ) / s
        quat[mask_diag, diag_idx + 1] = 0.25 * s
        quat[mask_diag, (diag_idx + 1) % 3 + 1] = (
            rot[mask_diag, diag_idx, (diag_idx + 1) % 3]
            + rot[mask_diag, (diag_idx + 1) % 3, diag_idx]
        ) / s
        quat[mask_diag, (diag_idx + 2) % 3 + 1] = (
            rot[mask_diag, diag_idx, (diag_idx + 2) % 3]
            + rot[mask_diag, (diag_idx + 2) % 3, diag_idx]
        ) / s

    return quat, trans
