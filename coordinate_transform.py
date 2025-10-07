#!/usr/bin/env python3
"""
Coordinate Transformation Utilities

This script provides utilities to transform points between camera coordinate system
and OptiTrack world coordinate system using saved camera extrinsics.

Usage:
    python coordinate_transform.py --test
"""

import numpy as np
import cv2
import argparse
from typing import Tuple, Union


class CoordinateTransformer:
    """Transform coordinates between camera and OptiTrack world coordinate systems."""

    def __init__(self, extrinsics_file: str = 'extrinsics_optitrack.npz'):
        """
        Initialize coordinate transformer with saved extrinsics.

        Args:
            extrinsics_file: Path to saved extrinsics file (.npz)
        """
        self.extrinsics_file = extrinsics_file
        self.R_camera_to_world = None
        self.t_camera_to_world = None
        self.R_world_to_camera = None
        self.t_world_to_camera = None

        self.load_extrinsics()

    def load_extrinsics(self):
        """Load and prepare extrinsics for coordinate transformation."""
        try:
            data = np.load(self.extrinsics_file)
            rvec = data['rvec']  # Rotation vector (world to camera in OptiTrack coords)
            tvec = data['tvec']  # Translation vector (world to camera in OptiTrack coords)

            # Convert rotation vector to rotation matrix
            R_world_to_camera, _ = cv2.Rodrigues(rvec)
            t_world_to_camera = tvec.flatten()

            # Standard camera extrinsics define world-to-camera transformation:
            # P_camera = R_world_to_camera @ P_world + t_world_to_camera

            # For coordinate transformation, we often need the inverse (camera-to-world):
            # P_world = R_camera_to_world @ P_camera + t_camera_to_world

            # World-to-camera transformation (standard extrinsics)
            self.R_world_to_camera = R_world_to_camera
            self.t_world_to_camera = t_world_to_camera.reshape(3, 1)

            # Camera-to-world transformation (inverse)
            self.R_camera_to_world = R_world_to_camera.T  # Inverse of rotation
            self.t_camera_to_world = -self.R_camera_to_world @ self.t_world_to_camera

            print(f"✅ Loaded extrinsics from {self.extrinsics_file}")
            print(f"📍 Camera position in world: {self.t_camera_to_world.flatten()}")
            print(f"   (Computed as inverse of extrinsics)")

        except Exception as e:
            print(f"❌ Error loading extrinsics: {e}")
            raise

    def camera_to_world(self, points_camera: np.ndarray) -> np.ndarray:
        """
        Transform points from camera coordinate system to OptiTrack world coordinate system.

        Args:
            points_camera: Points in camera coordinates, shape (3,) or (N, 3)

        Returns:
            Points in world coordinates, same shape as input
        """
        points_camera = np.asarray(points_camera, dtype=np.float64)

        # Handle single point or multiple points
        if points_camera.ndim == 1:
            if len(points_camera) != 3:
                raise ValueError("Point must have 3 coordinates (x, y, z)")
            points_camera = points_camera.reshape(1, 3)
            single_point = True
        else:
            single_point = False
            if points_camera.shape[1] != 3:
                raise ValueError("Points must have shape (N, 3)")

        # Transform: P_world = R @ P_camera + t
        points_world = (self.R_camera_to_world @ points_camera.T).T + self.t_camera_to_world.T

        if single_point:
            return points_world.flatten()
        return points_world

    def world_to_camera(self, points_world: np.ndarray) -> np.ndarray:
        """
        Transform points from OptiTrack world coordinate system to camera coordinate system.

        Args:
            points_world: Points in world coordinates, shape (3,) or (N, 3)

        Returns:
            Points in camera coordinates, same shape as input
        """
        points_world = np.asarray(points_world, dtype=np.float64)

        # Handle single point or multiple points
        if points_world.ndim == 1:
            if len(points_world) != 3:
                raise ValueError("Point must have 3 coordinates (x, y, z)")
            points_world = points_world.reshape(1, 3)
            single_point = True
        else:
            single_point = False
            if points_world.shape[1] != 3:
                raise ValueError("Points must have shape (N, 3)")

        # Transform: P_camera = R @ (P_world - t)
        points_camera = (self.R_world_to_camera @ (points_world - self.t_camera_to_world).T).T

        if single_point:
            return points_camera.flatten()
        return points_camera

    def get_camera_pose_in_world(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get camera position and orientation in OptiTrack world coordinates.

        Returns:
            position: Camera position (X, Y, Z) in meters
            rotation_matrix: Camera orientation as 3x3 rotation matrix
        """
        return self.t_camera_to_world.flatten(), self.R_camera_to_world

    def get_transformation_matrix(self, camera_to_world: bool = True) -> np.ndarray:
        """
        Get 4x4 homogeneous transformation matrix.

        Args:
            camera_to_world: If True, return camera-to-world transform, else world-to-camera

        Returns:
            4x4 transformation matrix
        """
        T = np.eye(4)
        if camera_to_world:
            T[:3, :3] = self.R_camera_to_world
            T[:3, 3] = self.t_camera_to_world.flatten()
        else:
            T[:3, :3] = self.R_world_to_camera
            T[:3, 3] = self.t_world_to_camera.flatten()
        return T


def test_coordinate_transform():
    """Test coordinate transformation with sample points."""
    print("\n=== Testing Coordinate Transformation ===\n")

    try:
        transformer = CoordinateTransformer('extrinsics_optitrack.npz')
    except Exception as e:
        print("❌ Could not load extrinsics. Please run camera_extrinsics.py first and save extrinsics.")
        return

    # Get camera pose
    cam_pos, cam_rot = transformer.get_camera_pose_in_world()
    print(f"\n📷 Camera Position in OptiTrack World:")
    print(f"   X: {cam_pos[0]:.3f}m (left+/right-)")
    print(f"   Y: {cam_pos[1]:.3f}m (up+/down-)")
    print(f"   Z: {cam_pos[2]:.3f}m (forward+/back-)")

    # Test 1: Transform point in front of camera
    print("\n=== Test 1: Point in Camera Coordinates ===")
    point_camera = np.array([0.0, 0.0, 1.0])  # 1m in front of camera
    print(f"Point in camera coords: {point_camera}")

    point_world = transformer.camera_to_world(point_camera)
    print(f"Point in world coords:  [{point_world[0]:.3f}, {point_world[1]:.3f}, {point_world[2]:.3f}]")

    # Verify round-trip
    point_camera_back = transformer.world_to_camera(point_world)
    print(f"Round-trip back:        [{point_camera_back[0]:.3f}, {point_camera_back[1]:.3f}, {point_camera_back[2]:.3f}]")
    print(f"✅ Round-trip error: {np.linalg.norm(point_camera - point_camera_back):.6f}m")

    # Test 2: Transform multiple points
    print("\n=== Test 2: Multiple Points ===")
    points_camera = np.array([
        [0.0, 0.0, 1.0],   # 1m forward
        [0.5, 0.0, 1.0],   # 1m forward, 0.5m right
        [-0.5, 0.0, 1.0],  # 1m forward, 0.5m left
    ])
    print(f"Points in camera coords:")
    print(points_camera)

    points_world = transformer.camera_to_world(points_camera)
    print(f"\nPoints in world coords:")
    print(points_world)

    # Test 3: World origin in camera coordinates
    print("\n=== Test 3: World Origin (Marker Center) ===")
    world_origin = np.array([0.0, 0.0, 0.0])
    origin_in_camera = transformer.world_to_camera(world_origin)
    print(f"World origin in camera coords: [{origin_in_camera[0]:.3f}, {origin_in_camera[1]:.3f}, {origin_in_camera[2]:.3f}]")

    # Test 4: Get transformation matrix
    print("\n=== Test 4: Transformation Matrix ===")
    T_cam_to_world = transformer.get_transformation_matrix(camera_to_world=True)
    print("Camera-to-World 4x4 Matrix:")
    print(T_cam_to_world)


def example_usage():
    """Show example usage code."""
    print("\n=== Example Usage ===\n")
    print("""
# Load the coordinate transformer
from coordinate_transform import CoordinateTransformer

transformer = CoordinateTransformer('extrinsics_optitrack.npz')

# Transform a single point from camera to world
point_camera = np.array([0.1, 0.2, 0.5])  # x, y, z in camera frame
point_world = transformer.camera_to_world(point_camera)
print(f"World coords: {point_world}")

# Transform multiple points
points_camera = np.array([[0, 0, 1], [0.5, 0, 1]])  # (N, 3)
points_world = transformer.camera_to_world(points_camera)

# Reverse transformation
point_camera_back = transformer.world_to_camera(point_world)

# Get camera pose in world
cam_position, cam_rotation = transformer.get_camera_pose_in_world()

# Get 4x4 transformation matrix for other libraries (e.g., Open3D, PyTorch3D)
T = transformer.get_transformation_matrix(camera_to_world=True)
    """)


def main():
    parser = argparse.ArgumentParser(description='Coordinate transformation utilities')
    parser.add_argument('--test', action='store_true', help='Run test transformations')
    parser.add_argument('--example', action='store_true', help='Show example usage')
    parser.add_argument('--extrinsics', type=str, default='extrinsics_optitrack.npz',
                       help='Path to extrinsics file')

    args = parser.parse_args()

    if args.example:
        example_usage()
    elif args.test:
        test_coordinate_transform()
    else:
        print("Use --test to run tests or --example to see usage examples")
        print("Example: python coordinate_transform.py --test")


if __name__ == "__main__":
    main()
