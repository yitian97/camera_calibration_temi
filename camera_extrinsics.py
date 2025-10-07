#!/usr/bin/env python3
"""
Camera Extrinsics Estimation Script

This script estimates camera extrinsics (position and orientation) relative to a known ground point.
It uses ArUco markers or checkerboard patterns for pose estimation.

Requirements:
- OpenCV with ArUco support
- NumPy
- A printed ArUco marker or checkerboard pattern

Usage:
1. Print an ArUco marker (recommended) or checkerboard pattern
2. Place the marker at your known ground point (0,0)
3. Run this script and point the camera at the marker
4. The script will display the camera's position and rotation relative to the marker
"""

import cv2
import numpy as np
import argparse
import sys
from typing import Tuple, Optional

class CameraExtrinsicsEstimator:
    def __init__(self, camera_id: int = 0, marker_size: float = 0.05):
        """
        Initialize the camera extrinsics estimator.
        
        Args:
            camera_id: Camera device ID (usually 0 for default webcam)
            marker_size: Physical size of the marker in meters
        """
        self.camera_id = camera_id
        self.marker_size = marker_size
        self.cap = None
        self.camera_matrix = None
        self.dist_coeffs = None
        self.calibrated = False
        
        # ArUco dictionary and parameters (OpenCV 4.x+ API)
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_5X5_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        self.aruco_detector = cv2.aruco.ArucoDetector(self.aruco_dict, self.aruco_params)
        
        # Checkerboard parameters (alternative to ArUco)
        self.checkerboard_size = (9, 6)  # Internal corners
        self.checkerboard_square_size = 0.025  # 25mm squares
        
    def initialize_camera(self) -> bool:
        """Initialize the camera and attempt to calibrate it."""
        print("Initializing camera...")
        self.cap = cv2.VideoCapture(self.camera_id)
        
        if not self.cap.isOpened():
            print(f"Error: Could not open camera {self.camera_id}")
            return False
            
        # Set camera properties for better quality
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        self.cap.set(cv2.CAP_PROP_FPS, 30)
        
        print("Camera initialized successfully")
        return True
    
    def calibrate_camera(self) -> bool:
        """
        Perform camera calibration using a checkerboard pattern.
        This needs to be done once for accurate pose estimation.
        """
        print("\n=== Camera Calibration ===")
        print("Show a checkerboard pattern to the camera from different angles.")
        print("Press 'c' to capture calibration images, 'q' to quit calibration")
        print("You need at least 10 good images for calibration")
        
        # Prepare object points
        objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
        objp *= self.checkerboard_square_size
        
        # Arrays to store object points and image points
        objpoints = []  # 3D points in real world space
        imgpoints = []  # 2D points in image plane
        
        calibration_images = 0
        target_images = 15
        
        while calibration_images < target_images:
            ret, frame = self.cap.read()
            if not ret:
                continue
                
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Find checkerboard corners
            ret_corners, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
            
            # Draw and display the corners
            cv2.drawChessboardCorners(frame, self.checkerboard_size, corners, ret_corners)
            cv2.putText(frame, f"Calibration images: {calibration_images}/{target_images}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.putText(frame, "Press 'c' to capture, 'q' to quit", 
                       (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            cv2.imshow('Camera Calibration', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('c') and ret_corners:
                objpoints.append(objp)
                imgpoints.append(corners)
                calibration_images += 1
                print(f"Captured calibration image {calibration_images}/{target_images}")
            elif key == ord('q'):
                break
        
        cv2.destroyWindow('Camera Calibration')
        
        if len(objpoints) < 10:
            print("Not enough calibration images. Please try again.")
            return False
        
        print("Performing camera calibration...")
        ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(
            objpoints, imgpoints, gray.shape[::-1], None, None
        )
        
        if ret:
            self.camera_matrix = camera_matrix
            self.dist_coeffs = dist_coeffs
            self.calibrated = True
            print("Camera calibration completed successfully!")
            print(f"Camera matrix:\n{camera_matrix}")
            print(f"Distortion coefficients: {dist_coeffs.flatten()}")
            return True
        else:
            print("Camera calibration failed!")
            return False
    
    def load_calibration(self, calibration_file: str) -> bool:
        """Load camera calibration from a file."""
        try:
            data = np.load(calibration_file)
            self.camera_matrix = data['camera_matrix']
            self.dist_coeffs = data['dist_coeffs']
            self.calibrated = True
            print(f"Loaded calibration from {calibration_file}")
            return True
        except Exception as e:
            print(f"Error loading calibration: {e}")
            return False
    
    def save_calibration(self, calibration_file: str) -> bool:
        """Save camera calibration to a file."""
        if not self.calibrated:
            print("No calibration data to save")
            return False
        
        try:
            np.savez(calibration_file, 
                    camera_matrix=self.camera_matrix, 
                    dist_coeffs=self.dist_coeffs)
            print(f"Calibration saved to {calibration_file}")
            return True
        except Exception as e:
            print(f"Error saving calibration: {e}")
            return False
    
    def estimate_pose_aruco(self, frame, skip_calibration: bool = False, use_optitrack_coords: bool = True) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Estimate camera pose using ArUco marker detection."""
        if not self.calibrated and not skip_calibration:
            return None, None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers (OpenCV 4.x+ API)
        corners, ids, rejected = self.aruco_detector.detectMarkers(gray)
        
        if ids is not None and len(ids) > 0:
            if skip_calibration:
                # Use default camera matrix for rough estimation
                h, w = frame.shape[:2]
                fx = fy = max(h, w)  # Rough focal length estimate
                cx, cy = w/2, h/2    # Image center
                camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                dist_coeffs = np.zeros((4, 1), dtype=np.float32)
            else:
                camera_matrix = self.camera_matrix
                dist_coeffs = self.dist_coeffs
            
            # Define marker corners in 3D space (OpenCV convention: X-right, Y-down, Z-forward)
            # When marker is on ground, this defines a marker lying flat with:
            # - X pointing right
            # - Y pointing forward (marker top direction)
            # - Z pointing up (perpendicular to marker plane)
            marker_corners_3d = np.array([
                [-self.marker_size/2, self.marker_size/2, 0],   # top-left
                [self.marker_size/2, self.marker_size/2, 0],    # top-right
                [self.marker_size/2, -self.marker_size/2, 0],   # bottom-right
                [-self.marker_size/2, -self.marker_size/2, 0]   # bottom-left
            ], dtype=np.float32)
            
            rvecs = []
            tvecs = []
            for i in range(len(ids)):
                # solvePnP for each detected marker
                success, rvec, tvec = cv2.solvePnP(
                    marker_corners_3d, corners[i], camera_matrix, dist_coeffs
                )
                if success:
                    rvecs.append(rvec)
                    tvecs.append(tvec)

            # Draw marker detection
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)

            # Draw coordinate axes
            for i in range(len(rvecs)):
                if use_optitrack_coords:
                    # Transform to OptiTrack coordinate system for visualization
                    # OptiTrack: X-left, Y-up, Z-forward
                    # Marker local: X-right, Y-forward, Z-up
                    # We need to rotate the axes to show OptiTrack world frame

                    # Rotation from marker local to OptiTrack world:
                    # Marker X (right) -> OptiTrack -X (left becomes positive right in display)
                    # Marker Y (forward) -> OptiTrack Z (forward)
                    # Marker Z (up) -> OptiTrack Y (up)
                    R_marker_to_optitrack = np.array([
                        [-1, 0, 0],  # Marker X (right) -> -X (left)
                        [0, 0, 1],   # Marker Z (up) -> Y (up)
                        [0, 1, 0]    # Marker Y (forward) -> Z (forward)
                    ], dtype=np.float32)

                    R_marker, _ = cv2.Rodrigues(rvecs[i])
                    R_optitrack = R_marker @ R_marker_to_optitrack
                    rvec_viz, _ = cv2.Rodrigues(R_optitrack)

                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvec_viz, tvecs[i], self.marker_size * 0.5)
                else:
                    # Draw marker local coordinate system
                    cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, rvecs[i], tvecs[i], self.marker_size * 0.5)

            # 返回第一个 marker 的 rvec/tvec
            if rvecs and tvecs:
                return rvecs[0], tvecs[0]
            else:
                return None, None

        return None, None
    
    def estimate_pose_checkerboard(self, frame, skip_calibration: bool = False) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Estimate camera pose using checkerboard detection."""
        if not self.calibrated and not skip_calibration:
            return None, None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        
        if ret:
            # Prepare object points
            objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
            objp *= self.checkerboard_square_size
            
            if skip_calibration:
                # Use default camera matrix for rough estimation
                h, w = frame.shape[:2]
                fx = fy = max(h, w)  # Rough focal length estimate
                cx, cy = w/2, h/2    # Image center
                camera_matrix = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)
                dist_coeffs = np.zeros((4, 1), dtype=np.float32)
            else:
                camera_matrix = self.camera_matrix
                dist_coeffs = self.dist_coeffs
            
            # Solve PnP
            ret, rvec, tvec = cv2.solvePnP(objp, corners, camera_matrix, dist_coeffs)
            
            if ret:
                # Draw checkerboard corners
                cv2.drawChessboardCorners(frame, self.checkerboard_size, corners, ret)
                
                # Draw coordinate axes
                cv2.drawFrameAxes(frame, camera_matrix, dist_coeffs, 
                                rvec, tvec, 0.1)
                
                return rvec, tvec
        
        return None, None
    
    def rotation_vector_to_euler(self, rvec: np.ndarray) -> Tuple[float, float, float]:
        """Convert rotation vector to Euler angles (roll, pitch, yaw) in degrees."""
        # Convert rotation vector to rotation matrix
        R, _ = cv2.Rodrigues(rvec)
        
        # Extract Euler angles (ZYX convention)
        sy = np.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
        singular = sy < 1e-6
        
        if not singular:
            x = np.arctan2(R[2, 1], R[2, 2])
            y = np.arctan2(-R[2, 0], sy)
            z = np.arctan2(R[1, 0], R[0, 0])
        else:
            x = np.arctan2(-R[1, 2], R[1, 1])
            y = np.arctan2(-R[2, 0], sy)
            z = 0
        
        return np.degrees(x), np.degrees(y), np.degrees(z)
    
    def opencv_to_optitrack(self, rvec: np.ndarray, tvec: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        将标记到相机的变换(marker_to_camera)转换为OptiTrack世界坐标系到相机的变换。

        步骤：
        1. solvePnP 返回的是 marker_to_camera (标记本地坐标系到相机)
        2. 标记本地坐标系: X-right, Y-forward, Z-up
        3. OptiTrack 世界坐标系: X-left, Y-up, Z-forward
        4. 需要先将标记坐标系转换到世界坐标系，再到相机坐标系

        Args:
            rvec: 标记到相机的旋转向量(OpenCV坐标系)
            tvec: 标记到相机的平移向量(OpenCV坐标系)

        Returns:
            rvec_world_to_camera: OptiTrack世界坐标系到相机的旋转向量
            tvec_world_to_camera: OptiTrack世界坐标系到相机的平移向量
        """
        # Step 1: 将 marker_to_camera 的 OpenCV 表示转换为旋转矩阵
        R_marker_to_camera_opencv, _ = cv2.Rodrigues(rvec)
        t_marker_to_camera_opencv = tvec

        # Step 2: 定义标记本地坐标系到OptiTrack世界坐标系的变换
        # 标记本地: X-right, Y-forward, Z-up
        # OptiTrack: X-left, Y-up, Z-forward
        # 变换: Marker X -> World -X, Marker Y -> World Z, Marker Z -> World Y
        R_marker_to_world = np.array([
            [-1, 0, 0],  # Marker X (right) -> World -X (becomes left)
            [0, 0, 1],   # Marker Z (up) -> World Y (up)
            [0, 1, 0]    # Marker Y (forward) -> World Z (forward)
        ], dtype=np.float32)

        # 世界到标记的变换(逆变换)
        R_world_to_marker = R_marker_to_world.T
        t_world_to_marker = np.zeros((3, 1))  # 标记中心在世界原点

        # Step 3: 链式变换 world -> marker -> camera
        # R_world_to_camera = R_marker_to_camera @ R_world_to_marker
        # t_world_to_camera = R_marker_to_camera @ t_world_to_marker + t_marker_to_camera
        R_world_to_camera_opencv = R_marker_to_camera_opencv @ R_world_to_marker
        t_world_to_camera_opencv = R_marker_to_camera_opencv @ t_world_to_marker + t_marker_to_camera_opencv

        # Step 4: 转换为旋转向量
        rvec_world_to_camera, _ = cv2.Rodrigues(R_world_to_camera_opencv)
        tvec_world_to_camera = t_world_to_camera_opencv

        return rvec_world_to_camera, tvec_world_to_camera

    def run_extrinsics_estimation(self, use_aruco: bool = True, skip_calibration: bool = False, use_optitrack_coords: bool = True):
        """Main loop for extrinsics estimation."""
        if not self.initialize_camera():
            return
        # Try to load existing calibration first (unless skipping)
        if not skip_calibration:
            if not self.load_calibration('camera_calibration.npz'):
                print("No existing calibration found. Starting calibration process...")
                if not self.calibrate_camera():
                    print("Calibration failed. Exiting.")
                    return
                self.save_calibration('camera_calibration.npz')
        else:
            print("Skipping intrinsic calibration - using rough camera parameters")
        print("\n=== Extrinsics Estimation ===")
        if use_optitrack_coords:
            print("📍 Coordinate System: OptiTrack (X-right, Y-up, Z-forward)")
            print("   Place marker flat on ground with Z-axis pointing forward")
        else:
            print("📍 Coordinate System: OpenCV (X-right, Y-down, Z-forward)")
        print("Point the camera at the marker/checkerboard at your ground point (0,0)")
        print("Press 'q' to quit, 's' to save extrinsics")
        if skip_calibration:
            print("WARNING: Using uncalibrated camera - results may be less accurate")
        last_rvec, last_tvec = None, None
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            # Estimate pose (OpenCV坐标系)
            if use_aruco:
                rvec, tvec = self.estimate_pose_aruco(frame, skip_calibration, use_optitrack_coords)
            else:
                rvec, tvec = self.estimate_pose_checkerboard(frame, skip_calibration)
            if rvec is not None and tvec is not None:
                # 转换到OptiTrack坐标系（如需）
                if use_optitrack_coords:
                    rvec_disp, tvec_disp = self.opencv_to_optitrack(rvec, tvec)
                    coord_label = "OptiTrack"
                else:
                    rvec_disp, tvec_disp = rvec, tvec
                    coord_label = "OpenCV"
                last_rvec, last_tvec = rvec_disp, tvec_disp
                position = tvec_disp.flatten()
                rotation_euler = self.rotation_vector_to_euler(rvec_disp)
                # Display information
                cv2.putText(frame, f"Position (X,Y,Z): ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Rotation (Roll,Pitch,Yaw): ({rotation_euler[0]:.1f}, {rotation_euler[1]:.1f}, {rotation_euler[2]:.1f})", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Coordinate System: {coord_label}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2)
                if skip_calibration:
                    cv2.putText(frame, "UNCALIBRATED - Results may be inaccurate", 
                               (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                # Print to console
                print(f"\rCamera Position: X={position[0]:.3f}m, Y={position[1]:.3f}m, Z={position[2]:.3f}m", end="")
                print(f" | Rotation: Roll={rotation_euler[0]:.1f}°, Pitch={rotation_euler[1]:.1f}°, Yaw={rotation_euler[2]:.1f}°", end="")
            else:
                cv2.putText(frame, "No marker detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            cv2.putText(frame, "Press 'q' to quit, 's' to save extrinsics", (10, frame.shape[0]-20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
            cv2.imshow('Camera Extrinsics Estimation', frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                if last_rvec is not None and last_tvec is not None:
                    coord_system = 'optitrack' if use_optitrack_coords else 'opencv'
                    np.savez(f'extrinsics_{coord_system}.npz', rvec=last_rvec, tvec=last_tvec, coordinate_system=coord_system)
                    print(f"\n💾 Extrinsics saved to extrinsics_{coord_system}.npz")
                    print(f"   Position (X,Y,Z): {last_tvec.flatten()}")
                    if use_optitrack_coords:
                        print(f"   Coordinate System: OptiTrack (X-right, Y-up, Z-forward)")
                    else:
                        print(f"   Coordinate System: OpenCV (X-right, Y-down, Z-forward)")
                else:
                    print("\n❌ No valid extrinsics to save!")
        self.cleanup()
    
    def cleanup(self):
        """Clean up resources."""
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("\nExiting...")


def main():
    parser = argparse.ArgumentParser(description='Estimate camera extrinsics from webcam')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID (default: 0)')
    parser.add_argument('--marker-size', type=float, default=0.09, help='ArUco marker size in meters (default: 0.05)')
    parser.add_argument('--use-checkerboard', action='store_true', help='Use checkerboard instead of ArUco marker')
    parser.add_argument('--calibrate-only', action='store_true', help='Only perform camera calibration')
    parser.add_argument('--skip-calibration', action='store_true', help='Skip intrinsic calibration and use rough camera parameters')
    parser.add_argument('--opencv-coords', action='store_true', help='Output in OpenCV coordinate system (default: OptiTrack)')
    args = parser.parse_args()
    estimator = CameraExtrinsicsEstimator(camera_id=args.camera, marker_size=args.marker_size)
    if args.calibrate_only:
        if estimator.initialize_camera():
            estimator.calibrate_camera()
            estimator.save_calibration('camera_calibration.npz')
            estimator.cleanup()
    else:
        estimator.run_extrinsics_estimation(use_aruco=not args.use_checkerboard, skip_calibration=args.skip_calibration, use_optitrack_coords=not args.opencv_coords)


if __name__ == "__main__":
    main()
