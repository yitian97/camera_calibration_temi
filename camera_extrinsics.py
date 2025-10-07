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
        
        # ArUco dictionary and parameters
        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
        self.aruco_params = cv2.aruco.DetectorParameters()
        
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
    
    def estimate_pose_aruco(self, frame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Estimate camera pose using ArUco marker detection."""
        if not self.calibrated:
            return None, None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect ArUco markers
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.aruco_params)
        
        if ids is not None and len(ids) > 0:
            # Estimate pose of the marker
            rvecs, tvecs, _ = cv2.aruco.estimatePoseSingleMarkers(
                corners, self.marker_size, self.camera_matrix, self.dist_coeffs
            )
            
            # Draw marker and pose
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            
            for i in range(len(ids)):
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, 
                                rvecs[i], tvecs[i], self.marker_size * 0.5)
            
            return rvecs[0], tvecs[0]
        
        return None, None
    
    def estimate_pose_checkerboard(self, frame) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """Estimate camera pose using checkerboard detection."""
        if not self.calibrated:
            return None, None
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret, corners = cv2.findChessboardCorners(gray, self.checkerboard_size, None)
        
        if ret:
            # Prepare object points
            objp = np.zeros((self.checkerboard_size[0] * self.checkerboard_size[1], 3), np.float32)
            objp[:, :2] = np.mgrid[0:self.checkerboard_size[0], 0:self.checkerboard_size[1]].T.reshape(-1, 2)
            objp *= self.checkerboard_square_size
            
            # Solve PnP
            ret, rvec, tvec = cv2.solvePnP(objp, corners, self.camera_matrix, self.dist_coeffs)
            
            if ret:
                # Draw checkerboard corners
                cv2.drawChessboardCorners(frame, self.checkerboard_size, corners, ret)
                
                # Draw coordinate axes
                cv2.drawFrameAxes(frame, self.camera_matrix, self.dist_coeffs, 
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
    
    def run_extrinsics_estimation(self, use_aruco: bool = True):
        """Main loop for extrinsics estimation."""
        if not self.initialize_camera():
            return
        
        # Try to load existing calibration first
        # if not self.load_calibration('camera_calibration.npz'):
        #     print("No existing calibration found. Starting calibration process...")
        #     if not self.calibrate_camera():
        #         print("Calibration failed. Exiting.")
        #         return
        #     self.save_calibration('camera_calibration.npz')
        
        print("\n=== Extrinsics Estimation ===")
        print("Point the camera at the marker/checkerboard at your ground point (0,0)")
        print("Press 'q' to quit")
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            # Estimate pose
            if use_aruco:
                rvec, tvec = self.estimate_pose_aruco(frame)
            else:
                rvec, tvec = self.estimate_pose_checkerboard(frame)
            
            if rvec is not None and tvec is not None:
                # Convert to more readable format
                position = tvec.flatten()
                rotation_euler = self.rotation_vector_to_euler(rvec)
                
                # Display information
                cv2.putText(frame, f"Position (X,Y,Z): ({position[0]:.3f}, {position[1]:.3f}, {position[2]:.3f})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Rotation (Roll,Pitch,Yaw): ({rotation_euler[0]:.1f}, {rotation_euler[1]:.1f}, {rotation_euler[2]:.1f})", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                
                # Print to console
                print(f"\rCamera Position: X={position[0]:.3f}m, Y={position[1]:.3f}m, Z={position[2]:.3f}m", end="")
                print(f" | Rotation: Roll={rotation_euler[0]:.1f}°, Pitch={rotation_euler[1]:.1f}°, Yaw={rotation_euler[2]:.1f}°", end="")
            else:
                cv2.putText(frame, "No marker detected", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            
            cv2.imshow('Camera Extrinsics Estimation', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
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
    parser.add_argument('--marker-size', type=float, default=0.05, help='ArUco marker size in meters (default: 0.05)')
    parser.add_argument('--use-checkerboard', action='store_true', help='Use checkerboard instead of ArUco marker')
    parser.add_argument('--calibrate-only', action='store_true', help='Only perform camera calibration')
    
    args = parser.parse_args()
    
    estimator = CameraExtrinsicsEstimator(camera_id=args.camera, marker_size=args.marker_size)
    
    if args.calibrate_only:
        if estimator.initialize_camera():
            estimator.calibrate_camera()
            estimator.save_calibration('camera_calibration.npz')
            estimator.cleanup()
    else:
        estimator.run_extrinsics_estimation(use_aruco=not args.use_checkerboard)


if __name__ == "__main__":
    main()

