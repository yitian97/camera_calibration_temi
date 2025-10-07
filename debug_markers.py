#!/usr/bin/env python3
"""
Debug script to help identify why markers are not being detected.
"""

import cv2
import numpy as np
import sys

# 支持的 ArUco 字典列表
ARUCO_DICTIONARIES = [
    ("DICT_4X4_50", cv2.aruco.DICT_4X4_50),
    ("DICT_4X4_100", cv2.aruco.DICT_4X4_100),
    ("DICT_4X4_250", cv2.aruco.DICT_4X4_250),
    ("DICT_5X5_50", cv2.aruco.DICT_5X5_50),
    ("DICT_5X5_100", cv2.aruco.DICT_5X5_100),
    ("DICT_5X5_250", cv2.aruco.DICT_5X5_250),
    ("DICT_6X6_50", cv2.aruco.DICT_6X6_50),
    ("DICT_6X6_100", cv2.aruco.DICT_6X6_100),
    ("DICT_6X6_250", cv2.aruco.DICT_6X6_250),
    ("DICT_7X7_50", cv2.aruco.DICT_7X7_50),
    ("DICT_7X7_100", cv2.aruco.DICT_7X7_100),
    ("DICT_7X7_250", cv2.aruco.DICT_7X7_250),
    ("DICT_ARUCO_ORIGINAL", cv2.aruco.DICT_ARUCO_ORIGINAL),
]

def test_aruco_detection():
    """Test ArUco marker detection with different parameters."""
    print("=== ArUco Marker Detection Debug ===")
    
    # Initialize camera
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return False
    
    print("✅ Camera opened successfully")
    
    # Initialize ArUco detector
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    aruco_params = cv2.aruco.DetectorParameters()
    aruco_detector = cv2.aruco.ArucoDetector(aruco_dict, aruco_params)
    
    print("✅ ArUco detector initialized")
    print(f"Dictionary: DICT_6X6_250")
    print(f"Parameters: {aruco_params}")
    
    # Test with different parameter settings
    test_params = [
        ("Default", aruco_params),
        ("Adaptive threshold", cv2.aruco.DetectorParameters()),
        ("Corner refinement", cv2.aruco.DetectorParameters())
    ]
    
    # Modify parameters for better detection
    test_params[1][1].adaptiveThreshWinSizeMin = 3
    test_params[1][1].adaptiveThreshWinSizeMax = 23
    test_params[1][1].adaptiveThreshWinSizeStep = 10
    
    test_params[2][1].cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    test_params[2][1].cornerRefinementWinSize = 5
    test_params[2][1].cornerRefinementMaxIterations = 30
    test_params[2][1].cornerRefinementMinAccuracy = 0.1
    
    print("\n=== Testing Detection Parameters ===")
    print("Point your camera at an ArUco marker and press 'q' to quit")
    print("Press '1', '2', '3' to switch between parameter sets")
    
    current_params = 0
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        frame_count += 1
        
        # Get current parameter set
        name, params = test_params[current_params]
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Detect markers
        corners, ids, rejected = detector.detectMarkers(gray)
        
        # Draw results
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            print(f"\r✅ Detected {len(ids)} marker(s) with {name} parameters", end="")
        else:
            print(f"\r❌ No markers detected with {name} parameters (frame {frame_count})", end="")
        
        # Add info text
        cv2.putText(frame, f"Parameters: {name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"Markers detected: {len(ids) if ids is not None else 0}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 1,2,3 to change params, 'q' to quit", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('ArUco Detection Debug', frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('1'):
            current_params = 0
            print(f"\nSwitched to {test_params[0][0]} parameters")
        elif key == ord('2'):
            current_params = 1
            print(f"\nSwitched to {test_params[1][0]} parameters")
        elif key == ord('3'):
            current_params = 2
            print(f"\nSwitched to {test_params[2][0]} parameters")
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def generate_test_marker():
    """Generate a test ArUco marker and save it."""
    print("\n=== Generating Test ArUco Marker ===")
    
    # Create ArUco marker
    aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_6X6_250)
    
    # Generate marker with ID 0
    marker_id = 0
    marker_size = 200  # pixels
    
    marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
    
    # Save marker
    cv2.imwrite('test_aruco_marker.png', marker_image)
    print(f"✅ Test ArUco marker (ID: {marker_id}) saved as 'test_aruco_marker.png'")
    print(f"   Size: {marker_size}x{marker_size} pixels")
    print("   Print this marker and test detection with it")
    
    return True

def test_checkerboard_detection():
    """Test checkerboard detection."""
    print("\n=== Checkerboard Detection Debug ===")
    
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return False
    
    print("✅ Camera opened successfully")
    print("Point your camera at a checkerboard pattern and press 'q' to quit")
    
    checkerboard_size = (9, 6)  # Internal corners
    
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Find checkerboard corners
        ret_corners, corners = cv2.findChessboardCorners(gray, checkerboard_size, None)
        
        if ret_corners:
            cv2.drawChessboardCorners(frame, checkerboard_size, corners, ret_corners)
            print(f"\r✅ Checkerboard detected", end="")
        else:
            print(f"\r❌ No checkerboard detected", end="")
        
        cv2.putText(frame, f"Checkerboard: {'Detected' if ret_corners else 'Not detected'}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        cv2.imshow('Checkerboard Detection Debug', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    return True

def test_aruco_detection_multi_dict():
    """支持多字典切换的 ArUco 检测调试界面"""
    print("=== ArUco Marker Detection (Multi-Dictionary) ===")
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Could not open camera")
        return False
    print("✅ Camera opened successfully")
    current_dict_idx = 2  # 默认 DICT_4X4_250
    params = cv2.aruco.DetectorParameters()
    frame_count = 0
    print("\n=== 操作说明 ===")
    print("按 n/p 切换字典，q 退出，s 保存当前帧")
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        frame_count += 1
        dict_name, dict_type = ARUCO_DICTIONARIES[current_dict_idx]
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        detector = cv2.aruco.ArucoDetector(aruco_dict, params)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = detector.detectMarkers(gray)
        if ids is not None and len(ids) > 0:
            cv2.aruco.drawDetectedMarkers(frame, corners, ids)
            detected_ids = ids.flatten().tolist()
            print(f"\r✅ [{dict_name}] 检测到 {len(ids)} 个标记: {detected_ids}", end="")
        else:
            print(f"\r❌ [{dict_name}] 未检测到标记 (帧 {frame_count})", end="")
        cv2.putText(frame, f"字典: {dict_name}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"检测到: {len(ids) if ids is not None else 0} 个标记", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "n=下一个 | p=上一个 | q=退出 | s=保存", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, f"{current_dict_idx + 1}/{len(ARUCO_DICTIONARIES)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.imshow('ArUco Dictionary Test', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('n'):
            current_dict_idx = (current_dict_idx + 1) % len(ARUCO_DICTIONARIES)
            print(f"\n切换到: {ARUCO_DICTIONARIES[current_dict_idx][0]}")
        elif key == ord('p'):
            current_dict_idx = (current_dict_idx - 1) % len(ARUCO_DICTIONARIES)
            print(f"\n切换到: {ARUCO_DICTIONARIES[current_dict_idx][0]}")
        elif key == ord('s'):
            filename = f'detection_{dict_name}_{frame_count}.jpg'
            cv2.imwrite(filename, frame)
            print(f"\n💾 已保存: {filename}")
    cap.release()
    cv2.destroyAllWindows()
    return True

def generate_test_markers_multi_dict():
    """批量生成多字典多 ID 的 ArUco 标记图片"""
    print("\n=== 批量生成 ArUco 测试标记 ===")
    marker_size = 400  # 像素
    marker_ids = [0, 1, 2]  # 每种字典生成 3 个 ID
    for dict_name, dict_type in ARUCO_DICTIONARIES:
        aruco_dict = cv2.aruco.getPredefinedDictionary(dict_type)
        for marker_id in marker_ids:
            marker_image = cv2.aruco.generateImageMarker(aruco_dict, marker_id, marker_size)
            border_size = 50
            bordered_marker = cv2.copyMakeBorder(
                marker_image, border_size, border_size, border_size, border_size,
                cv2.BORDER_CONSTANT, value=255
            )
            filename = f'aruco_{dict_name}_id{marker_id}.png'
            cv2.imwrite(filename, bordered_marker)
            print(f"✅ 生成: {filename}")
    print("\n📋 使用说明:")
    print("1. 打印不同字典的标记进行测试")
    print("2. 用多字典检测功能查找你的标记属于哪种字典")
    print("3. 确保打印尺寸至少 10cm x 10cm")
    print("4. 在良好光照下测试")
    return True

def main():
    print("🔍 Marker Detection Debug Tool")
    print("=" * 40)
    
    while True:
        print("\nSelect test to run:")
        print("1. Test ArUco marker detection (single dictionary)")
        print("2. Generate test ArUco marker (single dictionary)")
        print("3. Test checkerboard detection")
        print("4. Test ArUco marker detection (multi-dictionary)")
        print("5. Generate test ArUco markers (multi-dictionary)")
        print("6. Exit")

        choice = input("\nEnter choice (1-6): ").strip()

        if choice == '1':
            test_aruco_detection()
        elif choice == '2':
            generate_test_marker()
        elif choice == '3':
            test_checkerboard_detection()
        elif choice == '4':
            test_aruco_detection_multi_dict()
        elif choice == '5':
            generate_test_markers_multi_dict()
        elif choice == '6':
            print("Exiting...")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
