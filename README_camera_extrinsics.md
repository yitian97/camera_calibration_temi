# Camera Extrinsics Estimation

This script estimates camera extrinsics (position and orientation) relative to a known ground point using computer vision techniques.

## Features

- **Camera Calibration**: Automatic camera calibration using checkerboard patterns
- **ArUco Marker Support**: High-precision pose estimation using ArUco markers
- **Checkerboard Support**: Alternative pose estimation using checkerboard patterns
- **Real-time Display**: Live visualization of camera position and orientation
- **Calibration Persistence**: Save and load camera calibration data

## Installation

1. Install Python dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Basic Usage (ArUco Marker)

1. **Print an ArUco marker**: Generate a marker using online tools or OpenCV
2. **Place marker at ground point**: Position the marker at your known (0,0) location
3. **Run the script**:
```bash
python camera_extrinsics.py
```

### Using Checkerboard Pattern

```bash
python camera_extrinsics.py --use-checkerboard
```

### Command Line Options

- `--camera ID`: Camera device ID (default: 0)
- `--marker-size SIZE`: ArUco marker size in meters (default: 0.05)
- `--use-checkerboard`: Use checkerboard instead of ArUco marker
- `--calibrate-only`: Only perform camera calibration

### Examples

```bash
# Use default settings with ArUco marker
python camera_extrinsics.py

# Use camera 1 with 10cm marker
python camera_extrinsics.py --camera 1 --marker-size 0.1

# Use checkerboard pattern
python camera_extrinsics.py --use-checkerboard

# Only calibrate camera (no pose estimation)
python camera_extrinsics.py --calibrate-only
```

## How It Works

1. **Camera Calibration**: The script first calibrates the camera using a checkerboard pattern to determine intrinsic parameters
2. **Marker Detection**: Detects ArUco markers or checkerboard patterns in the camera feed
3. **Pose Estimation**: Uses PnP (Perspective-n-Point) algorithm to estimate camera pose relative to the marker
4. **Coordinate System**: The marker defines the world coordinate system with (0,0,0) at the marker center

## Output

The script displays:
- **Position**: Camera position in meters relative to the marker (X, Y, Z)
- **Rotation**: Camera orientation as Euler angles (Roll, Pitch, Yaw) in degrees
- **Visual Feedback**: Live camera feed with detected markers and coordinate axes

## Tips for Best Results

1. **Good Lighting**: Ensure adequate lighting for marker detection
2. **Stable Camera**: Keep the camera steady for accurate measurements
3. **Marker Quality**: Use high-contrast, well-printed markers
4. **Distance**: Keep the marker at a reasonable distance (not too close or too far)
5. **Angle**: Try different viewing angles for better pose estimation

## Troubleshooting

- **No marker detected**: Check lighting and marker visibility
- **Calibration failed**: Ensure checkerboard is clearly visible and captured from multiple angles
- **Poor accuracy**: Recalibrate the camera or check marker quality
- **Camera not found**: Try different camera IDs (0, 1, 2, etc.)

## File Structure

- `camera_extrinsics.py`: Main script
- `requirements.txt`: Python dependencies
- `camera_calibration.npz`: Saved camera calibration data (created after first run)

