import cv2
import os
import numpy as np

# Input and output directories
input_dir = "C:\\Users\\33426\\Desktop\\CV_2252709_yxh_Assignment2\\problem4\\picture"
output_dir = "C:\\Users\\33426\\Desktop\\CV_2252709_yxh_Assignment2\\problem4\\result"

# Create the output directory if it doesn't exist
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Chessboard parameters
pattern_size = (9, 6)

# Initialize variables
obj_points = []  # Store 3D points in the world coordinate system
img_points = []  # Store 2D points in the image plane

# Prepare 3D points of the chessboard
objp = np.zeros((pattern_size[0] * pattern_size[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:pattern_size[0], 0:pattern_size[1]].T.reshape(-1, 2)

# First pass: Camera calibration
for img_name in os.listdir(input_dir):
    input_image_path = os.path.join(input_dir, img_name)

    # Read the input image
    frame = cv2.imread(input_image_path)
    if frame is None:
        print(f"Unable to read the input image: {input_image_path}. Please check the path.")
        continue

    # Detect chessboard corners
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

    if ret:
        # Refine corner positions
        corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                  (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001))

        # Add 3D and 2D points
        obj_points.append(objp)
        img_points.append(corners2)
    else:
        print(f"Chessboard corners not found in image: {input_image_path}")

# Camera calibration
if len(obj_points) > 0:
    ret, camera_matrix, dist_coeffs, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, gray.shape[::-1], None, None)
    
    # Save camera parameters to txt file
    param_file_path = os.path.join(output_dir, "camera_parameters.txt")
    with open(param_file_path, 'w') as f:
        f.write("Camera Matrix:\n")
        f.write(str(camera_matrix))
        f.write("\n\nDistortion Coefficients:\n")
        f.write(str(dist_coeffs))
    
    print("Camera parameters saved to:", param_file_path)

    # Second pass: Bird's eye view transformation
    for img_name in os.listdir(input_dir):
        input_image_path = os.path.join(input_dir, img_name)

        # Read the input image
        frame = cv2.imread(input_image_path)
        if frame is None:
            continue

        # Undistort the image
        undistorted_image = cv2.undistort(frame, camera_matrix, dist_coeffs)

        # Convert to grayscale and detect chessboard corners
        gray = cv2.cvtColor(undistorted_image, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None)

        if ret:
            # Refine corner positions
            corners2 = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1),
                                      (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.001))

            # Select points for perspective transformation
            src_pts = np.float32([corners2[0], corners2[pattern_size[0] - 1], 
                                corners2[-1], corners2[-pattern_size[0]]])

            # Define destination points
            h, w = undistorted_image.shape[:2]
            dst_pts = np.float32([[702.0, 514.0],  # 左上
                                 [1222.0, 514.0],  # 右上
                                 [1222.0, 914.0],  # 右下
                                 [702.0, 914.0]])  # 左下

            # Compute and apply perspective transformation
            M = cv2.getPerspectiveTransform(src_pts, dst_pts)
            birdseye_image = cv2.warpPerspective(undistorted_image, M, (w, h))

            # Save the result
            output_image_path = os.path.join(output_dir, f"birdseye_{img_name}")
            cv2.imwrite(output_image_path, birdseye_image)

            print(f"Bird's eye view saved as: {output_image_path}")
        else:
            print(f"Chessboard corners not found in image: {input_image_path}")
else:
    print("No valid images found for camera calibration")