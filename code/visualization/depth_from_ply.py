import numpy as np
import open3d as o3d  # To load the point cloud
import cv2  # To save the depth image

import sys
from path_config import DATA_FOLDER, CONFIG_FOLDER, CODE_FOLDER
sys.path.append(str(CODE_FOLDER))
from my_utils.read_camera_matrices import read_camera_matrices_dic

def generate_depth_image_from_ply(ply_file, intrinsic_matrix, extrinsic_matrix, resolution):
    """
    Generates a depth image from a point cloud using camera intrinsic and extrinsic matrices.

    Args:
        ply_file (str): Path to the .ply file containing the point cloud.
        intrinsic_matrix (np.array): 3x3 camera intrinsic matrix.
        extrinsic_matrix (np.array): 4x4 camera extrinsic matrix.
        resolution (tuple): The (width, height) of the output depth image.

    Returns:
        depth_image (np.array): Depth image (2D array) from the camera's view.
    """
    # Load point cloud from the .ply file
    pcd = o3d.io.read_point_cloud(ply_file)
    points = np.asarray(pcd.points)

    # Convert 3D points to homogeneous coordinates (N x 4) by adding a column of 1s
    ones = np.ones((points.shape[0], 1))
    points_homog = np.hstack((points, ones))

    # Apply extrinsic matrix to transform points to the camera coordinate system
    camera_coords = (extrinsic_matrix @ points_homog.T).T  # Transformed 3D points in camera coordinates

    print("extrinsic_matrix",extrinsic_matrix)

    # Only keep points in front of the camera (positive z-coordinate in camera coordinates)
    camera_coords = camera_coords[camera_coords[:, 2] > 0]

    # Project the 3D points into the 2D image plane using the intrinsic matrix
    pixel_coords = (intrinsic_matrix @ camera_coords[:, :3].T).T

    # Normalize by the depth (Z-coordinate) to get pixel coordinates
    pixel_coords[:, 0] /= pixel_coords[:, 2]
    pixel_coords[:, 1] /= pixel_coords[:, 2]

    # Get image resolution (width, height)
    width, height = resolution

    # Initialize depth image (filled with infinity initially)
    depth_image = np.full((height, width), np.inf)

    # Loop through each point and update the depth image
    for i in range(pixel_coords.shape[0]):
        x = int(pixel_coords[i, 0])
        y = int(pixel_coords[i, 1])
        z = camera_coords[i, 2]

        # Ensure the coordinates are within image bounds
        if 0 <= x < width and 0 <= y < height:
            # Update the depth image with the closest depth (z-value)
            depth_image[y, x] = min(depth_image[y, x], z)

    # Replace infinity values with 0 for areas where no points were projected
    depth_image[depth_image == np.inf] = 0

    return depth_image

# Example usage
ply_file = str(DATA_FOLDER / 'camera/0/scene/scene.ply')
matrices_path = "/home/ja/Projects/Simulation/data/camera/0/camera_0002.npz"
resolution = (640, 480)  # Image resolution (width, height)

matrix_dic = read_camera_matrices_dic(matrices_path=matrices_path)

intrinsic_matrix = matrix_dic["intrinsic"]
extrinsic_matrix = matrix_dic["extrinsic"]

# Generate depth image
depth_image = generate_depth_image_from_ply(ply_file, intrinsic_matrix, extrinsic_matrix, resolution)

print("matrices_path",matrices_path)
cv2.imshow("result", (depth_image * 255 / np.max(depth_image)).astype(np.uint8))
cv2.waitKey(0)

# # Save or display the depth image (scaling for visualization)
# cv2.imwrite("depth_image.png", (depth_image * 255 / np.max(depth_image)).astype(np.uint8))
