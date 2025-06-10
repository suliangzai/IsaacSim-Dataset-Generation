import open3d as o3d
import numpy as np
import cv2
import os
from path_config import DATA_FOLDER, OUTPUT_FOLDER

# Set the depth and ply directories
depth_dir = DATA_FOLDER / 'camera/depth'
ply_dir = DATA_FOLDER / 'camera/ply'

# Create the ply folder if it doesn't exist
os.makedirs(ply_dir, exist_ok=True)

# Load the camera intrinsic parameters from cam_K.txt
intrinsic_file_path = DATA_FOLDER / 'camera/cam_K.txt'
with open(intrinsic_file_path, 'r') as f:
    intrinsic_lines = f.readlines()
    intrinsic_matrix = np.array([float(value) for line in intrinsic_lines for value in line.split()]).reshape(3, 3)

# Extract focal lengths and principal point from the intrinsic matrix
fx = intrinsic_matrix[0, 0]  # Focal length in x-axis
fy = intrinsic_matrix[1, 1]  # Focal length in y-axis
cx = intrinsic_matrix[0, 2]  # Principal point in x-axis
cy = intrinsic_matrix[1, 2]  # Principal point in y-axis

scaling_factor = 1000.0  # Scaling factor for depth values (depends on your depth sensor)

# Iterate over all PNG files in the depth directory
for depth_file in os.listdir(depth_dir):
    if depth_file.endswith('.png'):
        # Construct the full path to the depth image and the corresponding PLY output
        depth_image_path = os.path.join(depth_dir, depth_file)
        ply_output_path = os.path.join(ply_dir, os.path.splitext(depth_file)[0] + '.ply')

        # Load depth image (in PNG format)
        depth_image = cv2.imread(depth_image_path, cv2.IMREAD_UNCHANGED)  # Load 16-bit PNG depth image

        # Check if the image was loaded correctly
        if depth_image is None:
            print(f"Error: Unable to load depth image at {depth_image_path}")
            continue

        # Get image dimensions
        height, width = depth_image.shape

        # Create a point cloud from the depth image
        points = []
        for v in range(height):
            for u in range(width):
                z = depth_image[v, u] / scaling_factor  # Depth value (in meters)
                if z == 0:  # Skip if no depth value
                    continue
                x = (u - cx) * z / fx
                y = (v - cy) * z / fy
                points.append([x, y, z])

        # Convert to Open3D point cloud format
        points = np.array(points)
        point_cloud = o3d.geometry.PointCloud()
        point_cloud.points = o3d.utility.Vector3dVector(points)

        # Save point cloud to PLY file
        o3d.io.write_point_cloud(ply_output_path, point_cloud)

        print(f"Point cloud saved to {ply_output_path}")

# Optionally visualize one of the point clouds (first one for example)
if len(os.listdir(ply_dir)) > 0:
    first_ply_path = os.path.join(ply_dir, os.listdir(ply_dir)[0])
    pcd = o3d.io.read_point_cloud(first_ply_path)
    o3d.visualization.draw_geometries([pcd], window_name='Point Cloud Visualization')
