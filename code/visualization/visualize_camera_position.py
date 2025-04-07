import numpy as np
import open3d as o3d
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

import sys
from path_config import DATA_FOLDER, CONFIG_FOLDER, CODE_FOLDER
sys.path.append(str(CODE_FOLDER))
from my_utils.read_camera_matrices import read_camera_matrices_dic

def plot_camera_in_world(extrinsic_matrix, ax, camera_size=1, color='blue'):
    """ Visualize a camera's position and orientation in world coordinates. """
    # Extract the position (translation part) from the extrinsic matrix
    position = extrinsic_matrix[:3, 3]
    
    # The rotation matrix is the top-left 3x3 part of the extrinsic matrix
    rotation_matrix = extrinsic_matrix[:3, :3]
    
    # Define the camera's local frame (axes directions: forward, right, up)
    # These represent a unit frame for the camera (scaled by camera_size)
    camera_axes = np.array([[0, 0, -camera_size],  # Forward (z direction)
                            [camera_size, 0, 0],   # Right (x direction)
                            [0, camera_size, 0]])  # Up (y direction)

    # Transform the camera's local axes by the rotation matrix
    world_axes = rotation_matrix @ camera_axes.T

    # Plot camera position
    ax.scatter(position[0], position[1], position[2], color=color, s=50, label='Camera')

    # Plot the camera's orientation axes
    ax.quiver(position[0], position[1], position[2], 
              world_axes[0, 0], world_axes[1, 0], world_axes[2, 0], color='red', length=camera_size, label='Forward')
    ax.quiver(position[0], position[1], position[2], 
              world_axes[0, 1], world_axes[1, 1], world_axes[2, 1], color='green', length=camera_size, label='Right')
    ax.quiver(position[0], position[1], position[2], 
              world_axes[0, 2], world_axes[1, 2], world_axes[2, 2], color='blue', length=camera_size, label='Up')

def visualize_camera_and_pointcloud(extrinsic_matrices, point_cloud):
    """ Visualize camera positions and a point cloud in the world. """
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    
    # Set axis labels
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')

    # Plot the point cloud
    point_cloud_np = np.asarray(point_cloud.points)
    ax.scatter(point_cloud_np[:, 0], point_cloud_np[:, 1], point_cloud_np[:, 2], s=1, c='gray', label='Point Cloud')

    # Plot each camera in the world coordinate system
    for i, extrinsic_matrix in enumerate(extrinsic_matrices):
        plot_camera_in_world(extrinsic_matrix, ax, color=np.random.rand(3,))
    
    # Set equal scaling for all axes
    ax.set_box_aspect([1, 1, 1])
    
    plt.legend()
    plt.show()

# 读取相机外参矩阵
matrices_path = "/home/ja/Projects/Simulation/data/camera/0/camera_0001.npz"
matrices_dic = read_camera_matrices_dic(matrices_path)
extrinsic_matrix = matrices_dic["extrinsic"]
print("extrinsic_matrix", extrinsic_matrix)

# 读取点云数据
ply_file = str(DATA_FOLDER / "camera/0/scene/scene.ply")
point_cloud = o3d.io.read_point_cloud(ply_file)
print(f"Loaded point cloud with {len(point_cloud.points)} points")

# 可视化相机位置和点云
visualize_camera_and_pointcloud([extrinsic_matrix], point_cloud)
