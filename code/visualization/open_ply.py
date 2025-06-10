import open3d as o3d
from path_config import DATA_FOLDER, OUTPUT_FOLDER

def get_edge_values(pcd):
    # Get the axis-aligned bounding box of the point cloud
    bbox = pcd.get_axis_aligned_bounding_box()
    # Get the minimum and maximum bounds of the bounding box
    min_bound = bbox.min_bound
    max_bound = bbox.max_bound
    return min_bound, max_bound

# Path to the scene PLY file
scene_ply = str(DATA_FOLDER / "test/camera/110/scene/scene.ply")
# scene_ply = "/home/ja/Projects/Simulation/output/transformed_point_cloud.ply"

# Load the point cloud from the PLY file
pcd = o3d.io.read_point_cloud(scene_ply)

# Print basic information about the point cloud
print(pcd)

# Get and print the edge values (min and max bounds)
min_bound, max_bound = get_edge_values(pcd)
print(f"Min Bound: {min_bound}")
print(f"Max Bound: {max_bound}")

# Visualize the point cloud
o3d.visualization.draw_geometries([pcd])
