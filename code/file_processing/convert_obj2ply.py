import trimesh
import numpy as np
from path_config import DATA_FOLDER, OUTPUT_FOLDER

# Load the OBJ file using trimesh
mesh = DATA_FOLDER / 'mesh/obj/test.obj'
output_file = DATA_FOLDER / 'mesh/ply/test.ply'

# Sample points from the mesh surface (e.g., 10000 points)
# You can adjust the number of points as needed
num_points = 10000
points, _ = trimesh.sample.sample_surface(mesh, num_points)

# Save the point cloud to a file (e.g., PLY format)
trimesh.points.PointCloud(points).export(output_file)

print(f"Point cloud saved to {output_file}")
