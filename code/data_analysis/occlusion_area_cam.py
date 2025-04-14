import numpy as np
import json
from pathlib import Path
import os
import sys
from tqdm import tqdm
import cv2  # Add OpenCV import
from scipy.spatial import ConvexHull
<<<<<<< HEAD
=======
from sklearn.neighbors import KDTree
from sklearn.decomposition import PCA
>>>>>>> 1d7eaa7 (update)
from path_config import DATA_FOLDER, CONFIG_FOLDER, CODE_FOLDER

sys.path.append(str(CODE_FOLDER))

from my_utils import read_camera_matrices


def load_point_cloud_with_labels(ply_file):
    """从 PLY 文件中加载点云数据及标签"""
    points = []
    labels = []
    with open(ply_file, 'r') as f:
        is_header = True
        for line in f:
            if is_header:
                if line.startswith("end_header"):
                    is_header = False
                continue
            # 解析每一行的数据
            x, y, z, label = line.strip().split()
            points.append([float(x), float(y), float(z)])
            labels.append(label)  # 保留标签的字符串形式
    return np.array(points), np.array(labels)


def project_points_to_image(points, intrinsic_matrix, extrinsic_matrix):
    """将 3D 点投影到 2D 图像平面上"""
    # Convert points to homogeneous coordinates
    num_points = points.shape[0]
    homogeneous_points = np.hstack((points, np.ones((num_points, 1))))

    # Apply extrinsic matrix (world to camera transformation)
    camera_coords = extrinsic_matrix @ homogeneous_points.T

    # Project points onto image plane using intrinsic matrix
    image_coords = intrinsic_matrix @ camera_coords[:3, :]

    # Normalize by the third row to get pixel coordinates
    image_coords /= image_coords[2, :]
    
    # Return both image coordinates and depth values (z-coordinate of camera space)
    return image_coords[:2, :].T, camera_coords[2, :]


def save_depth_image(depths, image_coords, image_size, file_path):
    """保存深度图像并展示"""
    # 确保目录存在
    file_path = Path(file_path)
    directory = file_path.parent
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)

    depth_image = np.full(image_size, 1e6)  # 初始化深度图像为大值，而不是无穷大

    for img_point, depth in zip(image_coords, depths):
        pixel_coords = np.round(img_point).astype(int)
        if 0 <= pixel_coords[0] < image_size[1] and 0 <= pixel_coords[1] < image_size[0]:
            depth_image[pixel_coords[1], pixel_coords[0]] = np.minimum(depth_image[pixel_coords[1], pixel_coords[0]], depth)

    # 将无效值（依然是大值）设置为 0 以避免影响归一化
    depth_image[depth_image == 1e6] = 0

    # 归一化深度值以适应图像显示（0 - 255）
    normalized_depth_image = cv2.normalize(depth_image, None, 0, 255, cv2.NORM_MINMAX)
    normalized_depth_image = np.uint8(normalized_depth_image)
    file_path = str(file_path)
    cv2.imwrite(file_path, normalized_depth_image)



<<<<<<< HEAD
def compute_occlusion_with_camera_params(points, labels, intrinsic_matrix, extrinsic_matrix, image_size=(640, 480)):
    """基于相机内参和外参计算物体之间的遮挡关系，忽略中心超出图像边界的物体"""
=======
def compute_occlusion_with_camera_params(points, labels, intrinsic_matrix, extrinsic_matrix, image_size=(640, 480), min_visible_ratio=0.1, min_occlusion_ratio=0.05):
    """基于相机内参和外参计算物体之间的遮挡关系，忽略中心超出图像边界的物体"""

>>>>>>> 1d7eaa7 (update)
    unique_labels = np.unique(labels)
    num_objects = len(unique_labels)
    occlusion_matrix = np.zeros((num_objects, num_objects), dtype=int)
    occlusion_areas = {label: {other_label: 0 for other_label in unique_labels} for label in unique_labels}

    # 将点云投影到图像平面
    image_coords, depths = project_points_to_image(points, intrinsic_matrix, extrinsic_matrix)

    # 用于存储物体中心的位置
    object_pixel_coords = {label: [] for label in unique_labels}

    # 收集每个标签的所有像素坐标
    for img_point, depth, label in zip(image_coords, depths, labels):
        pixel_coords = tuple(np.round(img_point).astype(int))  # Round to pixel coordinates
        object_pixel_coords[label].append(pixel_coords)

    # 计算每个标签的中心位置并检查其是否超出图像边界
    objects_out_of_bounds = set()
    for label, coords in object_pixel_coords.items():
        coords = np.array(coords)
        if len(coords) == 0:
            continue
        center_coords = np.mean(coords, axis=0).astype(int)  # Calculate the center

        # If the center is out of bounds, mark the object and skip occlusion calculations
        if not (0 <= center_coords[0] < image_size[0] and 0 <= center_coords[1] < image_size[1]):
            objects_out_of_bounds.add(label)  # Mark this label as out of bounds
            continue

    # 构建深度图，仅考虑图像内的物体
    depth_map = {}
    for img_point, depth, label in zip(image_coords, depths, labels):
        if label in objects_out_of_bounds:
            continue  # Skip points of objects that are out of bounds

        key = tuple(np.round(img_point).astype(int))  # Round to pixel coordinates
        if key not in depth_map:
            # First time this pixel is encountered, create a dictionary to store depths for each label
            depth_map[key] = {label: depth}
        else:
            # If the label is already present at this pixel, update it with the smaller depth
            if label in depth_map[key]:
                depth_map[key][label] = min(depth_map[key][label], depth)
            else:
                depth_map[key][label] = depth

    # 计算遮挡情况
    for key, label_depth_dict in depth_map.items():
        # Sort by depth (closer objects first)
        sorted_labels_depths = sorted(label_depth_dict.items(), key=lambda x: x[1])

        # Iterate over all pairs of objects at this pixel and mark occlusions
        for i in range(len(sorted_labels_depths)):
            occluding_label = sorted_labels_depths[i][0]  # The closer (occluding) object
            for j in range(i + 1, len(sorted_labels_depths)):
                occluded_by_label = sorted_labels_depths[j][0]  # The further (occluded) object

                # Skip occluded object if it's out of bounds
                if occluded_by_label in objects_out_of_bounds:
                    continue

                occluding_idx = np.where(unique_labels == occluding_label)[0][0]
                occluded_idx = np.where(unique_labels == occluded_by_label)[0][0]

                # Increment the occlusion area between these objects
                occlusion_areas[occluding_label][occluded_by_label] += 1

    # 初始化遮挡矩阵
    for occluding_label, occluded_dict in occlusion_areas.items():
        occluding_idx = np.where(unique_labels == occluding_label)[0][0]
        for occluded_by_label, area in occluded_dict.items():
            occluded_idx = np.where(unique_labels == occluded_by_label)[0][0]

            # If the object is out of bounds, set occlusion area to 0
            if occluded_by_label in objects_out_of_bounds or area < 20:
                occlusion_areas[occluding_label][occluded_by_label] = 0
            elif area > 0:
                occlusion_matrix[occluding_idx][occluded_idx] = 1

    # 对于所有超出图像范围的物体，将它们的遮挡矩阵设置为空列表
    for label in objects_out_of_bounds:
        occlusion_matrix[np.where(unique_labels == label)] = 0  # No occlusions for out-of-bounds objects

    # 返回更新后的遮挡矩阵
    return occlusion_matrix, unique_labels, occlusion_areas

<<<<<<< HEAD

=======
def compute_support_matrix(points, labels, unique_labels, 
                          eps_scale=0.2, min_contact_ratio=0.008):

    num_objects = len(unique_labels)
    support_matrix = np.zeros((num_objects, num_objects), dtype=int)
    obj_properties = compute_z_properties(points, labels, unique_labels)

    obj_top_points = {}
    obj_bottom_points = {}
    for label in unique_labels:
        points_l = points[labels == label]
        props = obj_properties[label]
        
        top_threshold = props['top'] - eps_scale * props['height']
        obj_top_points[label] = points_l[points_l[:, 2] >= top_threshold]
        
        bottom_threshold = props['bottom'] + eps_scale * props['height']
        obj_bottom_points[label] = points_l[points_l[:, 2] <= bottom_threshold]

    for i, label_a in enumerate(unique_labels):
        a_props = obj_properties[label_a]
        a_top = a_props['top']
        a_height = a_props['height']
        eps_a = eps_scale * a_height

        a_top_pts = obj_top_points[label_a]
        if len(a_top_pts) == 0:
            continue

        for j, label_b in enumerate(unique_labels):
            if i == j:
                continue

            b_props = obj_properties[label_b]
            b_bottom = b_props['bottom']
            b_height = b_props['height']

            eps_height = (a_height + b_height) * 0.5 * eps_scale
            if abs(a_top - b_bottom) > eps_height:
                continue

            b_bottom_pts = obj_bottom_points[label_b]
            if len(b_bottom_pts) == 0:
                continue

            min_samples = max(3, int(min_contact_ratio * min(
                len(a_top_pts), len(b_bottom_pts)
            )))

            tree = KDTree(b_bottom_pts[:, :2])
            distances, _ = tree.query(a_top_pts[:, :2], k=1)
            contact_count = np.sum(distances <= eps_height)

            if contact_count >= min_samples:
                support_matrix[j][i] = 1

    # 消除双向支撑冲突
    for i in range(num_objects):
        for j in range(num_objects):
            if i != j and support_matrix[i][j] and support_matrix[j][i]:
                a_avg_z = obj_properties[unique_labels[i]]['avg_z']
                b_avg_z = obj_properties[unique_labels[j]]['avg_z']
                if a_avg_z > b_avg_z:
                    support_matrix[j][i] = 0  # 保留i支撑j
                else:
                    support_matrix[i][j] = 0  # 保留j支撑i

    return support_matrix

def compute_z_properties(points, labels, unique_labels):
    """增强属性计算：增加平均高度"""
    properties = {}
    for label in unique_labels:
        obj_points = points[labels == label]
        z_values = obj_points[:, 2]
        properties[label] = {
            'top': np.max(z_values),
            'bottom': np.min(z_values),
            'height': np.max(z_values) - np.min(z_values),
            'avg_z': np.mean(z_values)  # 用于冲突解决
        }
    return properties
>>>>>>> 1d7eaa7 (update)

def build_occlusion_tree(occlusion_matrix, unique_labels):
    """构建遮挡树形结构"""
    num_objects = len(unique_labels)

    # 将标签转换为字符串，并初始化树结构
    tree = {label: [] for label in unique_labels}

    for i in range(num_objects):
        for j in range(num_objects):
            if occlusion_matrix[i, j] > 0:
                tree[unique_labels[i]].append(unique_labels[j])

    return tree


def save_occlusion_tree_to_file(tree, occlusion_areas, file_path):
    """将遮挡树形结构及每对物体的遮挡区域面积保存为 JSON 文件，若文件或目录不存在则创建"""
    
    # 确保目录存在
    file_path = Path(file_path)
    directory = file_path.parent
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    
    # 保存文件
    data = {
        "occlusion_tree": tree,
        "occlusion_areas": occlusion_areas
    }
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=4)


def save_object_positions(image_coords, labels, file_path):
    """保存每个物体在深度图像中的中心像素位置及其标签"""
    # 确保目录存在
    file_path = Path(file_path)
    directory = file_path.parent
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    
    # 创建字典以保存每个标签的中心像素位置
    object_positions = {}
    object_pixel_coords = {}

    # 收集每个标签的所有像素坐标
    for img_point, label in zip(image_coords, labels):
        pixel_coords = tuple(map(int, np.round(img_point).astype(int)))  # Convert to Python native int
        label = str(label)  # Ensure label is a string
        if label not in object_pixel_coords:
            object_pixel_coords[label] = []
        object_pixel_coords[label].append(pixel_coords)
    
    # 计算每个标签的中心像素位置
    for label, coords in object_pixel_coords.items():
        coords = np.array(coords)
        center_coords = np.mean(coords, axis=0)  # Calculate the mean (center)
        center_coords = tuple(map(int, center_coords))  # Convert to native Python int
        x_min, y_min = np.min(coords, axis=0)
        x_max, y_max = np.max(coords, axis=0)
        width = x_max - x_min
        height = y_max - y_min
        object_positions[label] = {
            "center": tuple(map(int, center_coords)),
            "width": int(width),
            "height": int(height),
            "x_min": int(x_min),
            "x_max": int(x_max),
            "y_min": int(y_min),
            "y_max": int(y_max)
    }  # Store as a tuple for JSON compatibility
    
    # 保存中心位置数据到 JSON 文件
    with open(file_path, 'w') as f:
        json.dump(object_positions, f, indent=4)

def save_object_segmentation(image_coords, labels, file_path, image_shape):
    """
    Save polygon and mask for each object in the image.

    Parameters:
        image_coords (list): List of coordinates of each object in the image.
        labels (list): List of corresponding labels for each object.
        file_path (str): Path where the segmentation JSON file will be saved.
        image_shape (tuple): Shape of the image (height, width).
    """
    # Initialize the output dictionary
    segmentation_data = {}
    object_pixel_coords = {}

    # Create a list to store the polygons and masks
    objects_data = []
    
    # 收集每个标签的所有像素坐标
    for img_point, label in zip(image_coords, labels):
        pixel_coords = tuple(map(int, np.round(img_point).astype(int)))  # Convert to Python native int
        label = str(label)  # Ensure label is a string
        if label not in object_pixel_coords:
            object_pixel_coords[label] = []
        object_pixel_coords[label].append(pixel_coords)

    for label, coords in object_pixel_coords.items():
        # Initialize the mask for each object (background is 0, object is 255)
        mask = np.zeros(image_shape, dtype=np.uint8)
        
        # Fill the polygon area with 255 (object) based on the provided coordinates
        coords = np.array(coords, dtype=np.int32)
        coords = coords.reshape((-1, 1, 2))  # Ensure it's in the correct format for fillPoly
        cv2.fillPoly(mask, [coords], 255)

        # Ensure the mask is 2D and of the correct dtype (uint8)
        if len(mask.shape) != 2:
            mask = mask[:, :, 0]  # If it's not 2D, squeeze it to make it 2D
        if mask.dtype != np.uint8:
            mask = mask.astype(np.uint8)  # Ensure it's uint8 type (values 0-255)
        
        # Extract polygon points from the mask using findContours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # In case of multiple contours, we can select the largest or deal with them as needed
        polygon_points = []
        for contour in contours:
            contour = contour.squeeze()  # Remove unnecessary dimensions (contour is a 3D array)
            polygon_points.append(contour.tolist())

        # Prepare the object data

        object_data = {
            "label": label,
            "mask": mask.tolist(),  # Store the binary mask as a list of lists
            "polygon": polygon_points
        }

        # Append the object data to the list
        objects_data.append(object_data)

    # Save the segmentation data to a JSON file
    segmentation_data["objects"] = objects_data

    # Ensure the output directory exists
    file_path = Path(file_path)
    directory = file_path.parent
    if not directory.exists():
        directory.mkdir(parents=True, exist_ok=True)
    
    with open(file_path, 'w') as f:
        json.dump(segmentation_data, f, indent=4)

<<<<<<< HEAD

=======
def combine_relations(support_matrix, occlusion_matrix):
    """
    结合支撑矩阵和遮挡矩阵生成最终关系矩阵。
    """
    # 初始化关系矩阵为遮挡矩阵的副本
    relation_matrix = np.copy(occlusion_matrix)
    
    # 遍历所有可能的物体对
    n = support_matrix.shape[0]
    for i in range(n):
        for j in range(n):
            # 处理支撑关系
            if support_matrix[i, j] == 1:
                # 强制设置i在j下方
                relation_matrix[i, j] = 1
                # 禁止反向关系（j不可能在i下方）
                relation_matrix[j, i] = 0
    
    return relation_matrix.astype(int)
>>>>>>> 1d7eaa7 (update)

def generate_occlusion_cam(config):
    # 主循环
    for i in tqdm(range(config['start_from'],config['end_with']), desc="Occlude Cameras"):
        # 加载点云数据和标签
        try:
            ROOT_FOLDER = DATA_FOLDER / 'test' # for test only
            ply_file = ROOT_FOLDER / f'camera/{i}/scene/scene.ply'
            points, labels = load_point_cloud_with_labels(ply_file)
<<<<<<< HEAD

=======
>>>>>>> 1d7eaa7 (update)
            for j in tqdm(range(config['num_frames']), desc=f"Processing Scenes in Camera {i}", leave = False):
                image_path = os.path.join(DATA_FOLDER, f'test/camera/{i}/rgb_{j:04d}.png')
            
                # Load image to get actual dimensions
                image = cv2.imread(image_path)

                matrices_path = Path(ROOT_FOLDER) / f'camera/{i}/camera_{j:04d}.npz'
                matrices_dic = read_camera_matrices.read_camera_matrices_dic(matrices_path)

                # 将点云投影到图像平面，并获取深度
                image_coords, depths = project_points_to_image(points, matrices_dic["intrinsic"], matrices_dic["extrinsic"])

                # 计算物体间的遮挡关系及遮挡面积
                occlusion_matrix, unique_labels, occlusion_areas = compute_occlusion_with_camera_params(points, labels, matrices_dic["intrinsic"], matrices_dic["extrinsic"])

<<<<<<< HEAD
                # 构建遮挡树
                occlusion_tree = build_occlusion_tree(occlusion_matrix, unique_labels)
=======
                # 支撑矩阵（接触面检测）
                support_matrix = compute_support_matrix(points, labels, unique_labels)

                relation_matrix = combine_relations(support_matrix, occlusion_matrix)
                print("support_matrix:")
                print(support_matrix)
                print("occlusion_matrix:")
                print(occlusion_matrix)
                print("final_matrix:")
                print(relation_matrix)
                print("1")

                # 构建遮挡树
                occlusion_tree = build_occlusion_tree(relation_matrix, unique_labels)
>>>>>>> 1d7eaa7 (update)

                # 保存遮挡树和遮挡区域面积到文件
                output_file = ROOT_FOLDER / f'camera/{i}/scene/occlusion/occlusion_tree_{j:04d}.json'
                save_occlusion_tree_to_file(occlusion_tree, occlusion_areas, output_file)

                # 保存物体在深度图像中的位置
                position_file = ROOT_FOLDER / f'camera/{i}/scene/position/position_{j:04d}.json'
                save_object_positions(image_coords, labels, position_file)

                # 保存物体的分割数据
                segmentation_file = ROOT_FOLDER / f'camera/{i}/scene/segmentation/segmentation_{j:04d}.json'
                save_object_segmentation(image_coords, labels, segmentation_file, image.shape)

                # 保存深度图像并显示 (可选)
                depth_file = ROOT_FOLDER / f'camera/{i}/scene/depth/depth_image_{j:04d}.png'
<<<<<<< HEAD
                save_depth_image(depths, image_coords, image.shape, depth_file)
=======
                #save_depth_image(depths, image_coords, image.shape, depth_file)
>>>>>>> 1d7eaa7 (update)
        except:
            print(f"file {i} dose not exist.")
            continue