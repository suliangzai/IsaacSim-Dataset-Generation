import numpy as np
import json
from path_config import DATA_FOLDER, OUTPUT_FOLDER


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


def compute_object_occlusion_relationship(points, labels):
    """计算物体之间的遮挡关系，以物体为单位"""
    unique_labels = np.unique(labels)
    num_objects = len(unique_labels)
    occlusion_matrix = np.zeros((num_objects, num_objects), dtype=int)
    depth_map = {}

    # 构建深度图
    for point, label in zip(points, labels):
        x, y, z = point
        key = (round(x, 2), round(y, 2))  # 将 (x, y) 四舍五入到 0.01 的精度
        if key not in depth_map:
            depth_map[key] = (z, label)
        else:
            if z > depth_map[key][0]:
                depth_map[key] = (z, label)

    # 判断物体的遮挡情况
    for i, label in enumerate(unique_labels):
        # print("unique_labels",unique_labels)
        # print("label",label)
        object_points = points[labels == label]
        is_occluded = False
        for point in object_points:
            x, y, z = point
            key = (round(x, 2), round(y, 2))
            if key in depth_map and depth_map[key][1] != label:
                if z < depth_map[key][0]:
                    is_occluded = True
                    occluded_by_label = depth_map[key][1]
                    j = np.where(unique_labels == occluded_by_label)[0][0]
                    occlusion_matrix[i][j] = 1

    return occlusion_matrix, unique_labels


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


def save_occlusion_tree_to_file(tree, file_path):
    """将遮挡树形结构保存为 JSON 文件"""
    with open(file_path, 'w') as f:
        json.dump(tree, f, indent=4)
    print(f"遮挡关系树形结构已保存至: {file_path}")


# 加载点云数据和标签
ply_file = DATA_FOLDER / 'scene/ply/points.ply'
points, labels = load_point_cloud_with_labels(ply_file)

# 计算物体间的遮挡关系
occlusion_matrix, unique_labels = compute_object_occlusion_relationship(
    points, labels)

# 构建遮挡树
occlusion_tree = build_occlusion_tree(occlusion_matrix, unique_labels)

# 保存遮挡树到文件
output_file = DATA_FOLDER / 'scene/occlusion/occlusion_tree.json'
save_occlusion_tree_to_file(occlusion_tree, output_file)
