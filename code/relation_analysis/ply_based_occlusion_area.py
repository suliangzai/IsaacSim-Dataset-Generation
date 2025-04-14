import numpy as np
import json
from pathlib import Path
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
    """计算物体之间的遮挡关系及每对物体的遮挡区域"""
    unique_labels = np.unique(labels)
    num_objects = len(unique_labels)
    occlusion_matrix = np.zeros((num_objects, num_objects), dtype=int)
    occlusion_areas = {label: {other_label: 0 for other_label in unique_labels} for label in unique_labels}
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

    # 判断物体的遮挡情况并记录每对物体的遮挡面积
    for i, label in enumerate(unique_labels):
        object_points = points[labels == label]
        for point in object_points:
            x, y, z = point
            key = (round(x, 2), round(y, 2))
            if key in depth_map and depth_map[key][1] != label:
                if z < depth_map[key][0]:
                    occluded_by_label = depth_map[key][1]
                    j = np.where(unique_labels == occluded_by_label)[0][0]
                    occlusion_matrix[i][j] = 1
                    occlusion_areas[label][occluded_by_label] += 1  # 增加物体对之间的遮挡面积

    # # 计算遮挡区域面积（如 0.01 单位代表面积，则简单计数即可）
    # for label in occlusion_areas:
    #     for other_label in occlusion_areas[label]:
    #         occlusion_areas[label][other_label] *= 0.01 * 0.01  # 将计数转换为面积

    return occlusion_matrix, unique_labels, occlusion_areas


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
    print(f"遮挡关系树形结构及物体对的遮挡区域面积已保存至: {file_path}")


# 加载点云数据和标签
ply_file = DATA_FOLDER / 'camera/0/scene/scene.ply'
points, labels = load_point_cloud_with_labels(ply_file)

# 计算物体间的遮挡关系及遮挡面积
occlusion_matrix, unique_labels, occlusion_areas = compute_object_occlusion_relationship(points, labels)

# 构建遮挡树
occlusion_tree = build_occlusion_tree(occlusion_matrix, unique_labels)

# 保存遮挡树和遮挡区域面积到文件
output_file = DATA_FOLDER / 'camera/0/scene/occlusion/occlusion_tree.json'
save_occlusion_tree_to_file(occlusion_tree, occlusion_areas, output_file)
