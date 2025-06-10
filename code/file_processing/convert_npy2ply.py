import os
import numpy as np
from plyfile import PlyData, PlyElement
from path_config import DATA_FOLDER, OUTPUT_FOLDER

def npy_to_ply(npy_file, ply_file):
    """
    将Isaac Sim生成的点云 .npy 文件转换为 .ply 文件.
    :param npy_file: 输入的 .npy 文件路径
    :param ply_file: 输出的 .ply 文件路径
    """
    # 加载 .npy 文件
    data = np.load(npy_file)

    # 检查数据格式是否为 (N, 3) 或 (N, 6)
    if data.shape[1] == 3:
        # 只有点云坐标信息 (x, y, z)
        points = np.zeros(data.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4')])
        points['x'] = data[:, 0]
        points['y'] = data[:, 1]
        points['z'] = data[:, 2]
    elif data.shape[1] == 6:
        # 包含点云坐标和颜色信息 (x, y, z, r, g, b)
        points = np.zeros(data.shape[0], dtype=[('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
                                                ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')])
        points['x'] = data[:, 0]
        points['y'] = data[:, 1]
        points['z'] = data[:, 2]
        points['red'] = data[:, 3]
        points['green'] = data[:, 4]
        points['blue'] = data[:, 5]
    else:
        raise ValueError(f"{npy_file} 格式错误, 预期为 (N, 3) 或 (N, 6)，实际为 {data.shape}")

    # 创建 ply 文件
    ply_element = PlyElement.describe(points, 'vertex')
    PlyData([ply_element], text=True).write(ply_file)
    print(f"成功将 {npy_file} 转换为 {ply_file}")


def convert_all_npy_to_ply(folder_path):
    """
    将文件夹中所有 .npy 文件转换为 .ply 文件.
    :param folder_path: 包含 .npy 文件的文件夹路径
    """
    output_folder = os.path.join(folder_path, '../ply')
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.npy'):
            npy_file = os.path.join(folder_path, file_name)
            ply_file = os.path.join(output_folder, file_name.replace('.npy', '.ply'))
            try:
                npy_to_ply(npy_file, ply_file)
            except Exception as e:
                print(f"转换 {npy_file} 时发生错误: {e}")


folder_path = DATA_FOLDER / "camera/npy"  # 替换为你的文件夹路径
convert_all_npy_to_ply(folder_path)
