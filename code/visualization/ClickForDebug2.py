import cv2
import numpy as np
from pathlib import Path
import os
import json
from tqdm import tqdm

# 定义输入和输出文件夹路径
script_dir = Path(os.path.dirname(os.path.abspath(__file__)))
rgb_folder = script_dir / '../data/RGB'
mask_folder = script_dir / '../data/masks'  # 替换为你的mask文件夹路径
depth_folder = script_dir / '../data/depth'  # 替换为你的深度图文件夹路径
result_folder = script_dir / '../data/occlusion'

# 遮挡关系文件路径
occlusion_tree_file = script_dir / '../data/info/occlusion_tree.json'
instance_segmentation_mapping_file = script_dir / \
    '../data/info/instance_segmentation_mapping_0000.json'

# 获取mask文件夹中所有的mask图片
mask_files = sorted(mask_folder.glob('*.png'))  # 假设mask文件为.png格式

# 读取遮挡关系和实例分割映射文件
with open(occlusion_tree_file, 'r') as f:
    occlusion_tree = json.load(f)

with open(instance_segmentation_mapping_file, 'r') as f:
    instance_segmentation_mapping = json.load(f)

# 将颜色映射转换为物体名称的字典
color_to_object = {}
for color_str, object_path in instance_segmentation_mapping.items():
    # 将字符串表示的颜色转换为tuple格式
    color_tuple = tuple(map(int, color_str.strip('()').split(',')))
    # 获取物体的名称
    object_name = object_path.split('/')[-1]
    color_to_object[color_tuple[:3]] = object_name  # 只使用前3个值（RGB）

# 获取物体名称到颜色的映射
object_to_color = {v: k for k, v in color_to_object.items()}

# 定义全局变量用于鼠标点击交互和目标切换
clicked_color = None
clicked_mask = None
clicked_adjacent_mask = None
occluding_mask = None
occluded_by_target_mask = None
unoccluded_mask = None
depth_img_global = None
current_index = 0  # 当前图像索引
current_target_index = 0  # 当前目标物体的索引
target_object_list = list(object_to_color.keys())  # 目标物体列表
show_unoccluded = False  # 标志是否显示未遮挡物体


def update_target_mask(target_object_name, mask_img_rgb):
    """
    更新与目标物体相关的遮挡和被遮挡信息
    """
    global clicked_mask, clicked_adjacent_mask, occluding_mask, occluded_by_target_mask

    # 更新目标物体的mask
    clicked_color = object_to_color.get(target_object_name)
    clicked_mask = np.all(mask_img_rgb == clicked_color, axis=-1)

    # 获取遮挡目标的物体列表（目标被哪些物体遮挡）
    occluding_objects = occlusion_tree.get(target_object_name, [])

    # 初始化相邻物体、遮挡物体和被目标遮挡物体的mask
    clicked_adjacent_mask = np.zeros(clicked_mask.shape, dtype=np.uint8)
    occluding_mask = np.zeros(clicked_mask.shape, dtype=np.uint8)
    occluded_by_target_mask = np.zeros(clicked_mask.shape, dtype=np.uint8)

    # 查找遮挡物体和相邻物体的mask
    for color_tuple, object_name in color_to_object.items():
        # 跳过自身
        if object_name == target_object_name:
            continue

        # 创建相邻物体的mask
        adjacent_object_mask = np.all(mask_img_rgb == color_tuple, axis=-1)

        # 判断是否是遮挡物
        if object_name in occluding_objects:
            occluding_mask = np.logical_or(
                occluding_mask, adjacent_object_mask)
        else:
            clicked_adjacent_mask = np.logical_or(
                clicked_adjacent_mask, adjacent_object_mask)

    # 查找目标物体遮挡的其他物体
    for object_name, occluded_objects in occlusion_tree.items():
        if target_object_name in occluded_objects:
            # 获取被目标遮挡的物体的mask
            for color_tuple, obj_name in color_to_object.items():
                if obj_name == object_name:
                    occluded_by_target_mask = np.logical_or(
                        occluded_by_target_mask, np.all(mask_img_rgb == color_tuple, axis=-1))

    # 打印遮挡关系信息
    if occluding_objects:
        print(f"遮挡目标的物体: {', '.join(occluding_objects)}")
    occluded_names = [color_to_object[color] for color in color_to_object if np.any(
        np.all(mask_img_rgb == color, axis=-1) & occluded_by_target_mask)]
    if occluded_names:
        print(f"被目标遮挡的物体: {', '.join(occluded_names)}")


def show_masks_on_click(event, x, y, flags, param):
    """
    鼠标点击事件回调函数
    event: 事件类型
    x, y: 点击的坐标
    flags: 事件标志
    param: 传递的参数（包括mask图像、RGB图像和深度图）
    """
    global clicked_color, depth_img_global, current_target_index, show_unoccluded

    if event == cv2.EVENT_LBUTTONDOWN:
        mask_img = param['mask_img']
        depth_img = param['depth_img']
        kernel = np.ones((3, 3), np.uint8)

        # 将掩码图像从BGR格式转换为RGB格式
        mask_img_rgb = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)

        # 获取点击位置的掩码颜色
        clicked_color = tuple(mask_img_rgb[y, x].tolist())

        # 如果点击的是背景（黑色或未标注区域），则忽略
        if clicked_color not in color_to_object or color_to_object[clicked_color] in ["BACKGROUND", "UNLABELLED"]:
            print(f"点击的是背景或未标注区域: {clicked_color}")
            return

        # 获取点击物体的名称并更新当前目标索引
        clicked_object_name = color_to_object.get(clicked_color, None)
        if clicked_object_name is None:
            print(f"未找到点击物体的名称: {clicked_color}")
            return

        print(f"点击的物体名称: {clicked_object_name}")
        current_target_index = target_object_list.index(clicked_object_name)

        # 更新目标相关mask信息
        update_target_mask(clicked_object_name, mask_img_rgb)

        # 更新全局深度图像
        depth_img_global = depth_img.copy()

        # 每次点击后取消未遮挡物体的显示
        show_unoccluded = False


def show_unoccluded_objects(mask_img_rgb):
    """
    显示所有未被遮挡的物体，只保留白色掩码和背景
    """
    global unoccluded_mask
    unoccluded_mask = np.zeros(mask_img_rgb.shape[:2], dtype=np.uint8)

    for object_name, color_tuple in object_to_color.items():
        if object_name in ["BACKGROUND", "UNLABELLED"]:
            continue
        occluding_objects = occlusion_tree.get(object_name, [])
        if not occluding_objects:  # 没有遮挡物，说明是未被遮挡的物体
            object_mask = np.all(mask_img_rgb == color_tuple, axis=-1)
            unoccluded_mask = np.logical_or(unoccluded_mask, object_mask)

    print("显示未被遮挡的物体")


def reset_display(mask_img):
    """
    重置显示的图像，防止颜色重影
    """
    return np.zeros(mask_img.shape, dtype=np.uint8)


# 主循环控制变量
total_images = len(mask_files)
exit_program = False

while not exit_program and total_images > 0:
    # 获取当前图像的文件路径
    mask_file = mask_files[current_index]
    depth_file = depth_folder / mask_file.name
    rgb_file = rgb_folder / mask_file.name
    result_file = result_folder / mask_file.name

    # 确保对应的深度图文件存在
    if not depth_file.exists():
        print(f"深度图 {depth_file} 不存在，跳过该文件。")
        # 跳过不存在的图像，继续下一个
        current_index = (current_index + 1) % total_images
        continue

    # 读取mask图像和深度图
    mask_img = cv2.imread(str(mask_file))
    depth_img = cv2.imread(str(depth_file), cv2.IMREAD_UNCHANGED)  # 读取深度图
    rgb_img = cv2.imread(str(rgb_file))
    result_img = cv2.imread(str(result_file))

    # 创建窗口并显示原始彩色mask图像
    cv2.namedWindow("RGB Image")
    cv2.imshow("RGB Image", rgb_img)

    cv2.namedWindow("Result")
    cv2.imshow("Result", result_img)

    cv2.namedWindow("Mask Image")
    cv2.imshow("Mask Image", mask_img)

    cv2.namedWindow("Clicked Object and Adjacent")
    cv2.imshow("Clicked Object and Adjacent", mask_img)

    # 设置窗口的田字形排列
    # 获取屏幕宽度和高度
    screen_width, screen_height = 1920, 1080  # 假设屏幕分辨率为1920x1080

    window_width = screen_width // 2
    window_height = screen_height // 2

    cv2.moveWindow("RGB Image", 0, 0)  # 左上角
    cv2.moveWindow("Mask Image", window_width, 0)  # 右上角
    cv2.moveWindow("Result", 0, window_height)  # 左下角
    cv2.moveWindow("Clicked Object and Adjacent",
                   window_width, window_height)  # 右下角

    # 设置鼠标点击事件回调函数，绑定到RGB Image窗口
    cv2.setMouseCallback("RGB Image", show_masks_on_click, {
        'rgb_img': rgb_img, 'mask_img': mask_img, 'depth_img': depth_img})

    # 将掩码图像从BGR格式转换为RGB格式
    mask_img_rgb = cv2.cvtColor(mask_img, cv2.COLOR_BGR2RGB)

    # 设置目标物体
    current_target_name = target_object_list[current_target_index]
    update_target_mask(current_target_name, mask_img_rgb)

    while True:
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            # 退出整个程序
            exit_program = True
            break
        elif key == ord('n'):
            # 下一张图片
            current_index = (current_index + 1) % total_images
            break
        elif key == ord('p'):
            # 上一张图片
            current_index = (current_index - 1 + total_images) % total_images
            break
        elif key == ord('a'):
            # 切换到上一个目标物体
            current_target_index = (
                current_target_index - 1) % len(target_object_list)
            current_target_name = target_object_list[current_target_index]
            print(f"切换到目标物体: {current_target_name}")
            update_target_mask(current_target_name, mask_img_rgb)
            show_unoccluded = False  # 取消未遮挡物体的显示
        elif key == ord('d'):
            # 切换到下一个目标物体
            current_target_index = (
                current_target_index + 1) % len(target_object_list)
            current_target_name = target_object_list[current_target_index]
            print(f"切换到目标物体: {current_target_name}")
            update_target_mask(current_target_name, mask_img_rgb)
            show_unoccluded = False  # 取消未遮挡物体的显示
        elif key == ord('u'):
            # 显示所有未被遮挡的物体
            show_unoccluded_objects(mask_img_rgb)
            show_unoccluded = True

        # 初始化显示图像，防止重影
        combined_display = reset_display(mask_img)

        # 显示相关物体的mask
        if show_unoccluded and unoccluded_mask is not None:
            # 只显示未被遮挡的物体和背景色
            unoccluded_display = np.zeros(mask_img.shape, dtype=np.uint8)
            unoccluded_display[unoccluded_mask] = [255, 255, 255]
            combined_display = cv2.addWeighted(
                combined_display, 1.0, unoccluded_display, 1.0, 0)
        else:
            # 将点击物体的mask显示为绿色
            if clicked_mask is not None:
                clicked_object_display = np.zeros(
                    mask_img.shape, dtype=np.uint8)
                clicked_object_display[clicked_mask] = [0, 255, 0]

                # 将相邻非遮挡物体的mask显示为蓝色
                adjacent_object_display = np.zeros(
                    mask_img.shape, dtype=np.uint8)
                adjacent_object_display[clicked_adjacent_mask] = [255, 0, 0]

                # 将遮挡物体的mask显示为红色
                occluding_object_display = np.zeros(
                    mask_img.shape, dtype=np.uint8)
                occluding_object_display[occluding_mask] = [0, 0, 255]

                # 将被目标遮挡的物体mask显示为黄色
                occluded_by_target_display = np.zeros(
                    mask_img.shape, dtype=np.uint8)
                occluded_by_target_display[occluded_by_target_mask] = [
                    0, 255, 255]

                # 叠加显示四种mask
                combined_display = cv2.addWeighted(
                    clicked_object_display, 0.5, adjacent_object_display, 0.5, 0)
                combined_display = cv2.addWeighted(
                    combined_display, 0.7, occluding_object_display, 0.3, 0)
                combined_display = cv2.addWeighted(
                    combined_display, 0.9, occluded_by_target_display, 0.1, 0)


        cv2.imshow("Clicked Object and Adjacent", combined_display)

    # 释放当前窗口
    cv2.destroyAllWindows()

# 清除全局变量
clicked_color = None
clicked_mask = None
clicked_adjacent_mask = None
occluding_mask = None
occluded_by_target_mask = None
unoccluded_mask = None
depth_img_global = None
show_unoccluded = False

print("程序已退出")
