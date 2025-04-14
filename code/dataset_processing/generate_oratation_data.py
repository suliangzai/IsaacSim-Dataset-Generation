import os
import json
from PIL import Image
from tqdm import tqdm
from path_config import DATA_FOLDER, CONFIG_FOLDER, CODE_FOLDER

# 设置父目录和 JSON 文件路径
parent_directory = os.path.join(DATA_FOLDER, "test/camera/")
json_file_path = os.path.join(parent_directory, "dataset.json")
new_json_file_path = os.path.join(parent_directory, "dataset_extended.json")  # 新的 JSON 文件

# 读取原始 JSON 文件
with open(json_file_path, "r") as f:
    data = json.load(f)

# 新的 JSON 数据存储
new_data = []
counter= 0
# 遍历所有子目录和文件
for root, dirs, files in tqdm(os.walk(parent_directory)):
    print(counter)
    counter += 1
    subdir_name = os.path.basename(root)  # 获取子目录名
    for i in range(10):
        original_filename = f"rgb_000{i}.png"
        if original_filename in files:
            original_path = os.path.join(root, original_filename)
            
            # 查找原始 JSON 中的对应条目
            original_item = next((item for item in data if item.get("image_path") == f"test/camera/{subdir_name}/{original_filename}"), None)
            if original_item is None:
                continue
            
            # 打开原始图像
            img = Image.open(original_path)
            
            # 旋转并保存图像
            rotations = [90, 180, 270]
            new_filenames = [f"rgb_001{i}.png", f"rgb_002{i}.png", f"rgb_003{i}.png"]
            for angle, new_filename in zip(rotations, new_filenames):
                rotated_img = img.rotate(angle, expand=True)
                rotated_img.save(os.path.join(root, new_filename))
                
                # 创建新的 JSON 项并更新 image_path
                new_item = {
                    "dialogue": original_item["dialogue"],  # 保留原始的 dialogue
                    "image_path": f"test/camera/{subdir_name}/{new_filename}"
                }
                new_data.append(new_item)

# 保存到新的 JSON 文件（不修改原始文件）
with open(new_json_file_path, "w") as f:
    json.dump(new_data, f, indent=4)

print("图像处理和新的 JSON 文件更新完成！")
