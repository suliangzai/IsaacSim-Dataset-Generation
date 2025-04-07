import json
import os
import random
from path_config import DATA_FOLDER

def dataset_shuffle(config):
    # 设置 JSON 文件路径
    json_file_path = os.path.join(DATA_FOLDER, "test/camera/dataset.json")
    output_file_path = os.path.join(DATA_FOLDER, "test/camera/dataset_shuffle.json")

    # 读取生成的 JSON 文件
    with open(json_file_path, "r") as f:
        data = json.load(f)

    # 打乱数据顺序
    random.shuffle(data)

    # 将打乱顺序的数据写回到 JSON 文件
    with open(output_file_path, "w") as f:
        json.dump(data, f, indent=4)

    print(f"Finish Shuffling! Save to {output_file_path}")
