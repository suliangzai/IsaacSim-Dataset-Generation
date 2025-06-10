import csv
import random
from path_config import CONFIG_FOLDER

def load_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        reader = csv.reader(file)
        headers = next(reader)  # 读取并保存表头
        data = [dict(zip(headers, row)) for row in reader]  # 使用表头作为键创建字典
    return data

def get_random_objects(data, n):
    if n >= len(data):
        print("警告：请求的物体数多于文本中的数量，将返回所有物体。")
    return random.sample(data, min(n, len(data)))  # 防止n大于列表长度

def get_target_object(data, idx):
    assert idx <= len(data), f"idx {idx} out of range"
    return data[idx]

def get_random_object_from_file(file_path, n):
    data = load_data(file_path)
    return get_random_objects(data, n)

def get_random_table_from_file(file_path):
    data = load_data(file_path)
    return get_random_objects(data, 1)

def get_target_object_from_file(file_path, idx):
    data = load_data(file_path)
    return get_target_object(data, idx)
    

# for debuging
def get_obj_in_seq(file_path,i):
    data = load_data(file_path)
    return data[i*10:(i+1)*10]

# 使用示例
if __name__ == "__main__":
    file_path = CONFIG_FOLDER / 'object_info.csv'
    n = 5  # 想要随机选取的物体数量
    data = load_data(file_path)
    random_objects = get_random_objects(data, n)
    print("随机选取的物体是：")
    for obj in random_objects:
        print(obj)
