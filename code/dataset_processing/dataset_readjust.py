import os
import json
from tqdm import tqdm
from path_config import OUTPUT_FOLDER

def is_bbox_completely_outside(bbox, IMG_HEIGHT, IMG_WIDTH):
    return (bbox["x_min"] >= IMG_WIDTH and bbox["x_max"] >= IMG_WIDTH) or \
           (bbox["x_min"] < 0 and bbox["x_max"] < 0) or \
           (bbox["y_min"] >= IMG_HEIGHT and bbox["y_max"] >= IMG_HEIGHT) or \
           (bbox["y_min"] < 0 and bbox["y_max"] < 0)

def clamp_bbox(bbox, IMG_HEIGHT, IMG_WIDTH):
    bbox["x_min"] = max(0, min(bbox["x_min"], IMG_WIDTH - 1))
    bbox["y_min"] = max(0, min(bbox["y_min"], IMG_HEIGHT - 1))
    bbox["x_max"] = max(0, min(bbox["x_max"], IMG_WIDTH - 1))
    bbox["y_max"] = max(0, min(bbox["y_max"], IMG_HEIGHT - 1))
    if bbox["x_min"] == bbox["x_max"]:
        if bbox["x_min"] == 0:
            bbox["x_max"] += 1
        else:
            bbox["x_min"] -= 1
    if bbox["y_min"] == bbox["y_max"]:
        if bbox["y_min"] == 0:
            bbox["y_max"] += 1
        else:
            bbox["y_min"] -= 1
    return bbox

def process_json_file(json_path, IMG_HEIGHT, IMG_WIDTH):
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    modified = False
    shapes = data.get("shapes", [])
    to_remove_ids = set()  
    
    
    for shape in shapes:
        if "bndbox" not in shape:
            continue
            
        bbox = shape["bndbox"]
        if is_bbox_completely_outside(bbox, IMG_HEIGHT, IMG_WIDTH):
            to_remove_ids.add(shape["id"])
            modified = True
            # print(f"标记删除 {json_path} 中物体 {shape['id']} (完全超出画幅): {bbox}")
    
    if not to_remove_ids:
        for shape in shapes:
            if "bndbox" not in shape:
                continue
                
            original_bbox = shape["bndbox"].copy()
            new_bbox = clamp_bbox(shape["bndbox"], IMG_HEIGHT, IMG_WIDTH)
            
            if original_bbox != new_bbox:
                modified = True
                # print(f"修正 {json_path} 中物体 {shape['id']}: {original_bbox} -> {new_bbox}")
        
        if modified:
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
        return modified
    
    id_map = {}
    new_id = 0
    for shape in shapes:
        if shape["id"] not in to_remove_ids:
            id_map[shape["id"]] = new_id
            new_id += 1
    
    new_shapes = []
    for shape in shapes:
        if shape["id"] in to_remove_ids:
            continue  
        
        shape["id"] = id_map[shape["id"]]

        shape["parents"] = [id_map[pid] for pid in shape.get("parents", []) if pid in id_map]
        shape["children"] = [id_map[cid] for cid in shape.get("children", []) if cid in id_map]
        
        if "bndbox" in shape:
            original_bbox = shape["bndbox"].copy()
            new_bbox = clamp_bbox(shape["bndbox"], IMG_HEIGHT, IMG_WIDTH)
            
            if original_bbox != new_bbox:
                modified = True
                # print(f"修正 {json_path} 中物体 {shape['id']}: {original_bbox} -> {new_bbox}")
        
        new_shapes.append(shape)
    
    data["shapes"] = new_shapes
    
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)
    return modified

def readjust(config):
    IMG_WIDTH = config["resolution"][0]
    IMG_HEIGHT = config["resolution"][1]

    annotations_dir = OUTPUT_FOLDER / 'Annotations'
    json_files = [f for f in os.listdir(annotations_dir) if f.endswith('.json')]
    readjust_file = []
    for filename in tqdm(json_files, desc="Readjust Json file"):
        json_path = os.path.join(annotations_dir, filename)
        modified = process_json_file(json_path, IMG_HEIGHT, IMG_WIDTH)
        if modified == True:
            json_filename = os.path.basename(json_path)  
            readjust_file.append(json_filename)
    if readjust_file:
        print(f"Complete. Readjust file list{readjust_file}.")
    else:
        print("Complete. No file needs to be readjust.")

    # TODO: Add point polygon readjust
