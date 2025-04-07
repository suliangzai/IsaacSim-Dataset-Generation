import json
import os
import sys
import csv
from tqdm import tqdm
from path_config import DATA_FOLDER, CONFIG_FOLDER, CODE_FOLDER, OUTPUT_FOLDER
import random
import cv2
import shutil
import numpy as np

# Append custom module path
sys.path.append(str(CODE_FOLDER))

# Function to load the object info from the CSV file and create a name-to-description mapping
def load_object_info(config_file_path):
    object_info_mapping = {}
    try:
        with open(config_file_path, 'r') as csvfile:
            reader = csv.DictReader(csvfile)
            for row in reader:
                base_name = row['name']
                description = row['description']
                # Map both the base name and any potential suffixed names
                object_info_mapping[base_name] = description
    except FileNotFoundError:
        print(f"Configuration file not found: {config_file_path}")
    return object_info_mapping

# Function to load label mapping from the label_mapping.txt file
def load_label_mapping(label_mapping_path):
    label_mapping = {}
    try:
        with open(label_mapping_path, 'r') as file:           
            next(file) # Skip the first line of the file
            for line in file:
                line = line.strip()
                if line:
                    label_id, label_name = line.split(':')
                    label_mapping[label_id.strip()] = label_name.strip()
    except FileNotFoundError:
        print(f"Label mapping file not found: {label_mapping_path}")
    return label_mapping

# Function to load the position data from the position JSON file
def load_position_data(position_file_path):
    try:
        with open(position_file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Position file not found: {position_file_path}")
        return {}
    
def load_occlusion_data(path):
    # Load occlusion data from a JSON file
    try:
        with open(path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Position file not found: {path}")
        return {}
    
def load_segmentation_data(segmentation_file_path):
    try:
        with open(segmentation_file_path, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"Position file not found: {segmentation_file_path}")
        return {}
    
# Function to check if two objects are mutually occluding each other
def is_mutually_occluding(occlusion_tree, obj1, obj2):
    return obj2 in occlusion_tree.get(obj1, []) and obj1 in occlusion_tree.get(obj2, [])

# Function to determine relative position between two objects based on coordinates
def determine_position(obj1, obj2, position_data):
    if obj1 in position_data and obj2 in position_data:
        x1, y1 = position_data[obj1]
        x2, y2 = position_data[obj2]

        # Calculate the absolute differences in x and y coordinates
        dx = abs(x2 - x1)
        dy = abs(y2 - y1)

        # If the horizontal distance is greater, return left/right
        if dx > dy:
            if x1 < x2:
                return "left"
            else:
                return "right"
        # Otherwise, return above/below based on vertical distance
        else:
            if y1 < y2:
                return "above"
            else:
                return "below"
    return None

# Function to strip numerical suffix from object name (e.g., banana_01 -> banana)
def strip_suffix(object_name):
    parts = object_name.split('_')
    # Check if last part is numeric, if yes strip it
    if parts[-1].isdigit():
        return '_'.join(parts[:-1])  # Return everything before the last part
    return object_name  # Return the object name as is if no numeric suffix

# Function to map object names to their descriptions, with added position info if necessary
def get_object_description(label, label_mapping, object_info_mapping, position_data):
    base_name = label_mapping.get(label, label)  # Map from int to string name
    base_name_stripped = strip_suffix(base_name)  # Strip numerical suffix to get base name
    
    description = object_info_mapping.get(base_name, base_name_stripped)  # Get description based on base name

    # Find all objects with the same base name (ignoring suffixes)
    same_objects = [key for key, val in label_mapping.items() if strip_suffix(val) == base_name]

    if len(same_objects) > 1:
        # If there are multiple objects with the same base name, compare their positions
        for other_obj in same_objects:
            if other_obj != label:
                position_relation = determine_position(label, other_obj, position_data)
                if position_relation:
                    return f"the {position_relation} {description}"
    return description

# add description mapping 
def get_obj_description(target_name):
    obj_info_file = '/home/ja/Projects/Simulation/config/object_info_v2.csv'


# Function to generate the occlusion relationships from the occlusion_tree, now using object descriptions with positions
def generate_occlusion_relationship(tree, label_mapping, object_info_mapping, position_data, ignore_vocab, ignore_mutual_occlusion):
    if not tree or all(not occludees for occludees in tree.values()):
        # If no occlusion exists, return the special case message
        return "Objects on the desktop are not occluded"
    
    def get_label(label):
        # Map the label to the object description with position info if available
        return get_object_description(str(label), label_mapping, object_info_mapping, position_data)
    
    occlusion_descriptions = []
    for occluder, occludees in tree.items():
        for occludee in occludees:
            # Ignore if both objects are in a mutual occlusion relationship and the flag is set
            if ignore_mutual_occlusion and is_mutually_occluding(tree, occluder, occludee):
                continue
            # Ignore if either object is in the ignore vocabulary
            if label_mapping.get(occluder) in ignore_vocab or label_mapping.get(occludee) in ignore_vocab:
                continue
            occlusion_descriptions.append(f"{get_label(occluder)} occludes {get_label(occludee)}")

    return ', '.join(occlusion_descriptions) if occlusion_descriptions else "Objects on the desktop are not occluded"

# Function to update the dialogue entry, now including the image in the human dialogue
def update_dialogue(occlusion_relationship):
    sentences = [
        "Describe the occlusion situation of the objects on the desk in the picture.",
        "Explain the occlusion of the objects on the desk in the image.",
        "Detail how the objects on the desk are blocking each other in the photo.",
        "Describe how the objects on the desk are overlapping in the picture.",
        "Outline the occlusion patterns of the objects on the desk in the image.",
        "Provide an explanation of the objects' occlusion on the desk as seen in the picture.",
        "Illustrate the way objects on the desk obstruct one another in the image.",
        "Analyze the occlusion between the objects on the desk in the picture.",
        "Break down the overlapping of objects on the desk in the image.",
        "Depict how the objects on the desk conceal each other in the picture.",
        "Clarify the arrangement and occlusion of the objects on the desk in the photo."
    ]
    
    selected_sentence = random.choice(sentences)
    
    return [
        {"from": "human", "value": f"<image>{selected_sentence}"},
        {"from": "gpt", "value": occlusion_relationship}
    ]

# Function to check if a specific object (e.g., A) is occluded
def is_object_occluded(target_obj, occlusion_tree):
    for occluder, occludees in occlusion_tree.items():
        if target_obj in occludees:
            return True  # Target object is occluded
    return False  # Target object is not occluded

# Function to generate single object occlusion detection data
def generate_single_object_occlusion_data(tree, label_mapping, target_label, object_info_mapping, position_data):
    target_name = get_object_description(str(target_label), label_mapping, object_info_mapping, position_data)
    occluded = is_object_occluded(target_label, tree)
    
    if occluded:
        return [
            {"from": "human", "value": f"<image>Is {target_name} occluded by other object?"},
            {"from": "gpt", "value": "Yes."}
        ]
    else:
        return [
            {"from": "human", "value": f"<image>Is {target_name} occluded by other object?"},
            {"from": "gpt", "value": "No."}
        ]

# Function to identify which objects are occluding a target object
def identify_occluding_objects(target_label, occlusion_tree, label_mapping, object_info_mapping, position_data, permit_flag):
    target_name = get_object_description(str(target_label), label_mapping, object_info_mapping, position_data)
    occluders = []
    for occluder, occludees in occlusion_tree.items():
        if target_label in occludees:
            occluders.append(occluder)

    if not occluders:
        if permit_flag == True and target_name != 'Table':
            return False,[
                {"from": "human", "value": f"<image>What occludes {target_name}?"},
                {"from": "gpt", "value": f"{target_name} is not occluded by any objects."}
            ]
        return False, None

    occluder_descriptions = []
    for occluder in occluders:
        occluder_description = get_object_description(str(occluder), label_mapping, object_info_mapping, position_data)
        occluder_descriptions.append(occluder_description)

    return True,[
        {"from": "human", "value": f"<image>What occludes {target_name}?"},
        {"from": "gpt", "value": f": {', '.join(occluder_descriptions)}."}
    ]

# Function to list all objects present in the current scene based on label mapping
def list_all_objects_in_scene(label_mapping, object_info_mapping, position_data):
    all_objects = []
    for label, name in label_mapping.items():
        object_description = get_object_description(str(label), label_mapping, object_info_mapping, position_data)
        all_objects.append(object_description)

    return [
        {"from": "human", "value": f"<image>What are the objects on the table?"},
        {"from": "gpt", "value": f"{', '.join(all_objects)}."}
    ]

# Function to load the ignore vocabulary from a file
def load_ignore_vocabulary(vocab_path):
    ignore_vocab = set()
    try:
        with open(vocab_path, 'r') as file:
            for line in file:
                ignore_vocab.add(line.strip())
    except FileNotFoundError:
        print(f"Vocabulary file not found: {vocab_path}")
    return ignore_vocab

# Main processing loop
def process_occlusion_data(config):
    # Load object info mapping from CSV
    object_info_path = os.path.join(CONFIG_FOLDER, 'object_info.csv')
    object_info_mapping = load_object_info(object_info_path)

    # Load ignore vocabulary
    ignore_vocab_path = os.path.join(CONFIG_FOLDER, 'ignore_vocab.txt')
    ignore_vocab = load_ignore_vocabulary(ignore_vocab_path)

    updated_data = []
    permit_counter = 0
    for i in tqdm(range(config['start_from'],config['end_with']), desc="Generate Dataset, Processing Cameras"):
        # Load the label mapping for the current camera
        label_mapping_path = os.path.join(DATA_FOLDER, f'test/camera/{i}/scene/label_mapping.txt')
        label_mapping = load_label_mapping(label_mapping_path)
        num_obj = len(label_mapping)

        for j in tqdm(range(config['num_frames']), desc=f"Processing Scenes in Camera {i}", leave=False):
            occlusion_file_path = os.path.join(DATA_FOLDER, f'test/camera/{i}/scene/occlusion/occlusion_tree_{j:04d}.json')
            image_path = f'test/camera/{i}/rgb_{j:04d}.png'  # Image path added
            position_file_path = os.path.join(DATA_FOLDER, f'test/camera/{i}/scene/position/position_{j:04d}.json')

            try:
                # Load occlusion tree data
                with open(occlusion_file_path, 'r') as occlusion_file:
                    occlusion_data = json.load(occlusion_file)
                    occlusion_tree = occlusion_data.get('occlusion_tree', {})

                # Load position data
                position_data = load_position_data(position_file_path)

                # List all objects in the current scene
                all_objects_description = list_all_objects_in_scene(label_mapping, object_info_mapping, position_data)
                updated_entry = {
                    "dialogue": all_objects_description,
                    "image_path": image_path  # Adding image path to the entry
                }
                updated_data.append(updated_entry)
                
                scene_occlusion_exist = False
                for n in range(num_obj):
                    str_n = str(n)
                   
                    permit_flag = False
                    if permit_counter > 0:
                        permit_flag = True
                 
                    occlusion_exist, occluders_description = identify_occluding_objects(str_n, occlusion_tree, label_mapping, object_info_mapping, position_data, permit_flag)
                    if occluders_description != None:
                        if occlusion_exist == True:
                            permit_counter += 1
                            scene_occlusion_exist = True
                        else:
                            permit_counter -= 1
                        updated_entry = {
                            "dialogue": occluders_description,
                            "image_path": image_path  # Adding image path to the entry
                        }
                        updated_data.append(updated_entry)

                # Generate occlusion relationship
                if scene_occlusion_exist == True:
                    occlusion_relationship = generate_occlusion_relationship(occlusion_tree, label_mapping, object_info_mapping, position_data, ignore_vocab, ignore_mutual_occlusion=True)

                    # Update dialogue, including the image path in the human entry
                    updated_entry = {
                        "dialogue": update_dialogue(occlusion_relationship),
                        "image_path": image_path  # Adding image path to the entry
                    }
                    updated_data.append(updated_entry)

            except FileNotFoundError:
                print(f"File not found: {occlusion_file_path}")
            except json.JSONDecodeError:
                print(f"Error decoding JSON in file: {occlusion_file_path}")

    return updated_data

def generate_coco(config):
    print("Generate coco dataset.")
    dataset = {"info": [], "images": [], "annotations": [], "categories": []}
    dataset["info"].append({"description": "Test COCO Dataset","version": "1.0","year": 2025,"contributor": "Liangzai"})
    
    annotation_id = 0
    num_object = 0
    
    for i in tqdm(range(config['start_from'],config['end_with']), desc="Processing Cameras"):
        camera_folder = os.path.join(DATA_FOLDER, f'test/camera/{i}')
    
        if not os.path.exists(camera_folder):
            print(f"Skipping Camera {i}: Folder does not exist.")
            continue  # Skip to the next camera if the folder does not exist
        
        label_mapping_path = os.path.join(DATA_FOLDER, f'test/camera/{i}/scene/label_mapping.txt')
        label_mapping = load_label_mapping(label_mapping_path)
        
        for label_id, label_name in label_mapping.items():
            dataset["categories"].append({"id": num_object + int(label_id), "name": label_name, "supercategory": "object"})

        
        for j in tqdm(range(config['num_frames']), desc=f"Processing Scenes in Camera {i}", leave=False):
            image_path = os.path.join(DATA_FOLDER, f'test/camera/{i}/rgb_{j:04d}.png')
            position_file_path = os.path.join(DATA_FOLDER, f'test/camera/{i}/scene/position/position_{j:04d}.json')
            position_data = load_position_data(position_file_path)
            
            # Load image to get actual dimensions
            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image {image_path}. Skipping.")
                continue
            image_height, image_width, _ = image.shape  # Get real width & height

            image_entry = {"id": i * (config['num_frames']) + j, "file_name": image_path, "width": image_width, "height": image_height}
            dataset["images"].append(image_entry)
            
            # ** Step 1: Filter out invalid objects and renumber them. **
            valid_position_data = {}
            valid_object_ids = []
            for label, obj in position_data.items():
                category_id = int(label)
                x, y = obj["center"]
                width = obj["width"]
                height = obj["height"]
                x_min, x_max, y_min, y_max = obj["x_min"], obj["x_max"], obj["y_min"], obj["y_max"]
                area = width * height
                if area > (image_width * image_height):
                    continue  # Skip invalid object with too large area
                valid_position_data[label] = obj  # Store valid object
                valid_object_ids.append(category_id)  # Keep track of valid object ids
            
            # ** Step 2: Reassign new IDs to valid objects and create annotations. **
            object_id_map = {old_id: new_id for new_id, old_id in enumerate(valid_object_ids)}
            
            for label, obj in valid_position_data.items():
                category_id = int(label)
                new_id = object_id_map[category_id]  # Get the new ID
                x_min, y_min, x_max, y_max = obj["x_min"], obj["y_min"], obj["x_max"], obj["y_max"]
                width = obj["width"]
                height = obj["height"]
                bbox = [x_min, y_min, width, height]  
                annotation_entry = {
                    "id": annotation_id,
                    "image_id": image_entry["id"],
                    "category_id": num_object + new_id,  # Update category_id to the new ID
                    "bbox": bbox,
                    "area": bbox[2] * bbox[3],
                    "iscrowd": 0
                }
                dataset["annotations"].append(annotation_entry)
                annotation_id += 1
        
        # Update the num_object count after processing each camera
        num_object += len(label_mapping)
    
    return dataset

def visualize_coco_bboxes(dataset_path, num_samples=5):
    """
    读取 COCO 数据集并在图像上绘制 bbox 进行可视化验证。

    :param dataset_path: COCO 数据集 JSON 文件路径
    :param num_samples: 可视化的图像数量（默认 5 张）
    """
    # 读取 COCO JSON 数据
    with open(dataset_path, "r") as f:
        coco_data = json.load(f)
    
    # 创建 image_id 到 文件名的映射
    image_id_to_filename = {img["id"]: img["file_name"] for img in coco_data["images"]}
    
    # 取出所有 annotations
    annotations = coco_data["annotations"]
    
    # 只取前 `num_samples` 个不同的 image_id
    unique_image_ids = list(set(ann["image_id"] for ann in annotations))[:num_samples]
    
    for image_id in unique_image_ids:
        # 获取图像文件名
        image_filename = image_id_to_filename.get(image_id, None)
        if not image_filename:
            print(f" image_id {image_id} not found")
            continue
        
        # 获取图像路径
        image_path = str(DATA_FOLDER / image_filename) # 假设图像路径已经是相对路径
        # 读取图像

        image = cv2.imread(image_path)
        
        if image is None:
            print(f"can not load {image_path}")
            continue

        # OpenCV 读取的图像是 BGR，需要转换为 RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 绘制该 image_id 的所有 bbox
        for ann in annotations:
            if ann["image_id"] == image_id:
                x_min, y_min, width, height = ann["bbox"]
                x_min, y_min, width, height = int(x_min), int(y_min), int(width), int(height)

                # 画框
                cv2.rectangle(image, (x_min, y_min), (x_min + width, y_min + height), (255, 0, 0), 2)

                # 显示类别 ID
                category_id = ann["category_id"]
                cv2.putText(image, f"ID: {category_id}", (x_min, y_min - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)

        # Save the image to a file
        output_path = 'output_image.png'  # Specify your desired file path here
        cv2.imwrite(output_path, image)

def generate_reli(config):
    print("Generate Reli Dataset.")
    # Paths for Images and Annotations
    images_folder = OUTPUT_FOLDER / 'Images'
    annotations_folder = OUTPUT_FOLDER / 'Annotations'
    
    images_folder.mkdir(parents=True, exist_ok=True)
    annotations_folder.mkdir(parents=True, exist_ok=True)

    for i in tqdm(range(config['start_from'], config['end_with']), desc="Processing Cameras"):

        camera_folder = os.path.join(DATA_FOLDER, f'test/camera/{i}')
    
        if not os.path.exists(camera_folder):
            print(f"Skipping Camera {i}: Folder does not exist.")
            continue  # Skip to the next camera if the folder does not exist
        
        label_mapping_path = os.path.join(DATA_FOLDER, f'test/camera/{i}/scene/label_mapping.txt')
        label_mapping = load_label_mapping(label_mapping_path)

        # Process frames for each camera
        for j in tqdm(range(config['num_frames']), desc=f"Processing Scenes in Camera {i}", leave=False):

            image_path = os.path.join(DATA_FOLDER, f'test/camera/{i}/rgb_{j:04d}.png')
            position_file_path = os.path.join(DATA_FOLDER, f'test/camera/{i}/scene/position/position_{j:04d}.json')
            segmentation_file_path = os.path.join(DATA_FOLDER, f'test/camera/{i}/scene/segmentation/segmentation_{j:04d}.json')
            
            position_data = load_position_data(position_file_path)
            segmentation_data = load_segmentation_data(segmentation_file_path)
            # Generate new image name (zero-padded index)
            image_name = f"{str(i * config['num_frames'] + j).zfill(4)}.jpg"
            new_image_path = images_folder / image_name

            # Process occlusion tree for this camera
            occlusion_file_path = os.path.join(DATA_FOLDER, f'test/camera/{i}/scene/occlusion/occlusion_tree_{j:04d}.json')
            occlusion_data = load_occlusion_data(occlusion_file_path)
            occlusion_tree = occlusion_data["occlusion_tree"]

            image = cv2.imread(image_path)
            if image is None:
                print(f"Warning: Could not load image {image_path}. Skipping.")
                continue
            image_height, image_width, _ = image.shape  # Get height and width

            # Copy image to the new folder with new name
            cv2.imwrite(str(new_image_path), image)
            # Image entry for the annotation JSON
            image_entry = {
                "imagePath": str(new_image_path),
                "imageWidth": image_width,  
                "imageHeight": image_height 
            }

            # Create an individual annotation file for each image
            annotation_entry = {
                "type": "Relation",
                "shapes": [],
                "imagePath": image_entry["imagePath"],
                "imageWidth": image_entry["imageWidth"],
                "imageHeight": image_entry["imageHeight"]
            }

            # Prepare annotation data for each object in the position data
            valid_objects = {}  # Store valid objects to maintain correct relations
            relations = {key: {"parents": [], "children": []} for key in position_data.keys()}

            valid_position_data = {}
            valid_object_ids = []

            for label, obj in position_data.items():
                category_id = int(label)
                label_name = label_mapping[str(category_id)]  # Get the object name from category_id
                x_min, y_min, x_max, y_max = obj["x_min"], obj["y_min"], obj["x_max"], obj["y_max"]
                bbox_area = (x_max - x_min) * (y_max - y_min)
                
                if bbox_area > (image_width * image_height):
                    continue  # Ignore this object

                valid_position_data[label] = obj  # Keep valid objects
                valid_object_ids.append(category_id)  # Store valid object ids

            object_id_map = {old_id: new_id for new_id, old_id in enumerate(valid_object_ids)}
            for label, obj in valid_position_data.items():
                category_id = int(label)
                new_id = object_id_map[category_id]
                x_min, y_min, x_max, y_max = obj["x_min"], obj["y_min"], obj["x_max"], obj["y_max"]
                points = [
                    [x_min, y_min], [x_max, y_min], [x_max, y_max], [x_min, y_max]
                ]

                # Retrieve the polygon points from the segmentation data
                for obj in segmentation_data["objects"]:
                    if obj["label"] == str(category_id):
                        points = obj["polygon"]

                valid_objects[new_id] = {
                    "label": label_mapping[str(category_id)],  # Object name (label)
                    "id": new_id,  # new label_id (category_id)
                    "points": points,#points,
                    "bndbox": {
                        "x_min": x_min,
                        "y_min": y_min,
                        "x_max": x_max,
                        "y_max": y_max
                    },
                    "parents": [],
                    "children": [],
                    "features": {
                        "blue": False,
                        "red": False
                    },
                    "feature_vector": [0, 0],
                    "shape_type": "polygon"
                }
                annotation_entry["shapes"].append(valid_objects[new_id])

            # Step 3: Update occlusion relationships based on new IDs
            for obj_id, occluded_by in occlusion_tree.items():
                obj_id = int(obj_id)
                if obj_id not in object_id_map:
                    continue  # Skip if obj_id is not in object_id_map
                new_obj_id = int(object_id_map[obj_id])
                parent_ids = [object_id_map.get(int(p), None) for p in occluded_by if int(p) in object_id_map]
                parent_ids = [pid for pid in parent_ids if pid is not None]  # Remove None values
                
                # Update the relation for the valid object
                valid_objects[new_obj_id]["parents"] = parent_ids
                
            for obj_id, obj_data in valid_objects.items():
                for other_id, other_data in valid_objects.items():
                    if obj_id in other_data["parents"]:
                        obj_data["children"].append(other_id)

            # Save annotation to its own JSON file (named the same as the image)
            annotation_file_path = annotations_folder / f"{image_name.split('.')[0]}.json"
            with open(annotation_file_path, 'w') as f:
                json.dump(annotation_entry, f, indent=4)
    
    print(f"Reli data successfully saved to {str(OUTPUT_FOLDER)}")

def visualize_reli_bboxes(annotation_file_path):
    # Load the annotation file
    with open(annotation_file_path, 'r') as f:
        annotation_data = json.load(f)
    
    # Load the image corresponding to the annotation
    image_path = annotation_data['imagePath']
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Warning: Could not load image {image_path}. Skipping.")
        return

    # Draw bounding boxes, polygons, and relationships
    for shape in annotation_data['shapes']:
        # Get the polygon points
        points = shape['points']
        
        # Draw the bounding box
        x_min = shape['bndbox']['x_min']
        y_min = shape['bndbox']['y_min']
        x_max = shape['bndbox']['x_max']
        y_max = shape['bndbox']['y_max']
        cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)  # Green bounding box

        # Draw the polygon (if available)
        if points:
            polygon_points = [tuple(point) for point in points]
            cv2.polylines(image, [np.array(polygon_points)], isClosed=True, color=(0, 0, 255), thickness=2)  # Red polygon

        # Draw parent-child relationships
        parent_ids = shape['parents']
        child_ids = shape['children']
        
        # Draw relationships (arrows between parent and child)
        for parent_id in parent_ids:
            parent_shape = next((item for item in annotation_data['shapes'] if item['id'] == parent_id), None)
            if parent_shape:
                parent_bndbox = parent_shape['bndbox']
                cv2.arrowedLine(image, 
                                 (parent_bndbox['x_max'], parent_bndbox['y_max']),
                                 (x_min, y_min), 
                                 (255, 0, 0), 2)  # Blue arrow from parent to child

        for child_id in child_ids:
            child_shape = next((item for item in annotation_data['shapes'] if item['id'] == child_id), None)
            if child_shape:
                child_bndbox = child_shape['bndbox']
                cv2.arrowedLine(image, 
                                 (x_max, y_max),
                                 (child_bndbox['x_min'], child_bndbox['y_min']), 
                                 (255, 0, 0), 2)  # Blue arrow from child to parent

    # Save the image with the drawn annotations (bbox, polygon, relationships)
    output_image_path = os.path.join(os.path.dirname(annotation_file_path), 'output', os.path.basename(annotation_file_path).replace('.json', '.jpg'))
    os.makedirs(os.path.dirname(output_image_path), exist_ok=True)
    cv2.imwrite(output_image_path, image)

    print(f"Visualization complete. Image saved at: {output_image_path}")

# Save the updated dataset to a file
def save_dataset(data, output_path):
    with open(output_path, 'w') as file:
        json.dump(data, file, indent=4)
    print(f"Coco data successfully saved to {output_path}")

def generate_dataset(config):
    # Process occlusion data
    updated_data = generate_coco(config)

    # Define the output file path
    dataset_path = os.path.join(DATA_FOLDER, 'test/camera/dataset.json')

    # Save the updated dataset
    save_dataset(updated_data, dataset_path)

    #visualize_coco_bboxes(dataset_path, num_samples=1)

    generate_reli(config)

    for i in range(100):
        reli_path = str(OUTPUT_FOLDER / f'Annotations/{(3*i+1):04d}.json')
        visualize_reli_bboxes(reli_path)
