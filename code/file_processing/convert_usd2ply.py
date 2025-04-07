import argparse
import json
import os

import yaml

import sys
from path_config import DATA_FOLDER, CONFIG_FOLDER, CODE_FOLDER
sys.path.append(str(CODE_FOLDER))

# # Default config dict, can be updated/replaced using json/yaml config files ('--config' cli argument)
# config = {
#     "launch_config": {
#         "renderer": "RayTracedLighting",
#         "headless": False,
#     },
#     "env_url": "",
#     "working_area_size": (4, 4, 3),
#     "rt_subframes": 4,
#     "num_frames": 10,
#     "num_cameras": 3,
#     "camera_collider_radius": 0.5,
#     "disable_render_products_between_captures": False,
#     "simulation_duration_between_captures": 0.05,
#     "resolution": (640, 480),
#     "camera_properties_kwargs": {
#         "focalLength": 24.0,
#         "focusDistance": 400,
#         "fStop": 0.0,
#         "clippingRange": (0.01, 10000),
#     },
#     "camera_look_at_target_offset": 0.15,
#     "camera_distance_to_target_min_max": (0.25, 0.75),
#     "writer_type": "PoseWriter",
#     "writer_kwargs": {
#         "output_dir": "_out_obj_based_sdg_pose_writer",
#         "format": None,
#         "use_subfolders": False,
#         "write_debug_images": True,
#         "skip_empty_frames": False,
#     },
#     "labeled_assets_and_properties": [
#         {
#             "url": "/Isaac/Props/YCB/Axis_Aligned/008_pudding_box.usd",
#             "label": "pudding_box",
#             "count": 5,
#             "floating": True,
#             "scale_min_max": (0.85, 1.25),
#         },
#         {
#             "url": "/Isaac/Props/YCB/Axis_Aligned_Physics/006_mustard_bottle.usd",
#             "label": "mustard_bottle",
#             "count": 7,
#             "floating": True,
#             "scale_min_max": (0.85, 1.25),
#         },
#     ],
#     "shape_distractors_types": ["capsule", "cone", "cylinder", "sphere", "cube"],
#     "shape_distractors_scale_min_max": (0.015, 0.15),
#     "shape_distractors_num": 350,
#     "mesh_distractors_urls": [
#         "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxD_04_1847.usd",
#         "/Isaac/Environments/Simple_Warehouse/Props/SM_CardBoxA_01_414.usd",
#         "/Isaac/Environments/Simple_Warehouse/Props/S_TrafficCone.usd",
#     ],
#     "mesh_distractors_scale_min_max": (0.35, 1.35),
#     "mesh_distractors_num": 75,
#     "table_url": "/Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd",
#     "table_pos": (0,0,2),
# }



def usd2ply(config):
    import carb
    # Check if there are any config files (yaml or json) are passed as arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=False, help="Include specific config parameters (json or yaml))")
    args, unknown = parser.parse_known_args()
    args_config = {}
    if args.config and os.path.isfile(args.config):
        with open(args.config, "r") as f:
            if args.config.endswith(".json"):
                args_config = json.load(f)
            elif args.config.endswith(".yaml"):
                args_config = yaml.safe_load(f)
            else:
                carb.log_warn(f"File {args.config} is not json or yaml, will use default config")
    else:
        carb.log_warn(f"File {args.config} does not exist, will use default config")

    # Update the default config dict with the external one
    config.update(args_config)

    #print(f"[SDG] Using config:\n{config}")

    import random
    import time
    from itertools import chain

    import carb.settings

    # Custom util functions for the example
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    import usdrt
    from omni.isaac.core.utils.semantics import add_update_semantics, remove_all_semantics
    from omni.isaac.nucleus import get_assets_root_path
    from omni.physx import get_physx_interface, get_physx_scene_query_interface
    from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics

    #-------------------------------------------------------------------------------

    from pxr import Usd, UsdGeom, Gf
    import numpy as np

    def extract_geometry_to_pointcloud_with_labels(usd_path, output_file, label_mapping_file):
        # Open the USD file
        stage = Usd.Stage.Open(usd_path)
        # Traverse all geometry in the scene
        all_points = []
        labels = []
        for prim in stage.Traverse():
            print(f"Prim path: {prim.GetPath()}, Type: {prim.GetTypeName()}")
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                points = mesh.GetPointsAttr().Get()
                # Get the label (name) of the object
                # label = prim.GetParent().GetName()
                # label = prim.GetPath().pathString
                label_path = prim.GetPath().pathString
                try:
                    parts = label_path.split("/")
                    labeled_index = parts.index("Labeled") 
                    label = parts[labeled_index + 1]
                except:
                    label = prim.GetParent().GetName()

                # Check if the prim is transformable and apply the world transformation
                if prim.IsA(UsdGeom.Xformable):
                    xformable = UsdGeom.Xformable(prim)
                    transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    if not points:
                        break
                    world_points = [transform.Transform(p) for p in points]
                    all_points.extend(world_points)
                    # Extend the label list with the object label for each point
                    labels.extend([label] * len(world_points))
        # Save the point cloud with labels to a PLY file
        save_points_to_ply_with_labels(all_points, labels, output_file, label_mapping_file)

    def save_points_to_ply_with_labels(points, labels, ply_filename, mapping_filename):
        # Create a set of unique labels and map them to integers
        
        unique_labels = list(set(labels))
        label_to_int = {label: i for i, label in enumerate(unique_labels)}

        # Write the PLY file
        with open(ply_filename, 'w') as f:
            f.write("ply\n")
            f.write("format ascii 1.0\n")
            f.write(f"element vertex {len(points)}\n")
            f.write("property float x\n")
            f.write("property float y\n")
            f.write("property float z\n")
            f.write("property int label\n")  # Use integer labels
            f.write("end_header\n")
            for p, label in zip(points, labels):
                f.write(f"{p[0]} {p[1]} {p[2]} {label_to_int[label]}\n")

        # Write the label mapping to a separate file

        with open(mapping_filename, 'w') as f:
            f.write("Label Mapping (integer to object name):\n")
            for label, integer in label_to_int.items():
                clean_label = label.split("/")[-1]
                f.write(f"{integer}: {clean_label}\n")

    def generate_validation_file(usd_path, validation_filename, z_range=(-1.0, 1.0)):
        # Open the USD file
        stage = Usd.Stage.Open(usd_path)
        validation_result = {
            "validation": True,
            "reason": "",
            "out_of_range_prims": []
        }

        # Traverse all geometry in the scene
        for prim in stage.Traverse():
            # print(prim.GetName())
            if prim.IsA(UsdGeom.Mesh):
                mesh = UsdGeom.Mesh(prim)
                points = mesh.GetPointsAttr().Get()
                # Check if the prim is transformable and apply the world transformation
                if prim.IsA(UsdGeom.Xformable):
                    xformable = UsdGeom.Xformable(prim)
                    transform = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
                    
                    if points:
                        # print("points",points)
                        # Check if any z-value is out of the desired range
                        p = transform.Transform(points[0])
                        if p[2] < z_range[0] or p[2] > z_range[1]:
                            print(f"clipping scene detected in {usd_path}")
                            validation_result["validation"] = False
                            validation_result["reason"] = "clipping"
                            validation_result["out_of_range_prims"].append(prim.GetName())
                            break
                    else:
                        print(f"NoneType object detected in {usd_path}")
                        validation_result["validation"] = False
                        validation_result["reason"] = "NoneType"
                        validation_result["out_of_range_prims"].append(prim.GetName())
                        break

                    # world_points = [transform.Transform(p) for p in points]
                    # for p in world_points:
                    #     if p[2] < z_range[0] or p[2] > z_range[1]:
                    #         print(f"clipping scene detected in {usd_path}")
                    #         validation_result["validation"] = False
                    #         validation_result["reason"] = "clipping"
                    #         validation_result["out_of_range_prims"].append(prim.GetName())
                    #         break
                    if not validation_result["validation"]:
                        break

        # Write the validation result to the file
        with open(validation_filename, 'w') as f:
            json.dump(validation_result, f, indent=4)

    # Call the functions
    from path_config import DATA_FOLDER, OUTPUT_FOLDER
    from tqdm import tqdm
    for i in tqdm(range(config['start_from'],config['end_with']), desc="Convert usd to ply"):
        try:
            base_path = str(DATA_FOLDER / 'test/camera/')
            usd_path = base_path + f'/{i}/scene/scene.usd'
            output_file = base_path + f'/{i}/scene/scene.ply'
            label_mapping_file = base_path + f'/{i}/scene/label_mapping.txt'
            validation_file = base_path + f'/{i}/scene/validation.json'
            
            extract_geometry_to_pointcloud_with_labels(usd_path, output_file, label_mapping_file)
            generate_validation_file(usd_path, validation_file, z_range=(-1.0, 0.0))
        except:
            print(f"file {i} does not exist")
            continue
    # /home/ja/.local/share/ov/pkg/isaac-sim-4.0.0/python.sh /home/ja/Projects/Simulation/code/file_processing/convert_usd2ply.py