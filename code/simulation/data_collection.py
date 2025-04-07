# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

# Refer to https://docs.omniverse.nvidia.com/isaacsim/latest/replicator_tutorials/tutorial_replicator_object_based_sdg.html for origional example
# At the root of the project, run command: 
# /home/ja/.local/share/ov/pkg/isaac-sim-4.0.0/python.sh /home/ja/Projects/Simulation/code/simulation/data_collection.py

import argparse
import json
import os
import sys
import math
import numpy as np

import yaml
from isaacsim import SimulationApp

from path_config import DATA_FOLDER, CONFIG_FOLDER, CODE_FOLDER, MESH_FOLDER

sys.path.append(str(CODE_FOLDER))

from my_utils import random_choose_object




def data_collect(config, simulation_app):
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

    print(f"[SDG] Using config:\n{config}")

    import random
    import time
    from itertools import chain

    import carb.settings

    # Custom util functions for the example
    from code.simulation import object_based_sdg_utils
    import omni.replicator.core as rep
    import omni.timeline
    import omni.usd
    import usdrt
    from omni.isaac.core.utils.semantics import add_update_semantics, remove_all_semantics
    from omni.isaac.nucleus import get_assets_root_path
    from omni.physx import get_physx_interface, get_physx_scene_query_interface
    from pxr import PhysxSchema, Sdf, UsdGeom, UsdPhysics

    start_from = int(config.get("start_from", 0))
    end_with = int(config.get("end_with", 100))
    scene_num = start_from
    #*****************************************************
    # Each iteration represents a scene
    # 'start_from' specifies the starting scene number
    # 'end_with' specifies the ending scene number
    #*****************************************************

    while scene_num in range(start_from, end_with):
        try:
            print("scene_num",scene_num)
            # Isaac nucleus assets root path
            assets_root_path = get_assets_root_path()
            stage = None

            # ENVIRONMENT
            # Create an empty or load a custom stage (clearing any previous semantics)
            env_url = config.get("env_url", "")
            if env_url:
                env_path = env_url if env_url.startswith("omniverse://") else assets_root_path + env_url
                omni.usd.get_context().open_stage(env_path)
                stage = omni.usd.get_context().get_stage()
                # Remove any previous semantics in the loaded stage
                for prim in stage.Traverse():
                    remove_all_semantics(prim)
            else:
                omni.usd.get_context().new_stage()
                stage = omni.usd.get_context().get_stage()
                # Add a distant light to the empty stage
                distant_light = stage.DefinePrim("/World/Lights/DistantLight", "DistantLight")
                distant_light.CreateAttribute("inputs:intensity", Sdf.ValueTypeNames.Float).Set(400.0)
                if not distant_light.HasAttribute("xformOp:rotateXYZ"):
                    UsdGeom.Xformable(distant_light).AddRotateXYZOp()
                distant_light.GetAttribute("xformOp:rotateXYZ").Set((0, 60, 0))

            # Get the working area size and bounds (width=x, depth=y, height=z) 
            working_area_size = config.get("working_area_size", (3, 3, 3))
            generation_area_size = config.get("generation_area_size", (3, 3, 3))
            geration_area_min = (generation_area_size[0] / -2, generation_area_size[1] / -2, 0)
            geration_area_max = (generation_area_size[0] / -2, generation_area_size[1] / -2, working_area_size[2] / 2)

            # Create a collision box area around the assets to prevent them from drifting away
            object_based_sdg_utils.create_collision_box_walls(
                stage, "/World/CollisionWalls", working_area_size[0], working_area_size[1], working_area_size[2], visible=False
            )

            # Create a physics scene to add or modify custom physics settings
            usdrt_stage = usdrt.Usd.Stage.Attach(omni.usd.get_context().get_stage_id())
            physics_scenes = usdrt_stage.GetPrimsWithAppliedAPIName("PhysxSceneAPI")
            if physics_scenes:
                physics_scene = physics_scenes[0]
            else:
                physics_scene = UsdPhysics.Scene.Define(stage, "/PhysicsScene")
                physx_scene = PhysxSchema.PhysxSceneAPI.Apply(stage.GetPrimAtPath("/PhysicsScene"))
            physx_scene.GetTimeStepsPerSecondAttr().Set(60)


            # Add a random table
            print("adding table")
            table_colliders = []
            random_table_info = config.get("random_table_info", "")
            [random_table] = random_choose_object.get_random_table_from_file(random_table_info) # Randomly choose a table from the table list
            print("random_table",random_table)

            table_url = str(MESH_FOLDER / random_table['path'])

            # table_url = config.get("table_url", None) # Choose target table defined in configuration
            if table_url == None:
                print("No table detacted!")
            else:
                table_pos = tuple(config.get("table_pos", (0,0,0)))
                prim_path = omni.usd.get_stage_next_free_path(stage, f"/World/Table", False)
                prim = stage.DefinePrim(prim_path, "Xform")
                if config.get("table_seen") == True:
                    UsdGeom.Imageable(prim).GetVisibilityAttr().Set("invisible")
                asset_path = table_url if (table_url.startswith("omniverse://") or table_url.startswith("/home")) else assets_root_path + table_url
                prim.GetReferences().AddReference(asset_path)
                object_based_sdg_utils.set_transform_attributes(prim, location=table_pos, scale=(1.5, 1.5, 1.5))
                object_based_sdg_utils.add_colliders(prim)
                object_based_sdg_utils.add_rigid_body_dynamics(prim, disable_gravity=False)
                table_colliders.append(prim)
            
            # Add the objects to be trained in the environment with their labels and properties
            print("adding assets ...")
            floating_labeled_prims = []
            falling_labeled_prims = []
            labeled_prims = []
            random_object_info = config.get("random_object_info", "")
            min_obj_num = config.get("min_obj_num", 8)
            max_obj_num = config.get("max_obj_num", 15)
            random_objects = random_choose_object.get_random_object_from_file(random_object_info, random.randint(min_obj_num,max_obj_num))

            # Check if we should add a box
            if config.get("add_box") == "box":
                box_url = config.get("box_url", None)
                box_url = str(MESH_FOLDER / box_url)
                print("adding box")
                if box_url:
                    box_pos = tuple(config.get("box_pos", (0,0,0)))
                    box_rotation = tuple(config.get("box_rotation", (90, 0, 0)))
                    prim_path = omni.usd.get_stage_next_free_path(stage, f"/World/Box", False)
                    prim = stage.DefinePrim(prim_path, "Xform")
                    asset_path = box_url if (box_url.startswith("omniverse://") or box_url.startswith("/home")) else assets_root_path + box_url
                    prim.GetReferences().AddReference(asset_path)
                    object_based_sdg_utils.set_transform_attributes(prim, location=box_pos,rotation=box_rotation, scale=(0.015,0.015,0.015))
                    object_based_sdg_utils.add_colliders(prim)
                    object_based_sdg_utils.add_rigid_body_dynamics(prim, disable_gravity=False)
                else:
                    print("Box URL not specified in config!")

            if config.get("add_box") == "plate":
                plate_url = config.get("plate_url", None)
                box_url = str(MESH_FOLDER / plate_url)
                print("adding plate")
                if box_url:
                    box_pos = tuple(config.get("box_pos", (0,0,0)))
                    box_rotation = tuple(config.get("box_rotation", (90, 0, 0)))
                    prim_path = omni.usd.get_stage_next_free_path(stage, f"/World/Box", False)
                    prim = stage.DefinePrim(prim_path, "Xform")
                    asset_path = box_url if (box_url.startswith("omniverse://") or box_url.startswith("/home")) else assets_root_path + box_url
                    prim.GetReferences().AddReference(asset_path)
                    object_based_sdg_utils.set_transform_attributes(prim, location=box_pos,rotation=box_rotation, scale=(0.001,0.001,0.001))
                    object_based_sdg_utils.add_colliders(prim)
                    object_based_sdg_utils.add_rigid_body_dynamics(prim, disable_gravity=False)
                else:
                    print("Plate URL not specified in config!")

            # Create folder if it doesn't exist and open the file
            scene_folder_path = os.path.join(config.get("save_path"), str(scene_num), "scene")
            os.makedirs(scene_folder_path, exist_ok=True)  # Create the folder if it doesn't exist
            file_path = os.path.join(scene_folder_path, "object_data.txt")
            file = open(file_path, "w")  # Open the file for writing

            for obj in random_objects:
                obj_url = str(MESH_FOLDER / obj['path'])
                label = obj['name']
                min_count_each = config.get("min_count_each", 1)
                max_count_each = config.get("max_count_each", 1)
                count = random.randint(min_count_each, max_count_each)
                floating = False
                scale = float(obj['scale'])
                delta = 0.0
                scale_min_max = (scale - delta, scale + delta)
                
                # Write object name, path, and count into the file
                file.write(f"Name: {label}, Path: {obj_url}, Count: {count}\n")
                
                for i in range(count):
                    # Create a prim and add the asset reference
                    rand_loc, rand_rot, rand_scale = object_based_sdg_utils.get_random_transform_values(
                        loc_min=geration_area_min, loc_max=geration_area_max, scale_min_max=scale_min_max
                    )
                    print("rand_scale",rand_scale)
                    prim_path = omni.usd.get_stage_next_free_path(stage, f"/World/Labeled/{label}", False)
                    prim = stage.DefinePrim(prim_path, "Xform")
                    asset_path = obj_url if (obj_url.startswith("omniverse://") or obj_url.startswith("/home")) else assets_root_path + obj_url
                    prim.GetReferences().AddReference(asset_path)
                    object_based_sdg_utils.set_transform_attributes(prim, location=rand_loc, rotation=rand_rot, scale=rand_scale)
                    object_based_sdg_utils.add_colliders(prim)
                    object_based_sdg_utils.add_rigid_body_dynamics(prim, disable_gravity=floating)

                    # Label the asset
                    add_update_semantics(prim, label)
                    print(prim.GetName())
                    if floating:
                        floating_labeled_prims.append(prim)
                    else:
                        falling_labeled_prims.append(prim)

            file.close()
        except Exception as e:
            print(f"scene {scene_num} generate with error {e}. Retrying...")
            continue


        labeled_prims = floating_labeled_prims + falling_labeled_prims

        # REPLICATOR
        # Disable capturing every frame (capture will be triggered manually using the step function)
        rep.orchestrator.set_capture_on_play(False)

        # Create the camera prims and their properties
        print("creating cameras ...")
        cameras = []
        num_cameras = config.get("num_cameras", 1)
        camera_properties_kwargs = config.get("camera_properties_kwargs", {})
        for i in range(num_cameras):
            # Create camera and add its properties (focal length, focus distance, f-stop, clipping range, etc.)
            cam_prim = stage.DefinePrim(f"/World/Cameras/cam_{i}", "Camera")
            for key, value in camera_properties_kwargs.items():
                if cam_prim.HasAttribute(key):
                    cam_prim.GetAttribute(key).Set(value)
                else:
                    print(f"Unknown camera attribute with {key}:{value}")
            cameras.append(cam_prim)
        # Add collision spheres (disabled by default) to cameras to avoid objects overlaping with the camera view
        camera_colliders = []
        camera_collider_radius = config.get("camera_collider_radius", 0)
        if camera_collider_radius > 0:
            for cam in cameras:
                cam_path = cam.GetPath()
                cam_collider = stage.DefinePrim(f"{cam_path}/CollisionSphere", "Sphere")
                cam_collider.GetAttribute("radius").Set(camera_collider_radius)
                object_based_sdg_utils.add_colliders(cam_collider)
                collision_api = UsdPhysics.CollisionAPI(cam_collider)
                collision_api.GetCollisionEnabledAttr().Set(False)
                UsdGeom.Imageable(cam_collider).MakeInvisible()
                camera_colliders.append(cam_collider)

        # Wait an app update to ensure the prim changes are applied
        simulation_app.update()

        # Create render products using the cameras
        render_products = []
        resolution = config.get("resolution", (640, 480))
        for cam in cameras:
            rp = rep.create.render_product(cam.GetPath(), resolution)
            render_products.append(rp)

        # Enable rendering only at capture time
        disable_render_products_between_captures = config.get("disable_render_products_between_captures", True)
        if disable_render_products_between_captures:
            object_based_sdg_utils.set_render_products_updates(render_products, False, include_viewport=False)

        # Create the writer and attach the render products
        writer_type = config.get("writer_type", "PoseWriter")
        writer_kwargs = config.get("writer_kwargs", {})
        # If not an absolute path, set it relative to the current working directory
        output_scene_dir = os.path.join(config.get("save_path"), str(scene_num))
        if not os.path.exists(output_scene_dir):
            os.makedirs(output_scene_dir)
        writer_kwargs["output_dir"] = output_scene_dir
        if out_dir := output_scene_dir:
            if not os.path.isabs(out_dir):
                out_dir = os.path.join(os.getcwd(), out_dir)
                writer_kwargs["output_dir"] = out_dir
            print(f"[SDG] Writing data to: {out_dir}")
        if writer_type is not None and len(render_products) > 0:
            writer = rep.writers.get(writer_type)
            writer.initialize(**writer_kwargs)
            writer.attach(render_products)

        # RANDOMIZERS
        # Apply a random (mostly) uppwards velocity to the objects overlapping the 'bounce' area
        def on_overlap_hit(hit):
            prim = stage.GetPrimAtPath(hit.rigid_body)
            # Skip the camera collision spheres
            if prim not in camera_colliders and prim not in table_colliders :
                rand_vel = (random.uniform(-2, 2), random.uniform(-2, 2), random.uniform(4, 8))
                prim.GetAttribute("physics:velocity").Set(rand_vel)
            return True  # return True to continue the query


        # Area to check for overlapping objects (above the bottom collision box)
        overlap_area_thickness = 0.1
        overlap_area_origin = (0, 0, (-working_area_size[2] / 2) + (overlap_area_thickness / 2))
        overlap_area_extent = (
            working_area_size[0] / 2 * 0.99,
            working_area_size[1] / 2 * 0.99,
            overlap_area_thickness / 2 * 0.99,
        )


        # Triggered every physics update step to check for overlapping objects
        def on_physics_step(dt: float):
            # pass
            hit_info = get_physx_scene_query_interface().overlap_box(
                carb.Float3(overlap_area_extent),
                carb.Float3(overlap_area_origin),
                carb.Float4(0, 0, 0, 1),
                on_overlap_hit,
                False,  # pass 'False' to indicate an 'overlap multiple' query.
            )


        # Subscribe to the physics step events to check for objects overlapping the 'bounce' area
        # physx_sub = get_physx_interface().subscribe_physics_step_events(on_physics_step)


        # Pull assets towards the working area center by applying a random velocity towards the given target
        def apply_velocities_towards_target(assets, target=(0, 0, 0)):
            for prim in assets:
                loc = prim.GetAttribute("xformOp:translate").Get()
                strength = random.uniform(0.1, 1.0)
                pull_vel = ((target[0] - loc[0]) * strength, (target[1] - loc[1]) * strength, (target[2] - loc[2]) * strength)
                prim.GetAttribute("physics:velocity").Set(pull_vel)


        # Randomize camera poses to look at a random target asset (random distance and center offset)
        camera_distance_to_target_min_max = config.get("camera_distance_to_target_min_max", (0.1, 0.5))
        camera_look_at_target_offset = config.get("camera_look_at_target_offset", 0.2)

        def quaternion_to_rotation_matrix(quat):
            """Convert a quaternion (x, y, z, w) into a 3x3 rotation matrix."""
            # print(quat)
            w = quat.GetReal()
            xyz = quat.GetImaginary()
            x = xyz[0]
            y = xyz[1]
            z = xyz[2]
            return np.array([
                [1 - 2*y**2 - 2*z**2, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
                [2*x*y + 2*z*w, 1 - 2*x**2 - 2*z**2, 2*y*z - 2*x*w],
                [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x**2 - 2*y**2]
            ])

        def build_transformation_matrix(position, rotation_matrix):
            """Build a 4x4 transformation matrix from a 3x3 rotation matrix and a 3D position vector."""
            transformation_matrix = np.eye(4)
            transformation_matrix[:3, :3] = rotation_matrix
            transformation_matrix[:3, 3] = position

            return transformation_matrix
        
        def rotateXYZ_to_rotation_matrix(rotateXYZ):
            # Convert degrees to radians
            theta_x, theta_y, theta_z = np.deg2rad(rotateXYZ)
            
            # Rotation matrix around X-axis
            R_x = np.array([[1, 0, 0],
                            [0, np.cos(theta_x), -np.sin(theta_x)],
                            [0, np.sin(theta_x), np.cos(theta_x)]])
            
            # Rotation matrix around Y-axis
            R_y = np.array([[np.cos(theta_y), 0, np.sin(theta_y)],
                            [0, 1, 0],
                            [-np.sin(theta_y), 0, np.cos(theta_y)]])
            
            # Rotation matrix around Z-axis
            R_z = np.array([[np.cos(theta_z), -np.sin(theta_z), 0],
                            [np.sin(theta_z), np.cos(theta_z), 0],
                            [0, 0, 1]])
            
            # Combine the rotation matrices (Z * Y * X order)
            R = R_z @ R_y @ R_x
            
            return R

        
        def save_camera_matrices(cam, resolution, scene_num, i, DATA_FOLDER):
            # Fetch camera parameters
            focal_length = cam.GetAttribute("focalLength").Get()
            horiz_aperture = cam.GetAttribute("horizontalAperture").Get()
            width = resolution[0]
            height = resolution[1]

            # Compute vertical aperture and field of view
            vert_aperture = height / width * horiz_aperture
            fov = 2 * math.atan(horiz_aperture / (2 * focal_length))

            # Compute focal lengths in pixels and image center
            focal_x = height * focal_length / vert_aperture
            focal_y = width * focal_length / horiz_aperture
            center_x = width * 0.5
            center_y = height * 0.5

            # Construct intrinsic matrix
            intrinsic_matrix = np.array([[focal_x, 0, center_x],
                                        [0, focal_y, center_y],
                                        [0,  0,  1]])
            
            # Fetch camera extrinsic properties
            position = cam.GetAttribute("xformOp:translate").Get()
            rotation = cam.GetAttribute("xformOp:orient").Get()

            rotation_matrix = quaternion_to_rotation_matrix(rotation)

            ammend_matrix = rotateXYZ_to_rotation_matrix((180,0,0))
            rotation_matrix = rotation_matrix @ ammend_matrix

            extrinsic_matrix = build_transformation_matrix(position, rotation_matrix)
            extrinsic_matrix = np.linalg.inv(extrinsic_matrix)

            # Format the output path with zero-padded integer for 'i'
            file_name = f"camera_{i:04d}.npz"
            output_path = DATA_FOLDER / "test/camera" / str(scene_num) / file_name

            # Ensure the directory exists
            os.makedirs(output_path.parent, exist_ok=True)

            # Save the matrices in .npz format
            np.savez(output_path, intrinsic=intrinsic_matrix, extrinsic=extrinsic_matrix)

            print(f"Intrinsic and extrinsic matrices saved at: {output_path}")

        def randomize_camera_poses():
            for cam in cameras:
                # Get a random target asset to look at
                target_asset = random.choice(labeled_prims)
                # Add a look_at offset so the target is not always in the center of the camera view
                loc_offset = (
                    random.uniform(-camera_look_at_target_offset, camera_look_at_target_offset),
                    random.uniform(-camera_look_at_target_offset, camera_look_at_target_offset),
                    random.uniform(-camera_look_at_target_offset, camera_look_at_target_offset),
                )
                target_loc = target_asset.GetAttribute("xformOp:translate").Get() + loc_offset
                # Get a random distance to the target asset
                distance = random.uniform(camera_distance_to_target_min_max[0], camera_distance_to_target_min_max[1])
                # Get a random pose of the camera looking at the target asset from the given distance
                # cam_loc, quat = object_based_sdg_utils.get_random_pose_on_sphere(origin=target_loc, radius=distance)
                cam_loc, quat = object_based_sdg_utils.get_random_pose_on_half_sphere(origin=target_loc, radius=distance)
                object_based_sdg_utils.set_transform_attributes(cam, location=cam_loc, orientation=quat)
                # print("------------camera-----------")
                # position = cam.GetAttribute("xformOp:translate").Get()
                # rotation = cam.GetAttribute("xformOp:orient").Get()
                # print("position",position)
                # print("rotation",rotation) # (x,y,z,w)
                # rotation_matrix = quaternion_to_rotation_matrix(rotation)
                # ammend_matrix = rotateXYZ_to_rotation_matrix((180,0,0))
                # rotation_matrix =ammend_matrix @ rotation_matrix
                # transformation_matrix = build_transformation_matrix(cam_loc, rotation_matrix)
                # print("transformation_matrix",transformation_matrix)
                # print("------------camera-----------")


        # Temporarily enable camera colliders and simulate for the given number of frames to push out any overlapping objects
        def simulate_camera_collision(num_frames=1):
            for cam_collider in camera_colliders:
                collision_api = UsdPhysics.CollisionAPI(cam_collider)
                collision_api.GetCollisionEnabledAttr().Set(True)
            if not timeline.is_playing():
                timeline.play()
            for _ in range(num_frames):
                simulation_app.update()
            for cam_collider in camera_colliders:
                collision_api = UsdPhysics.CollisionAPI(cam_collider)
                collision_api.GetCollisionEnabledAttr().Set(False)


        # Create a randomizer for lights in the working area, manually triggered at custom events
        with rep.trigger.on_custom_event(event_name="randomize_lights"):
            lights = rep.create.light(
                light_type="Sphere",
                color = (random.uniform(0, 1), random.uniform(0, 1), random.uniform(0, 1)),
                temperature = np.random.normal(6500, 500),
                intensity = np.random.normal(35000, 5000),
                position = (random.uniform(geration_area_min[0], geration_area_max[0]),
                            random.uniform(geration_area_min[1], geration_area_max[1]),
                            random.uniform(geration_area_min[2], geration_area_max[2])),
                scale = random.uniform(0.1, 1),
                count = random.randint(0, 3),
            )


        # Create a randomizer for the dome background, manually triggered at custom events
        with rep.trigger.on_custom_event(event_name="randomize_dome_background"):
            dome_textures = [
                assets_root_path + "/NVIDIA/Assets/Skies/Indoor/autoshop_01_4k.hdr",
                assets_root_path + "/NVIDIA/Assets/Skies/Indoor/carpentry_shop_01_4k.hdr",
                assets_root_path + "/NVIDIA/Assets/Skies/Indoor/hotel_room_4k.hdr",
                assets_root_path + "/NVIDIA/Assets/Skies/Indoor/wooden_lounge_4k.hdr",
                assets_root_path + "/NVIDIA/Assets/Skies/Cloudy/abandoned_parking_4k.hdr",
                assets_root_path + "/NVIDIA/Assets/Skies/Cloudy/champagne_castle_1_4k.hdr",
                assets_root_path + "/NVIDIA/Assets/Skies/Cloudy/evening_road_01_4k.hdr",
                assets_root_path + "/NVIDIA/Assets/Skies/Cloudy/kloofendal_48d_partly_cloudy_4k.hdr",
                assets_root_path + "/NVIDIA/Assets/Skies/Cloudy/lakeside_4k.hdr",
                assets_root_path + "/NVIDIA/Assets/Skies/Cloudy/sunflowers_4k.hdr",
                assets_root_path + "/NVIDIA/Assets/Skies/Cloudy/table_mountain_1_4k.hdr",
                assets_root_path + "/NVIDIA/Assets/Skies/Clear/evening_road_01_4k.hdr",
                assets_root_path + "/NVIDIA/Assets/Skies/Clear/kloppenheim_02_4k.hdr",
                assets_root_path + "/NVIDIA/Assets/Skies/Clear/mealie_road_4k.hdr",
                assets_root_path + "/NVIDIA/Assets/Skies/Clear/noon_grass_4k.hdr",
                assets_root_path + "/NVIDIA/Assets/Skies/Clear/qwantani_4k.hdr",
                assets_root_path + "/NVIDIA/Assets/Skies/Clear/signal_hill_sunrise_4k.hdr",
                assets_root_path + "/NVIDIA/Assets/Skies/Clear/syferfontein_18d_clear_4k.hdr",
                assets_root_path + "/NVIDIA/Assets/Skies/Clear/venice_sunset_4k.hdr",
                assets_root_path + "/NVIDIA/Assets/Skies/Clear/white_cliff_top_4k.hdr",
                assets_root_path + "/NVIDIA/Assets/Skies/Studio/photo_studio_01_4k.hdr",
                assets_root_path + "/NVIDIA/Assets/Skies/Studio/studio_small_05_4k.hdr",
                assets_root_path + "/NVIDIA/Assets/Skies/Studio/studio_small_07_4k.hdr",
            ]
            dome_light = rep.create.light(light_type="Dome")
            with dome_light:
                rep.modify.attribute("inputs:texture:file", random.choice(dome_textures))
                rep.randomizer.rotation()


        # Capture motion blur by combining the number of pathtraced subframes samples simulated for the given duration
        def capture_with_motion_blur_and_pathtracing(duration=0.05, num_samples=8, spp=64):
            # For small step sizes the physics FPS needs to be temporarily increased to provide movements every syb sample
            orig_physics_fps = physx_scene.GetTimeStepsPerSecondAttr().Get()
            target_physics_fps = 1 / duration * num_samples
            if target_physics_fps > orig_physics_fps:
                print(f"[SDG] Changing physics FPS from {orig_physics_fps} to {target_physics_fps}")
                physx_scene.GetTimeStepsPerSecondAttr().Set(target_physics_fps)

            # Enable motion blur (if not enabled)
            is_motion_blur_enabled = carb.settings.get_settings().get("/omni/replicator/captureMotionBlur")
            if not is_motion_blur_enabled:
                carb.settings.get_settings().set("/omni/replicator/captureMotionBlur", True)
            # Number of sub samples to render for motion blur in PathTracing mode
            carb.settings.get_settings().set("/omni/replicator/pathTracedMotionBlurSubSamples", num_samples)

            # Set the render mode to PathTracing
            prev_render_mode = carb.settings.get_settings().get("/rtx/rendermode")
            carb.settings.get_settings().set("/rtx/rendermode", "PathTracing")
            carb.settings.get_settings().set("/rtx/pathtracing/spp", spp)
            carb.settings.get_settings().set("/rtx/pathtracing/totalSpp", spp)
            carb.settings.get_settings().set("/rtx/pathtracing/optixDenoiser/enabled", 0)

            # Make sure the timeline is playing
            if not timeline.is_playing():
                timeline.play()

            # Capture the frame by advancing the simulation for the given duration and combining the sub samples
            rep.orchestrator.step(delta_time=duration, pause_timeline=False)

            # Restore the original physics FPS
            if target_physics_fps > orig_physics_fps:
                print(f"[SDG] Restoring physics FPS from {target_physics_fps} to {orig_physics_fps}")
                physx_scene.GetTimeStepsPerSecondAttr().Set(orig_physics_fps)

            # Restore the previous render and motion blur  settings
            carb.settings.get_settings().set("/omni/replicator/captureMotionBlur", is_motion_blur_enabled)
            print(f"[SDG] Restoring render mode from 'PathTracing' to '{prev_render_mode}'")
            carb.settings.get_settings().set("/rtx/rendermode", prev_render_mode)


        # Update the app until a given simulation duration has passed (simulate the world between captures)
        def run_simulation_loop(duration):
            timeline = omni.timeline.get_timeline_interface()
            elapsed_time = 0.0
            previous_time = timeline.get_current_time()
            if not timeline.is_playing():
                timeline.play()
            app_updates_counter = 0
            while elapsed_time <= duration:
                simulation_app.update()
                elapsed_time += timeline.get_current_time() - previous_time
                previous_time = timeline.get_current_time()
                app_updates_counter += 1
                print(
                    f"\t Simulation loop at {timeline.get_current_time():.2f}, current elapsed time: {elapsed_time:.2f}, counter: {app_updates_counter}"
                )
            print(
                f"[SDG] Simulation loop finished in {elapsed_time:.2f} seconds at {timeline.get_current_time():.2f} with {app_updates_counter} app updates."
            )


        # SDG
        # Number of frames to capture
        num_frames = config.get("num_frames", 10)

        # Increase subframes if materials are not loaded on time, or ghosting artifacts appear on moving objects,
        # see: https://docs.omniverse.nvidia.com/extensions/latest/ext_replicator/subframes_examples.html
        rt_subframes = config.get("rt_subframes", -1)

        # Amount of simulation time to wait between captures
        sim_duration_between_captures = config.get("simulation_duration_between_captures", 0.025)

        # Initial trigger for randomizers before the SDG loop with several app updates (ensures materials/textures are loaded)
        rep.utils.send_og_event(event_name="randomize_dome_background")
        for _ in range(5):
            simulation_app.update()

        # Set the timeline parameters (start, end, no looping) and start the timeline
        timeline = omni.timeline.get_timeline_interface()
        timeline.set_start_time(0)
        timeline.set_end_time(1000000)
        timeline.set_looping(False)
        # If no custom physx scene is created, a default one will be created by the physics engine once the timeline starts
        timeline.play()
        timeline.commit()
        simulation_app.update()

        # Store the wall start time for stats
        wall_time_start = time.perf_counter()

        # Warm-up: Wait for objects to settle and stop moving 
        for i in range(200):
            if sim_duration_between_captures > 0:
                run_simulation_loop(duration=sim_duration_between_captures)
            else:
                simulation_app.update()

        # save the scene.usd file
        output_usd_path = os.path.join(config.get("save_path"), str(scene_num), "scene/scene.usd")

        try:
            stage.GetRootLayer().Export(output_usd_path)
            print(f"Scene saved to {output_usd_path}")
        except Exception as e:
            print("Errors occured while trying to save the scene.usd file")

        # Run the simulation and capture data triggering randomizations and actions at custom frame intervals
        for i in range(num_frames):
            randomize_camera_poses()
            if len(cameras) == 1:
                for cam in cameras:
                    save_camera_matrices(cam, resolution, scene_num, i, DATA_FOLDER)

            # Randomize lights locations and colors
            if i % 3 == 0:
                print(f"\t Randomizing lights")
                rep.utils.send_og_event(event_name="randomize_lights")

            # Randomize the texture of the dome background
            if i % 5 == 0:
                print(f"\t Randomizing dome background")
                rep.utils.send_og_event(event_name="randomize_dome_background")


            # Enable render products only at capture time
            if disable_render_products_between_captures:
                object_based_sdg_utils.set_render_products_updates(render_products, True, include_viewport=False)

            # Capture the current frame
            # print(f"[SDG] Capturing frame {i}/{num_frames}, at simulation time: {timeline.get_current_time():.2f}")
            rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes, pause_timeline=False)
            # if i % 5 == 0:
            #     capture_with_motion_blur_and_pathtracing(duration=0.025, num_samples=8, spp=128)
            # else:
            #     rep.orchestrator.step(delta_time=0.0, rt_subframes=rt_subframes, pause_timeline=False)

            # Disable render products between captures
            if disable_render_products_between_captures:
                object_based_sdg_utils.set_render_products_updates(render_products, False, include_viewport=False)

            # Run the simulation for a given duration between frame captures
            if sim_duration_between_captures > 0:
                run_simulation_loop(duration=sim_duration_between_captures)
            else:
                simulation_app.update()
        scene_num += 1

        # Wait for the data to be written (default writer backends are asynchronous)
        rep.orchestrator.wait_until_complete()

        # Get the stats
        wall_duration = time.perf_counter() - wall_time_start
        sim_duration = timeline.get_current_time()
        avg_frame_fps = num_frames / wall_duration
        num_captures = num_frames * num_cameras
        avg_capture_fps = num_captures / wall_duration
        print(
            f"[SDG] Captured {num_frames} frames, {num_captures} entries (frames * cameras) in {wall_duration:.2f} seconds.\n"
            f"\t Simulation duration: {sim_duration:.2f}\n"
            f"\t Simulation duration between captures: {sim_duration_between_captures:.2f}\n"
            f"\t Average frame FPS: {avg_frame_fps:.2f}\n"
            f"\t Average capture entries (frames * cameras) FPS: {avg_capture_fps:.2f}\n"
        )

        # Unsubscribe the physics overlap checks and stop the timeline
        # physx_sub.unsubscribe()
        # physx_sub = None
        timeline.stop()



