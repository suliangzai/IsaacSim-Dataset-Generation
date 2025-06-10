# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto. Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

import random

import numpy as np
from omni.kit.viewport.utility import get_active_viewport
from pxr import Gf, PhysxSchema, Usd, UsdGeom, UsdPhysics

#lzz
import os
import omni.physx
from omni.isaac.core.prims import GeometryPrimView
from omni.isaac.dynamic_control import _dynamic_control
import json


# Add transformation properties to the prim (if not already present)
def set_transform_attributes(prim, location=None, orientation=None, rotation=None, scale=None):
    if location is not None:
        if not prim.HasAttribute("xformOp:translate"):
            UsdGeom.Xformable(prim).AddTranslateOp()
        prim.GetAttribute("xformOp:translate").Set(location)
    if orientation is not None:
        if not prim.HasAttribute("xformOp:orient"):
            UsdGeom.Xformable(prim).AddOrientOp()
        prim.GetAttribute("xformOp:orient").Set(orientation)
    if rotation is not None:
        if not prim.HasAttribute("xformOp:rotateXYZ"):
            UsdGeom.Xformable(prim).AddRotateXYZOp()
        prim.GetAttribute("xformOp:rotateXYZ").Set(rotation)
    if scale is not None:
        if not prim.HasAttribute("xformOp:scale"):
            UsdGeom.Xformable(prim).AddScaleOp()
        prim.GetAttribute("xformOp:scale").Set(scale)


# Enables collisions with the asset (without rigid body dynamics the asset will be static)
def add_colliders(root_prim):
    # Iterate descendant prims (including root) and add colliders to mesh or primitive types
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.IsA(UsdGeom.Mesh) or desc_prim.IsA(UsdGeom.Gprim):
            # Physics
            if not desc_prim.HasAPI(UsdPhysics.CollisionAPI):
                collision_api = UsdPhysics.CollisionAPI.Apply(desc_prim)
            else:
                collision_api = UsdPhysics.CollisionAPI(desc_prim)
            collision_api.CreateCollisionEnabledAttr(True)
            # PhysX
            if not desc_prim.HasAPI(PhysxSchema.PhysxCollisionAPI):
                physx_collision_api = PhysxSchema.PhysxCollisionAPI.Apply(desc_prim)
            else:
                physx_collision_api = PhysxSchema.PhysxCollisionAPI(desc_prim)
            # Set PhysX specific properties
            physx_collision_api.CreateContactOffsetAttr(0.001)
            physx_collision_api.CreateRestOffsetAttr(0.0)

        # Add mesh specific collision properties only to mesh types
        if desc_prim.IsA(UsdGeom.Mesh):
            # Add mesh collision properties to the mesh (e.g. collider aproximation type)
            if not desc_prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_collision_api = UsdPhysics.MeshCollisionAPI.Apply(desc_prim)
            else:
                mesh_collision_api = UsdPhysics.MeshCollisionAPI(desc_prim)
            mesh_collision_api.CreateApproximationAttr().Set("convexHull")


# Check if prim (or its descendants) has colliders
def has_colliders(root_prim):
    for desc_prim in Usd.PrimRange(root_prim):
        if desc_prim.HasAPI(UsdPhysics.CollisionAPI):
            return True
    return False


# Enables rigid body dynamics (physics simulation) on the prim
def add_rigid_body_dynamics(prim, disable_gravity=False, angular_damping=None):
    if has_colliders(prim):
        # Physics
        if not prim.HasAPI(UsdPhysics.RigidBodyAPI):
            rigid_body_api = UsdPhysics.RigidBodyAPI.Apply(prim)
        else:
            rigid_body_api = UsdPhysics.RigidBodyAPI(prim)
        rigid_body_api.CreateRigidBodyEnabledAttr(True)
        # PhysX
        if not prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI.Apply(prim)
        else:
            physx_rigid_body_api = PhysxSchema.PhysxRigidBodyAPI(prim)
        physx_rigid_body_api.GetDisableGravityAttr().Set(disable_gravity)
        if angular_damping is not None:
            physx_rigid_body_api.CreateAngularDampingAttr().Set(angular_damping)
    else:
        print(f"Prim '{prim.GetPath()}' has no colliders. Skipping rigid body dynamics properties.")


# Add dynamics properties to the prim (if mesh or primitive) (rigid body to root + colliders to the meshes)
# https://docs.omniverse.nvidia.com/extensions/latest/ext_physics/rigid-bodies.html#rigid-body-simulation
def add_colliders_and_rigid_body_dynamics(prim, disable_gravity=False):
    # Add colliders to mesh or primitive types of the descendants of the prim (including root)
    add_colliders(prim)
    # Add rigid body dynamics properties (to the root only) only if it has colliders
    add_rigid_body_dynamics(prim, disable_gravity=disable_gravity)


# Createa  collision box area wrapping the given working area with origin in (0, 0, 0) with thickness towards outside
def create_collision_box_walls(stage, path, width, depth, height, thickness=0.5, visible=False):
    # Define the walls (name, location, size) with thickness towards outside of the working area
    walls = [
        ("floor", (0, 0, (height + thickness) / -2.0), (width, depth, thickness)),
        ("ceiling", (0, 0, (height + thickness) / 2.0), (width, depth, thickness)),
        ("left_wall", ((width + thickness) / -2.0, 0, 0), (thickness, depth, height)),
        ("right_wall", ((width + thickness) / 2.0, 0, 0), (thickness, depth, height)),
        ("front_wall", (0, (depth + thickness) / 2.0, 0), (width, thickness, height)),
        ("back_wall", (0, (depth + thickness) / -2.0, 0), (width, thickness, height)),
    ]
    for name, location, size in walls:
        prim = stage.DefinePrim(f"{path}/{name}", "Cube")
        scale = (size[0] / 2.0, size[1] / 2.0, size[2] / 2.0)
        set_transform_attributes(prim, location=location, scale=scale)
        add_colliders(prim)
        if not visible:
            UsdGeom.Imageable(prim).MakeInvisible()


# Create a random transformation values for location, rotation, and scale
def get_random_transform_values(
    loc_min=(0, 0, 0), loc_max=(1, 1, 1), rot_min=(0, 0, 0), rot_max=(360, 360, 360), scale_min_max=(0.1, 1.0)
):
    location = (
        random.uniform(loc_min[0], loc_max[0]),
        random.uniform(loc_min[1], loc_max[1]),
        random.uniform(loc_min[2], loc_max[2]),
    )
    rotation = (
        random.uniform(rot_min[0], rot_max[0]),
        random.uniform(rot_min[1], rot_max[1]),
        random.uniform(rot_min[2], rot_max[2]),
    )
    scale = tuple([random.uniform(scale_min_max[0], scale_min_max[1])] * 3)
    return location, rotation, scale


# Generate a random pose on a sphere looking at the origin
# https://docs.omniverse.nvidia.com/isaacsim/latest/reference_conventions.html
def get_random_pose_on_sphere(origin, radius, camera_forward_axis=(0, 0, -1)):
    origin = Gf.Vec3f(origin)
    camera_forward_axis = Gf.Vec3f(camera_forward_axis)

    # Generate random angles for spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.arcsin(np.random.uniform(-1, 1))

    # Spherical to Cartesian conversion
    x = radius * np.cos(theta) * np.cos(phi)
    y = radius * np.sin(phi)
    z = radius * np.sin(theta) * np.cos(phi)

    location = origin + Gf.Vec3f(x, y, z)

    # Calculate direction vector from camera to look_at point
    direction = origin - location
    direction_normalized = direction.GetNormalized()

    # Calculate rotation from forward direction (rotateFrom) to direction vector (rotateTo)
    rotation = Gf.Rotation(Gf.Vec3d(camera_forward_axis), Gf.Vec3d(direction_normalized))
    print("rotation\n",rotation)
    orientation = Gf.Quatf(rotation.GetQuat())

    return location, orientation

#lzz
def get_random_pose_on_half_sphere(origin, radius, camera_forward_axis=(0, 0, -1)):
    origin = Gf.Vec3f(origin)
    camera_forward_axis = Gf.Vec3f(camera_forward_axis)

    # Generate random angles for spherical coordinates
    theta = np.random.uniform(0, 2 * np.pi)
    phi = np.arcsin(np.random.uniform(-1, 1))

    # Spherical to Cartesian conversion
    x = radius * np.cos(theta) * np.cos(phi)
    y = radius * np.sin(phi)
    z = abs(radius * np.sin(theta) * np.cos(phi)) + 1

    location = origin + Gf.Vec3f(x, y, z)

    # Calculate direction vector from camera to look_at point
    direction = origin - location
    direction_normalized = direction.GetNormalized()

    # Calculate rotation from forward direction (rotateFrom) to direction vector (rotateTo)
    rotation = Gf.Rotation(Gf.Vec3d(camera_forward_axis), Gf.Vec3d(direction_normalized))
    # print("rotation???????\n",rotation)
    orientation = Gf.Quatf(rotation.GetQuat())
    # print("orientation!!!!!!!\n",orientation)
    # angle = rotation.GetAngle()
    # print("angle-------\n",angle)

    return location, orientation
    # return location, angle #lzz



# Enable or disable the render products and viewport rendering
def set_render_products_updates(render_products, enabled, include_viewport=False):
    for rp in render_products:
        rp.hydra_texture.set_updates_enabled(enabled)
    if include_viewport:
        get_active_viewport().updates_enabled = enabled


# lzz
def save_point_cloud(stage, output_dir, frame_number):
    """保存场景中所有物体的点云数据."""
    # 创建保存点云数据的目录
    point_cloud_dir = os.path.join(output_dir, "point_clouds")
    os.makedirs(point_cloud_dir, exist_ok=True)

    # 遍历场景中的所有Mesh对象，提取点云数据
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            mesh = UsdGeom.Mesh(prim)
            points_attr = mesh.GetPointsAttr()
            points = points_attr.Get()
            
            # 保存点云数据到文件
            if points:
                points_np = np.array(points)
                file_name = f"{prim.GetPath().pathString.replace('/', '_')}_frame_{frame_number}.npy"
                file_path = os.path.join(point_cloud_dir, file_name)
                np.save(file_path, points_np)
                print(f"Saved point cloud for {prim.GetPath().pathString} to {file_path}")


# def save_full_scene_point_cloud(stage, output_dir, frame_number):
#     """保存完整随机场景的点云数据到一个文件."""
#     # 创建保存点云数据的目录
#     point_cloud_dir = os.path.join(output_dir, "full_scene_point_clouds")
#     os.makedirs(point_cloud_dir, exist_ok=True)

#     all_points = []  # 存储整个场景的点云数据

#     # 遍历场景中的所有Mesh对象，提取点云数据
#     for prim in stage.Traverse():
#         if prim.GetTypeName() == "Mesh":
#             print(f"Mesh Prim Path: {prim.GetPath()}")
#             mesh = UsdGeom.Mesh(prim)
#             points_attr = mesh.GetPointsAttr()
#             points = points_attr.Get()

#             # 检查是否有点云数据
#             if points and len(points) > 0:
#                 points_np = np.array(points)

#                 # 获取物体的世界变换矩阵
#                 xformable = UsdGeom.Xformable(prim)
#                 local_to_world = xformable.ComputeLocalToWorldTransform(Usd.TimeCode.Default())
#                 if not local_to_world or len(local_to_world) != 16:
#                     print(f"Failed to compute transform for prim: {prim.GetPath()}")
#                     continue
                
#                 local_to_world_np = np.array(local_to_world).reshape(4, 4)
                
#                 # 如果有局部缩放，应用缩放到点云数据
#                 scale_attr = xformable.GetLocalScaleAttr()
#                 if scale_attr.HasAuthoredValue():
#                     scale = scale_attr.Get()
#                     if scale:
#                         points_np *= scale
                
#                 # 转换为齐次坐标系
#                 points_np_homogeneous = np.hstack((points_np, np.ones((points_np.shape[0], 1))))
#                 # 将点云数据转换到世界坐标系
#                 world_points = np.dot(points_np_homogeneous, local_to_world_np.T)[:, :3]

#                 # 添加到全局点云列表中
#                 all_points.extend(world_points)

#             print(f"Prim: {prim.GetPath()}")
#             print(f"Original Points: {points_np[:5]}")  # 打印前5个原始点云
#             print(f"World Points: {world_points[:5]}")  # 打印前5个转换后的点云



def save_full_scene_point_cloud(stage, output_dir, prim_path_list):
    """保存完整随机场景的点云数据到一个文件."""
    # 获取 dynamic control 接口
    dc = _dynamic_control.acquire_dynamic_control_interface()

    # 创建保存点云数据的目录
    point_cloud_dir = os.path.join(output_dir, "full_scene_point_clouds")
    os.makedirs(point_cloud_dir, exist_ok=True)

    all_points = []  # 存储整个场景的点云数据
    debug_info = {}  # 用于存储Parent Prim Path和local_to_world的字典

    # 遍历场景中的所有Mesh对象，提取点云数据
    for prim in stage.Traverse():
        if prim.GetTypeName() == "Mesh":
            print(f"Mesh Prim Path: {prim.GetPath()}")
            mesh = UsdGeom.Mesh(prim)
            points_attr = mesh.GetPointsAttr()
            points = points_attr.Get()

            # 检查是否有点云数据
            if points and len(points) > 0:
                points_np = np.array(points)

                points_np /= 100.0

                parent_prim_path = prim.GetParent().GetPath()
                parent_prim_str = str(parent_prim_path)  # 将 Sdf.Path 转换为字符串
                print(f"Parent Prim Path: {parent_prim_str}")

                # 使用 get_rigid_body 获取刚体句柄
                object_handle = dc.get_rigid_body(parent_prim_str)

                # 检查句柄是否有效
                if object_handle != _dynamic_control.INVALID_HANDLE:
                    # 获取物体的位姿（世界坐标系）
                    pose = dc.get_rigid_body_pose(object_handle)
                    position = pose.p
                    rotation = pose.r

                    # 将位置转换为 Gf.Vec3d
                    position_vec = Gf.Vec3d(position[0], position[1], position[2])

                    # 将位置和旋转转换为 4x4 的齐次变换矩阵
                    translation_matrix = Gf.Matrix4d(1.0).SetTranslate(position_vec)
                    rotation_matrix = Gf.Matrix4d(1.0).SetRotate(Gf.Rotation(Gf.Quatd(rotation[3], rotation[0], rotation[1], rotation[2])))
                    local_to_world = translation_matrix * rotation_matrix

                    print("local_to_world", local_to_world)

                    # 存储Parent Prim Path和local_to_world信息到debug_info字典中
                    # 使用GetRow(i)来获取矩阵的每一行，并将其转换为列表
                    matrix_list = [local_to_world.GetRow(i) for i in range(4)]
                    debug_info[parent_prim_str] = [list(row) for row in matrix_list]

                    # 将点云转换到世界坐标系
                    points_np_homogeneous = np.hstack((points_np, np.ones((points_np.shape[0], 1))))
                    world_points = np.dot(points_np_homogeneous, local_to_world)[:, :3]

                    # 添加到全局点云列表中
                    all_points.extend(world_points)
                else:
                    print(f"Invalid rigid body handle for {parent_prim_str}.")

    # 保存点云数据
    if all_points:
        all_points_np = np.array(all_points)
        file_name = "full_scene_point_cloud.npy"
        file_path = os.path.join(point_cloud_dir, file_name)
        np.save(file_path, all_points_np)
        print(f"Saved full scene point cloud to {file_path}")
    else:
        print("No point cloud data to save.")
    
    # 保存Parent Prim Path和local_to_world信息到文件
    debug_file_path = os.path.join(point_cloud_dir, "parent_prim_debug_info.json")
    with open(debug_file_path, 'w') as f:
        json.dump(debug_info, f, indent=4)
    print(f"Saved debug information to {debug_file_path}")






def save_pointcloud_as_npy(point_cloud, filename):
    if point_cloud.size == 0:
        print("No point cloud data to save.")
        return
    
    # 保存点云数据为 .npy 文件
    np.save(filename, point_cloud)
    print(f"Point cloud saved to {filename}.npy")


def get_scene_pointcloud(stage):
    # 获取场景中所有物体的几何信息
    geometry_prim_view = GeometryPrimView(prim_paths_expr="/World/*")
    all_vertices = []

    for prim in geometry_prim_view.prims:
        # 获取物体的世界坐标系下的顶点
        vertices = prim.get_world_vertices()
        if vertices is not None:
            all_vertices.append(vertices)

    # 合并所有顶点数据为一个点云
    if all_vertices:
        return np.concatenate(all_vertices, axis=0)
    else:
        return np.array([])  # 如果没有顶点数据则为空数组


