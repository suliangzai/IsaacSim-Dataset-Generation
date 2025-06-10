import logging
import os
from isaacsim import SimulationApp

from code.simulation import data_collection
from code.file_processing import convert_usd2ply
from code.file_processing import file_validation
from code.file_processing import move_invalid
from code.relation_analysis import occlusion_area_cam
from code.dataset_processing import generate_dataset
from code.dataset_processing import dataset_shuffle
from code.dataset_processing import split_dataset
from code.dataset_processing import dataset_readjust

from path_config import DATA_FOLDER, CONFIG_FOLDER, CODE_FOLDER

logging.basicConfig(level=logging.INFO)
os.environ["TORCH_DISTRIBUTED_DEBUG"] = "DETAIL"

# Default config dict, can be updated/replaced using json/yaml config files ('--config' cli argument)
config = {
    "launch_config": {                      # Configuration for the simulation app launch
        "renderer": "RayTracedLighting",    # Use Ray Traced Lighting for rendering
        "headless": False,                  # Whether to run the simulation without a GUI
    },
    "env_url": "",                                      # Path to the environment file (USD file for simulation)
    "working_area_size": (0.7, 0.7, 1.5),               # Dimensions of the working area in meters (x, y, z)
    "generation_area_size": (0.3, 0.3, 0.5),            # Dimensions of the object generation area (x, y, z)
    "rt_subframes": 4,                                  # Number of subframes for Ray Tracing (affects motion blur quality)
    "num_frames": 1,                                    # Total number of frames to capture
    "num_cameras": 1,                                   # Number of cameras to create
    "camera_collider_radius": 0.5,                      # Radius for collision spheres around cameras
    "disable_render_products_between_captures": False,  # Disable render products except during captures
    "simulation_duration_between_captures": 0.0,        # Duration to run simulation between captures (seconds)
    "resolution": (640, 480),                           # Resolution of the rendered images
    "camera_properties_kwargs": {                       # Properties for the camera configuration
        "focalLength": 24.0,                            # Camera focal length
        "focusDistance": 400,                           # Distance at which the camera is focused
        "fStop": 0.0,                                   # Aperture size for Depth of Field
        "clippingRange": (0.01, 10000),                 # Near and far clipping planes
    },
    "camera_look_at_target_offset": 0,              # Offset for the camera's target position
    "camera_distance_to_target_min_max": (0.5, 1),  # Min and max distance of the camera from the target
    "writer_type": "BasicWriter",                   # Writer type for storing captured data
    "writer_kwargs": {                              # Additional parameters for the writer
        "rgb": True,                                # Capture RGB data
        "use_common_output_dir": True,              # Use a common output directory for data
        "instance_segmentation": True,
        "distance_to_image_plane": True,
    },
    "save_path": DATA_FOLDER / "test/camera",                                   # Path to save captured data
    "table_url": "/Isaac/Props/Mounts/SeattleLabTable/table_instanceable.usd",  # USD file path for the table asset
    "table_pos": (0, 0, -0.4),                                                  # Position of the table in the simulation
    "random_object_info": CONFIG_FOLDER / "object_info.csv",                    # Path to CSV file containing random object info
    "min_obj_num": 2,                                                           # Minimum types of objects to generate
    "max_obj_num": 3,                                                           # Maximum types of objects to generate
    "min_count_each": 1,                                                        # Minimum count for each object type
    "max_count_each": 2,                                                        # Maximum count for each object type
    "random_table_info": CONFIG_FOLDER / "table_info.csv",                      # Path to CSV file containing random table info
    "start_from": 0,                                                            # Starting scene number for simulation
    "end_with": 10,                                                              # Ending scene number for simulation
    "dataset_format": "coco",                                                   # Format of final generated dataset.json
    "box_url":"objects_usd/Plastic_Box/scene.usdc",
    "plate_url":"objects_usd/Nail_Box_1_Lid/scene.usdc",
    "add_box": "None",                                                          # "box" or "plate"
    "some_table_invisible": True,
    "invisible_table_ratio": 0.5
}

def main():
    launch_config = config.get("launch_config", {})
    simulation_app = SimulationApp(launch_config=launch_config)

    data_collection.data_collect(config, simulation_app)
    convert_usd2ply.usd2ply(config)

    file_validation.validation(config)
    move_invalid.remove(config)
    occlusion_area_cam.generate_occlusion_cam(config)
    generate_dataset.generate_dataset(config)
    dataset_readjust.readjust(config)

    simulation_app.close()
    

if __name__ == "__main__":
    main()
