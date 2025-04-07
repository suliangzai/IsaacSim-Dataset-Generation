# Dataset Generation  
- `data_collection.py`  
- `convert_usd2ply.py`  
- `file_validation.py`  
- `move_invalid.py`  
- `occlusion_area_cam.py`
- `generate_dataset.py`  
- `dataset_shuffle.py`  
- `split_dataset.py`  

To generate a dataset, run the following codes sequentially.

## `data_collection.py`  
Collects data with Isaac Sim. The data includes one `scene.usd` file of the scene, ten RGB images from different views, and the corresponding ten camera matrices. Refer to [Issac-Sim tutorial](https://docs.omniverse.nvidia.com/isaacsim/latest/replicator_tutorials/tutorial_replicator_object_based_sdg.html) for origional example. 

Run the following command: 
```bash
~/.local/share/ov/pkg/isaac-sim-4.0.0/python.sh ~/Projects/Simulation/code/simulation/data_collection.py
```

> Need to run this command under "Simulation" file.

By default, this command creates a data file located at `<path_to_project>/data/test/camera`. You can customize the default path by modifying the `"save_path"` configuration. 

The following tree map illustrates a file structure under `<path_to_project>/data/test/camera` after running the command, with each `*` marking the newly created files or folders. 
```
<path_to_project>/data/test/camera
├── 0*
│   ├── camera_<0000~0009>.npz*
│   ├── metadata.txt*
│   ├── rgb_<0000~0009>.png*
│   └── scene*
│       ├── object_data.txt*
│       └── scene.usd*
├── 1*
│   ├── ...*
│   ...*
...*
```
Each folder (`0`, `1`, ...) represents a different scene. Each scene contains:
- **`camera_<0000~0009>.npz`**: Contains camera matrices.
- **`rgb_<0000~0009>.png`**: Corresponding RGB images.
- **`object_data.txt`**: Stores object information.
- **`scene.usd`**: The scene file.

You can open the `scene.usd` file in Isaac-Sim to view the scene.

## `convert_usd2ply.py`  
Converts the `scene.usd` file of a scene into a `scene.ply` file, which represents the 3D point cloud of the scene. Run the following command: 
```bash
~/.local/share/ov/pkg/isaac-sim-4.0.0/python.sh ~/Projects/Simulation/code/file_processing/convert_usd2ply.py
```
The following tree map illustrates a file structure under `<path_to_project>/data/test/camera` after running the command, with each `*` marking the newly created files or folders. 
```
<path_to_project>/data/test/camera
├── 0
│   ├── camera_<0000~0009>.npz
│   ├── metadata.txt
│   ├── rgb_<0000~0009>.png
│   └── scene
│       ├── label_mapping.txt*
│       ├── object_data.txt
│       ├── scene.ply*
│       ├── scene.usd
│       └── validation.json*
├── 1
│   ├── ...*
│   ...*
...*
```
This code generates the following files:
- **`label_mapping.txt`**: Records the mapping between objects and their labels.
- **`scene.ply`**: Contains the 3D point cloud of the scene.
- **`validation.json`**: A validation file for the scene.


## `file_validation.py`  
Checks each scene and determines whether it is invalid. A scene is considered invalid if clipping occurs. Run the following command: 
```bash
~/anaconda3/envs/lzz/bin/python ~/Projects/Simulation/code/file_processing/file_validation.py
```
The following tree map illustrates a file structure under `<path_to_project>/data/test/camera` after running the command, with each `*` marking the newly created files or folders. 
```
<path_to_project>/data/test/camera
├── 0
│   ├── camera_<0000~0009>.npz
│   ├── metadata.txt
│   ├── rgb_<0000~0009>.png
│   └── scene
│       ├── label_mapping.txt
│       ├── object_data.txt
│       ├── scene.ply
│       ├── scene.usd
│       └── validation.json
├── 1
│   ├── ...
│   ...
...
└── validation_results.txt*
```
This code generates the following files:
- **`validation_results.txt`**: Records the overall invalid scenes

## `move_invalid.py`  
Moves invalid scenes identified by the `file_validation.py` step into a folder named `invalid`. Run the following command: 
```bash
~/anaconda3/envs/lzz/bin/python ~/Projects/Simulation/code/file_processing/move_invalid.py
```
The following tree map illustrates a file structure under `<path_to_project>/data/test/camera` after running the command, with each `*` marking the newly created files or folders. 
```
<path_to_project>/data/test/camera
├── 0
│   ├── camera_<0000~0009>.npz
│   ├── metadata.txt
│   ├── rgb_<0000~0009>.png
│   └── scene
│       ├── label_mapping.txt
│       ├── object_data.txt
│       ├── scene.ply
│       ├── scene.usd
│       └── validation.json
├── 1
│   ├── ...
│   ...
...
├── validation_results.txt
└── invalid*
    ├── 0*
    │   ├── ...*
    |   ...*
    ...*
```
This code performs the following actions:

1. Moves all invalid folders (scenes) into a new folder named `invalid`, based on the contents of `validation_results.txt`.
2. Renames all remaining folders (scenes), starting from `0`.


## `occlusion_area_cam.py`
Generates occlusion tree for each camera view. Run the following command: 
```bash
~/anaconda3/envs/lzz/bin/python ~/Projects/Simulation/code/data_analysis/occlusion_area_cam.py
```
The following tree map illustrates a file structure under `<path_to_project>/data/test/camera` after running the command, with each `*` marking the newly created files or folders. 
```
<path_to_project>/data/test/camera
├── 0
│   ├── camera_<0000~0009>.npz
│   ├── metadata.txt
│   ├── rgb_<0000~0009>.png
│   └── scene
│       ├── label_mapping.txt
│       ├── object_data.txt
│       ├── occlusion*
│       │   └── occlusion_tree_<0000~0009>.json*
│       ├── position*
│       │   └── position_<0000~0009>.json*
│       ├── scene.ply
│       ├── scene.usd
│       └── validation.json
├── 1
│   ├── ...*
│   ...*
...*
├── validation_results.txt
└── invalid
    ├── 0
    │   ├── ...
    |   ...
    ...
```
This code generates the following files:
- **`occlusion_tree_<0000~0009>.json`**: Records the occlusion trees for each view.
- **`position_<0000~0009>.json`**: Records the positions of each object in each view.


## `generate_dataset.py`  
Generates occlusion and position data for each object using the `scene.ply` file. Creates a dataset named `dataset.json`. Run the following command: 
```bash
~/anaconda3/envs/lzz/bin/python ~/Projects/Simulation/code/simulation/generate_dataset.py
```
The following tree map illustrates a file structure under `<path_to_project>/data/test/camera` after running the command, with each `*` marking the newly created files or folders. 
```
<path_to_project>/data/test/camera
├── 0
│   ├── camera_<0000~0009>.npz
│   ├── metadata.txt
│   ├── rgb_<0000~0009>.png
│   └── scene
│       ├── label_mapping.txt
│       ├── object_data.txt
│       ├── occlusion
│       │   └── occlusion_tree_<0000~0009>.json
│       ├── position
│       │   └── position_<0000~0009>.json
│       ├── scene.ply
│       ├── scene.usd
│       └── validation.json
├── 1
│   ├── ...
│   ...
...
├── dataset.json*
├── validation_results.txt
└── invalid
    ├── 0
    │   ├── ...
    |   ...
    ...
```
This code generates the following files:
- **`dataset.json`**: Contains the dataset.

## `dataset_shuffle.py`  
Shuffles the sequence of data in `dataset.json` randomly. Run the following command: 
```bash
~/anaconda3/envs/lzz/bin/python ~/Projects/Simulation/code/simulation/dataset_shuffle.py
```
This code shuffles the data in `dataset.json`.

## `split_dataset.py`  
Splits the data in `dataset.json` into several files named `dataset<n>.json`, where `n` is a sequential number. Run the following command: 
```bash
~/anaconda3/envs/lzz/bin/python ~/Projects/Simulation/code/simulation/split_dataset.py
```
The following tree map illustrates a file structure under `<path_to_project>/data/test/camera` after running the command, with each `*` marking the newly created files or folders. 
```
<path_to_project>/data/test/camera
├── 0
│   ├── camera_<0000~0009>.npz
│   ├── metadata.txt
│   ├── rgb_<0000~0009>.png
│   └── scene
│       ├── label_mapping.txt
│       ├── object_data.txt
│       ├── occlusion
│       │   └── occlusion_tree_<0000~0009>.json
│       ├── position
│       │   └── position_<0000~0009>.json
│       ├── scene.ply
│       ├── scene.usd
│       └── validation.json
├── 1
│   ├── ...
│   ...
...
├── dataset.json
├── dataset<0~n>.json*
├── validation_results.txt
└── invalid
    ├── 0
    │   ├── ...
    |   ...
    ...
```
This code generates the following files:
- **`dataset<0~n>.json`**: n+1 sub-datasets split form `dataset.json`.