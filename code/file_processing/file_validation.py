import os
import json
from path_config import DATA_FOLDER, OUTPUT_FOLDER
from tqdm import tqdm

def process_validation_files(config, root_folder):
    invalid_cases = []
    out_of_range_prims_count = {}

    # Iterate through folders i from 0 to 999
    for i in tqdm(range(config['start_from'],config['end_with']), desc="Validate scene"):
        # Construct the path for each scene validation file
        validation_file_path = os.path.join(root_folder, f"{i}/scene/validation.json")
        
        # Check if the file exists
        if os.path.exists(validation_file_path):
            with open(validation_file_path, 'r') as f:
                try:
                    data = json.load(f)
                    
                    # Check if validation is False
                    if data.get("validation") == False:
                        invalid_cases.append(i)
                        
                        # Count occurrences of out_of_range_prims
                        out_of_range_prims = data.get("out_of_range_prims", [])
                        for prim in out_of_range_prims:
                            if prim in out_of_range_prims_count:
                                out_of_range_prims_count[prim] += 1
                            else:
                                out_of_range_prims_count[prim] = 1
                except json.JSONDecodeError:
                    print(f"Error decoding JSON in {validation_file_path}")

    # Output results to a validation_results.txt file
    with open(os.path.join(root_folder, 'validation_results.txt'), 'w') as result_file:
        # Write invalid case indexes
        result_file.write("Invalid Cases (i values):\n")
        result_file.write(", ".join(map(str, invalid_cases)) + "\n\n")
        
        # Write out_of_range_prims counts
        result_file.write("Out of Range Primitives Count:\n")
        for prim, count in out_of_range_prims_count.items():
            result_file.write(f"{prim}: {count}\n")

def validation(config):
    # Usage Example
    root_folder = str(DATA_FOLDER / 'test/camera')  # Replace this with the path to your root folder
    process_validation_files(config, root_folder)
