import json
import os
import math
from path_config import DATA_FOLDER, CONFIG_FOLDER, CODE_FOLDER

# Function to split the dataset into smaller sets
def split_dataset(data, num_splits):
    split_size = math.ceil(len(data) / num_splits)  # Calculate how many entries per split
    return [data[i:i + split_size] for i in range(0, len(data), split_size)]

# Save each split dataset as dataset{n}.json
def save_split_datasets(data, output_folder, num_splits):
    # Ensure the output folder exists
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Split the dataset
    split_data = split_dataset(data, num_splits)
    
    # Save each split with the correct name
    for i, subset in enumerate(split_data):
        output_path = os.path.join(output_folder, f'dataset{i+1}.json')
        with open(output_path, 'w') as file:
            json.dump(subset, file, indent=4)
    print(f"Data successfully saved to {output_folder}")

# Function to read the dataset from dataset.json
def read_dataset(input_file):
    try:
        with open(input_file, 'r') as file:
            return json.load(file)
    except FileNotFoundError:
        print(f"File not found: {input_file}")
        return []
    except json.JSONDecodeError:
        print(f"Error decoding JSON in file: {input_file}")
        return []

def split(config):
    # Read the dataset from dataset.json
    folder_name = "test"
    input_file = os.path.join(DATA_FOLDER, f'{folder_name}/camera/dataset.json')
    updated_data = read_dataset(input_file)

    if updated_data:  # Proceed if data was successfully loaded
        # Define the output folder for split datasets
        output_folder = os.path.join(DATA_FOLDER, f'{folder_name}/camera/split_dataset')

        # Save the split datasets into 100 files
        save_split_datasets(updated_data, output_folder, num_splits=100)
