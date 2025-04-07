import os
import shutil
from path_config import DATA_FOLDER, OUTPUT_FOLDER

def move_and_rename_folders(root_folder):
    invalid_folder = os.path.join(root_folder, 'invalid')
    os.makedirs(invalid_folder, exist_ok=True)  # Create ./invalid folder if it doesn't exist
    
    # Check if validation_results.txt exists
    validation_file = os.path.join(root_folder, 'validation_results.txt')
    if not os.path.exists(validation_file):
        print("Error: validation_results.txt not found.")
        return  # Exit if the file doesn't exist

    # Read the invalid cases from validation_results.txt
    invalid_cases = []
    try: 
    
        with open(validation_file, 'r') as f:
            for line in f:
                if line.startswith("Invalid Cases"):
                    invalid_cases = [int(x.strip()) for x in f.readline().split(",")]

        # Move invalid folders to ./invalid/
        for i in invalid_cases:
            folder_to_move = os.path.join(root_folder, str(i))
            if os.path.exists(folder_to_move):
                shutil.move(folder_to_move, invalid_folder)
                print(f"Remove scene {i}")

        # Rename the remaining valid folders starting from 0
        valid_folders = [folder for folder in os.listdir(root_folder) if folder.isdigit()]
        valid_folders.sort(key=lambda x: int(x))  # Sort by numeric order
        
        for new_index, folder_name in enumerate(valid_folders):
            old_path = os.path.join(root_folder, folder_name)
            new_path = os.path.join(root_folder, str(new_index))
            os.rename(old_path, new_path)
    
        # Move validation_results.txt into the invalid folder
        shutil.move(validation_file, os.path.join(invalid_folder, 'validation_results.txt'))
        
    except:
        print("No invalid cases to remove.")
        
def remove(config):
    # Usage Example
    root_folder = str(DATA_FOLDER / 'test/camera')   # Replace with your root folder path
    move_and_rename_folders(root_folder)
