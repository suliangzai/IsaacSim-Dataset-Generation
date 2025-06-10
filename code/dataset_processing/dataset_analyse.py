import os
import json
import csv
from collections import Counter
import matplotlib.pyplot as plt
from path_config import DATA_FOLDER, CONFIG_FOLDER, CODE_FOLDER


# Function to load object descriptions from CSV file
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

# Function to analyze the dataset for specific questions and answers
def analyze_occlusion_data(dataset_path, object_info_mapping):
    # Load the dataset
    with open(dataset_path, 'r') as file:
        data = json.load(file)

    print(f"Total number of data points: {len(data)}")

    # Sort object descriptions by length in descending order to ensure the longest match is checked first
    sorted_descriptions = sorted(object_info_mapping.values(), key=len, reverse=True)

    # Initialize counters
    what_occludes_counter = Counter()
    is_occluded_counter = Counter()
    object_mentions_counter = Counter()

    # Loop through the dataset and count relevant answers
    for entry in data:
        dialogue = entry['dialogue']
        
        for dialog in dialogue:
            # Analyze "What occludes XXX?" question
            if dialog['from'] == 'gpt' and 'is not occluded by any objects' in dialog['value']:
                what_occludes_counter['not_occluded'] += 1
            elif dialog['from'] == 'gpt' and 'occluded by' in dialog['value']:
                what_occludes_counter['occluded'] += 1

            # Analyze "Is XXX occluded by other objects?" question
            if dialog['from'] == 'gpt' and dialog['value'] == 'No.':
                is_occluded_counter['no'] += 1
            elif dialog['from'] == 'gpt' and dialog['value'] == 'Yes.':
                is_occluded_counter['yes'] += 1

            # Count object mentions based on the longest matching 'description' from object_info_mapping
            for description in sorted_descriptions:
                if description in dialog['value']:
                    object_mentions_counter[description] += 1
                    # Break to avoid counting shorter descriptions (e.g., 'plant' after 'plant pot')
                    break

    return what_occludes_counter, is_occluded_counter, object_mentions_counter

# Function to visualize the proportions using a pie chart
def visualize_proportions(what_occludes_counter, is_occluded_counter):
    # What occludes proportions
    labels_what_occludes = ['Not Occluded', 'Occluded']
    sizes_what_occludes = [what_occludes_counter['not_occluded'], what_occludes_counter['occluded']]

    # Is occluded proportions
    labels_is_occluded = ['No', 'Yes']
    sizes_is_occluded = [is_occluded_counter['no'], is_occluded_counter['yes']]

    # Create subplots for both pie charts
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))

    # Pie chart for "What occludes XXX?"
    axs[0].pie(sizes_what_occludes, labels=labels_what_occludes, autopct='%1.1f%%', startangle=90)
    axs[0].axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    axs[0].set_title('Proportion of "What occludes XXX?" Answers')

    # Pie chart for "Is XXX occluded by other objects?"
    axs[1].pie(sizes_is_occluded, labels=labels_is_occluded, autopct='%1.1f%%', startangle=90)
    axs[1].axis('equal')  # Equal aspect ratio ensures the pie chart is circular.
    axs[1].set_title('Proportion of "Is XXX occluded?" Answers')

    # Display the plots
    plt.tight_layout()
    plt.show()

# Function to display the top 10 most mentioned objects
def display_top_mentioned_objects(object_mentions_counter):
    # Get the top 10 most mentioned objects
    top_10_objects = object_mentions_counter.most_common(10)

    print("Top 10 Most Mentioned Objects:")
    for obj, count in top_10_objects:
        print(f"{obj}: {count} mentions")

    # Create a bar chart for visualization
    objects, counts = zip(*top_10_objects)
    plt.figure(figsize=(10, 6))
    plt.barh(objects, counts, color='skyblue')
    plt.xlabel("Number of Mentions")
    plt.ylabel("Objects")
    plt.title("Top 10 Most Mentioned Objects in the Dataset")
    plt.gca().invert_yaxis()  # Invert y-axis for better readability
    plt.show()

# Load, analyze, and visualize the dataset
dataset_path = os.path.join(DATA_FOLDER, 'test/camera/dataset.json')
object_info_path = os.path.join(CONFIG_FOLDER, 'object_info.csv')

# Load object descriptions
object_info_mapping = load_object_info(object_info_path)

# Analyze the dataset
what_occludes_counter, is_occluded_counter, object_mentions_counter = analyze_occlusion_data(dataset_path, object_info_mapping)

# Visualize the proportions of occlusion answers
visualize_proportions(what_occludes_counter, is_occluded_counter)

# Display the top 10 most mentioned objects
display_top_mentioned_objects(object_mentions_counter)
