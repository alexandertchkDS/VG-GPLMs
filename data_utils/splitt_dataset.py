import os
import random

def load_and_split_data(dataset_folder):
    
    # Function to load data from a file
    def load_data(file_path):
        with open(file_path, "r") as f:
            lines = f.readlines()
        video_data = {}
        for line in lines:
            video_id, text = line.strip().split(" ", 1)
            video_data[video_id] = text
        return video_data
    
    # Function to write data to a file
    def write_data(data, file_path):
        with open(file_path, "w") as f:
            for video_id in data:
                f.write(f"{video_id} {data[video_id]}\n")
    
    # Load the video IDs and text descriptions from tran.tok.txt and desc.tok.txt
    tran_tok_file = os.path.join(dataset_folder, "tran.tok.txt")
    desc_tok_file = os.path.join(dataset_folder, "desc.tok.txt")

    tran_data = load_data(tran_tok_file)
    desc_data = load_data(desc_tok_file)

    # Get the list of unique video IDs and shuffle them
    video_ids = list(set(tran_data.keys()) & set(desc_data.keys()))
    random.shuffle(video_ids)

    # Define the split ratios
    train_ratio = 0.7
    val_ratio = 0.15
    # Calculate the number of video IDs for each split
    num_videos = len(video_ids)
    num_train = int(num_videos * train_ratio)
    num_val = int(num_videos * val_ratio)

    # Split the video IDs into train, validation, and test sets
    train_ids = video_ids[:num_train]
    val_ids = video_ids[num_train:num_train+num_val]
    test_ids = video_ids[num_train+num_val:]

    # Prepare the data for each split
    train_tran_data = {video_id: tran_data[video_id] for video_id in train_ids}
    val_tran_data = {video_id: tran_data[video_id] for video_id in val_ids}
    test_tran_data = {video_id: tran_data[video_id] for video_id in test_ids}

    train_desc_data = {video_id: desc_data[video_id] for video_id in train_ids}
    val_desc_data = {video_id: desc_data[video_id] for video_id in val_ids}
    test_desc_data = {video_id: desc_data[video_id] for video_id in test_ids}

    # Write the data for each split into separate files
    write_data(train_tran_data, os.path.join(dataset_folder, "train_tran.tok.txt"))
    write_data(val_tran_data, os.path.join(dataset_folder, "val_tran.tok.txt"))
    write_data(test_tran_data, os.path.join(dataset_folder, "test_tran.tok.txt"))
    write_data(train_desc_data, os.path.join(dataset_folder, "train_desc.tok.txt"))
    write_data(val_desc_data, os.path.join(dataset_folder, "val_desc.tok.txt"))
    write_data(test_desc_data, os.path.join(dataset_folder, "test_desc.tok.txt"))

# Create the directory if it doesn't exist
dataset_folder = "/home/alexandertchk/VSCode/multimodal/VG-GPLMs/dataset-prod/text"
if not os.path.exists(dataset_folder):
    os.makedirs(dataset_folder)

load_and_split_data(dataset_folder)
