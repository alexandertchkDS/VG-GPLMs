import pandas as pd
import os

def rename_files(video_dir_path): 
    # Iterate over all files in the directory
    for filename in os.listdir(video_dir_path):
        # Check if the file has an .mp4 extension
        if filename.endswith(".mp4"):
            # Extract the video_id from the filename
            video_id = filename.split(']')[0][1:]
            # New name with just the video_id
            new_name = video_id + ".mp4"
            # Full paths for renaming
            old_path = os.path.join(video_dir_path, filename)
            new_path = os.path.join(video_dir_path, new_name)
            # Rename the file
            os.rename(old_path, new_path)
    print("Renaming done!")