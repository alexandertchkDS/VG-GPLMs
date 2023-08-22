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


def create_txt_files(csv_file_path, dataset_path):
    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Ensure 'id' and 'description' columns are of type string
    df['id'] = df['id'].astype(str)
    df['description'] = df['description'].astype(str)

    # Remove newline characters and extra spaces from 'id' and 'description' columns
    df['id'] = df['id'].str.replace('\n', ' ').str.strip()
    df['description'] = df['description'].str.replace('\n', ' ').str.strip()

    desc_path = os.path.join(dataset_path, 'desc.tok.txt')
    tran_path = os.path.join(dataset_path, 'tran.tok.txt')
    # Save to desc.tok.txt
    with open(desc_path, 'w') as desc, \
         open('tran_path', 'w') as tran:
        for _, row in df.iterrows():
            desc.write(f"{row['id']} {row['description']}\n")
            tran.write(f"{row['id']} {row['transcriptions']}\n")

    print("File desc.tok.txt has been created!")
    print("File trans.tok.txt has been created!")