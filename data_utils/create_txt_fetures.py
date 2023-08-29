import pandas as pd
import os

def create_text_features(csv_file_path, dataset_path):
    # Load the CSV into a pandas DataFrame
    df = pd.read_csv(csv_file_path)

    # Ensure 'id' and 'description' columns are of type string
    df['id'] = df['id'].astype(str)
    df['description'] = df['description'].astype(str)

    # Remove newline characters and extra spaces from 'id', 'description', and 'transcriptions' columns
    df['id'] = df['id'].str.replace('\n', ' ').str.strip()
    df['description'] = df['description'].str.replace('\n', ' ').str.strip()
    df['transcriptions'] = df['transcriptions'].astype(str).str.replace('\n', ' ').str.strip()

    desc_path = os.path.join(dataset_path, 'desc.tok.txt')
    tran_path = os.path.join(dataset_path, 'tran.tok.txt')
    # Save to desc.tok.txt
    with open(desc_path, 'w') as desc, \
         open(tran_path, 'w') as tran:
        for _, row in df.iterrows():
            desc.write(f"{row['id']} {row['description']}\n")
            tran.write(f"{row['id']} {row['transcriptions']}\n")

    print("File desc.tok.txt has been created!")
    print("File trans.tok.txt has been created!")

        
csv_file_path = '/home/ubuntu/Project/VG-GPLMs/Sample_data-small/sample_youtube_data_500.csv'
dataset_path = '/home/ubuntu/Project/VG-GPLMs/dataset/'
create_text_features(csv_file_path, dataset_path)