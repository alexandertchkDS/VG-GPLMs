import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from resnext_model import model_slow_r50
from data_loader import VideoDataset
from tqdm import tqdm

def create_video_features(mp4_file_path, video_features_output_dir, batch_size=4, num_workers=4):
    current_time = time.time()
    model = model_slow_r50()
    dataset = VideoDataset(mp4_file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    features_list = []

    for video_batch in dataloader:
        #video_batch = video_batch.float().div(255.0)
        video_batch = video_batch.to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            features = model(video_batch)
            features = features.squeeze(-1).squeeze(-1).squeeze(-1)  # Remove spatial dimensions
            features_list.append(features)

    # Concatenate all the features
    all_features = torch.cat(features_list, dim=0)
    if all_features.is_cuda:
        all_features = all_features.cpu()
    all_features = all_features.numpy()

    videofile_name = os.path.splitext(os.path.basename(mp4_file_path))[0]
    output_path = os.path.join(video_features_output_dir, videofile_name)
    os.makedirs(video_features_output_dir, exist_ok=True)
    np.save(output_path, all_features)

    print(f'Video features have shape like, {all_features.shape}')
    print(f"Total time: {time.time() - current_time:.0f}")

video_features_output_dir = '/home/ubuntu/Project/VG-GPLMs/dataset/video_features/'
video_dir = '/home/ubuntu/Project/VG-GPLMs/Sample_data-small/sample_youtube_videos_500'
video_names_lst = os.listdir(video_dir)
video_paths_lst = [os.path.join(video_dir, mp4_file) for mp4_file in video_names_lst]

for mp4_file in tqdm(video_paths_lst):
    create_video_features(mp4_file, video_features_output_dir)

