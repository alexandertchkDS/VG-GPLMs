import os
import time
import numpy as np
import torch
from torch.utils.data import DataLoader
from resnext_model import model_r3d_18
from data_loader import VideoDataset

def create_video_features(mp4_file_path, video_features_output_dir, batch_size=4, num_workers=4):
    current_time = time.time()
    model = model_r3d_18()
    dataset = VideoDataset(mp4_file_path)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    features_list = []

    for video_batch in dataloader:
        video_batch = video_batch.float().div(255.0)
        video_batch = video_batch.to("cuda" if torch.cuda.is_available() else "cpu")
        with torch.no_grad():
            features = model(video_batch)
            features = features.squeeze(-1).squeeze(-1).squeeze(-1)  # Remove spatial dimensions
            features_list.append(features)

    print(f'len(features_list): {len(features_list)}')
    print(f'features_list[0] shape: {features_list[0].shape}')

    # Concatenate all the features
    all_features = torch.cat(features_list, dim=0)
    if all_features.is_cuda:
        all_features = all_features.cpu()
    all_features = all_features.numpy()

    videofile_name = os.path.splitext(os.path.basename(mp4_file_path))[0]
    output_path = os.path.join(video_features_output_dir, videofile_name)
    os.makedirs(video_features_output_dir, exist_ok=True)
    np.save(output_path, all_features)

    print('all_features.shape', all_features.shape)
    print(f"Total time: {time.time() - current_time:.0f}")

mp4_file_path = '/home/ubuntu/Project/VG-GPLMs/Sample_data-small/video_dataset_4_files/-KXlKcGaMMo.mp4'
video_features_output_dir = '/home/ubuntu/Project/VG-GPLMs/dataset/video_features/'
create_video_features(mp4_file_path, video_features_output_dir)




