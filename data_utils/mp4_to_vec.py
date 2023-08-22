import torch
import numpy as np
import torch.nn as nn
import torchvision.io as io
from torchvision.models.video import r3d_18
import os
import time
import gc
gc.collect()

def extract_frames_torchvision(video_path, output_folder):
    current_time = time.time()
    # Read video
    video_tensor, _, _ = io.read_video(video_path, pts_unit="sec")
    
    # Load the pre-trained r3d_18 model
    model = r3d_18(pretrained=True)
    model = model.to("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()

    # Remove the classification head to get features before the fully connected layer
    model = torch.nn.Sequential(
    *(list(model.children())[:-1]),  # Everything except the last fc layer
    nn.Flatten(),  # Flatten the (B, 512, 1, 1, 1) tensor to (B, 512)
    nn.Linear(512, 2048)  # Expand dimensions from 512 to 2048
    ).to("cuda" if torch.cuda.is_available() else "cpu")

    # Split the tensor into chunks of 16 frames each
    split_video_tensor = video_tensor.split(16, dim=0)

    features_list = []

    with torch.no_grad():
        for chunk in split_video_tensor:
            chunk = chunk.float().div(255.0)
            
            if chunk.size(0) != 16:  # Ensuring the chunk has 16 frames
                continue
            
            # Reorder the axes
            chunk = chunk.permute(3, 0, 1, 2).unsqueeze(0)
            
            ### chunk permute shape: torch.Size([1, 16, 3, 360, 640])
            chunk = chunk.to("cuda" if torch.cuda.is_available() else "cpu")

            # Forward pass to get the features
            features = model(chunk)
            features = features.squeeze(-1).squeeze(-1).squeeze(-1)  # Remove spatial dimensions
            features_list.append(features)

    # Concatenate all the features
    all_features = torch.cat(features_list, dim=0)

    if all_features.is_cuda:
        all_features = all_features.cpu()
    all_features = all_features.numpy()

    videofile_name = os.path.basename(video_path)
    output_path = os.path.join(output_folder, videofile_name)
    os.makedirs(output_folder, exist_ok=True)
    np.save(output_path, all_features)
    print('all_features.shape', all_features.shape)  # Expected [number_of_chunks, 2048], where number_of_chunks = len(video_tensor) // 16
    print(f"Total time: {time.time() - current_time:.0f}")

# Directory containing your .mp4 files
video_directory = '/home/ubuntu/Project/VG-GPLMs/source_data/sample_youtube_videos_500/'
video_file_name = 'mXb6-AC5QJQ.mp4'
video_path = os.path.join(video_directory, video_file_name)
output_folder = '/home/ubuntu/Project/VG-GPLMs/dataset/video_features/'
extract_frames_torchvision(video_path, output_folder)


