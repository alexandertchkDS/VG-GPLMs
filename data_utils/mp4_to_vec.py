import torch
import numpy as np
import torch.nn as nn
import torchvision.io as io
from torchvision.models.video import r3d_18, R3D_18_Weights
import os

def extract_frames_torchvision(video_path, output_folder):
    # Read video
    video_tensor, audio_tensor, video_info = io.read_video(video_path, pts_unit="sec", output_format='TCHW')
    print("*")
    print(f"video_tensor type: {type(video_tensor)}")
    print(f'video_tensor shape: {video_tensor.shape}')
    print(f'video_tensor dtype: {video_tensor.dtype}')
    video_tensor = video_tensor[16:16*3, :, :, :]
    print("*")
    print(f"video_tensor type: {type(video_tensor)}")
    print(f'video_tensor shape: {video_tensor.shape}')
    print(f'video_tensor dtype: {video_tensor.dtype}')

# If you want to keep the tensor in float format for further processing, consider writing processed chunks to disk and loading as needed.

    print(f"video_tensor shape is {video_tensor.shape}\n\n\n")

    # Load the pre-trained r3d_18 model
    model = r3d_18(weights=R3D_18_Weights.KINETICS400_V1)
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

    print('eval')
    with torch.no_grad():
        for chunk in split_video_tensor:
            chunk = chunk.float().div(255.0)
            if chunk.size(0) != 16:  # Ensuring the chunk has 16 frames
                continue
            # Reorder the axes
            chunk = chunk.permute(1, 0, 2, 3).unsqueeze(0)  # Shape: [1, 3, 16, height, width]
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

    np.save(output_path, all_features)
    print('all_features.shape', all_features.shape)  # Expected [number_of_chunks, 2048], where number_of_chunks = len(video_tensor) // 16


video_path = '/home/alexandertchk/VSCode/multimodal/VG-GPLMs/dataset-prod/[-KXlKcGaMMo]Stormy winds pick up the car and thrown away.mp4'
output_folder = '/home/alexandertchk/VSCode/multimodal/VG-GPLMs/dataset-prod/'
extract_frames_torchvision(video_path, output_folder)
