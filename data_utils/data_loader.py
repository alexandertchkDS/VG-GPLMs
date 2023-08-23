import os
import gc
import numpy as np
from torch.utils.data import Dataset
import torchvision.io as io

class VideoDataset(Dataset):
    def __init__(self, video_file_path):
        """
        Initialize the VideoDataset.

        :param video_dir_path: Path to the directory containing video files.
        """
        if not os.path.exists(video_file_path):
            raise ValueError(f"Directory {video_file_path} does not exist.")
        
        self.video_file_path = video_file_path
        self.chunks = self._create_chunks()

    def __len__(self):
        return len(self.chunks)
    
    def _create_chunks(self):
        video_tensor, _, _ = io.read_video(self.video_file_path, pts_unit="sec")
        video_tensor = video_tensor.permute(3, 0, 1, 2)
        chunks = list(video_tensor.split(16, dim=1))
        if chunks[-1].shape[1] != 16:
            chunks = chunks[:-1]
        del video_tensor
        gc.collect()
        return chunks

    def __getitem__(self, idx):
        return self.chunks[idx]






