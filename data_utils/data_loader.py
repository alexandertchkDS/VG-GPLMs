import os
from torch.utils.data import Dataset
from pytorchvideo.data.encoded_video import EncodedVideo
from torchvision.transforms import Compose, Lambda
from torchvision.transforms._transforms_video import CenterCropVideo, NormalizeVideo
from pytorchvideo.transforms import ApplyTransformToKey, ShortSideScale, UniformTemporalSubsample
from moviepy.editor import VideoFileClip

class VideoDataset(Dataset):
    def __init__(self, video_file_path):
        """
        Initialize the VideoDataset.

        :param video_file_path: Path to the video file.
        """
        if not os.path.exists(video_file_path):
            raise ValueError(f"File {video_file_path} does not exist.")
        
        self.video_file_path = video_file_path
        self.video = EncodedVideo.from_path(self.video_file_path)

        # Use moviepy to extract metadata
        with VideoFileClip(self.video_file_path) as clip:
            self.fps = clip.fps
            self.duration = clip.duration
            self.num_frames = int(self.fps * self.duration)

        self.chunk_size = 16  # The number of frames you wish to process in one go
        self.num_chunks = self.num_frames // self.chunk_size

        # Transformation specific to the slow_r50 model
        side_size = 256
        mean = [0.45, 0.45, 0.45]
        std = [0.225, 0.225, 0.225]
        crop_size = 256

        self.transform = ApplyTransformToKey(
            key="video",
            transform=Compose(
                [
                    UniformTemporalSubsample(self.chunk_size),
                    Lambda(lambda x: x / 255.0),
                    NormalizeVideo(mean, std),
                    ShortSideScale(size=side_size),
                    CenterCropVideo(crop_size=(crop_size, crop_size))
                ]
            ),
        )

    def __len__(self):
        return self.num_chunks

    def __getitem__(self, idx):
        start_chunk_sec = idx * (self.chunk_size / self.fps)
        end_chunk_sec = start_chunk_sec + (self.chunk_size / self.fps)

        # Get video clip
        video_data = self.video.get_clip(start_sec=start_chunk_sec, end_sec=end_chunk_sec)

        # Apply transformations
        video_data = self.transform(video_data)
        return video_data["video"]

# # Test
# mp4_file_path = '/home/ubuntu/Project/VG-GPLMs/Sample_data-small/video_dataset_4_files/-KXlKcGaMMo.mp4'
# video_dataset = VideoDataset(mp4_file_path)
# print(len(video_dataset))
# print(video_dataset[0].shape)  # This should give you the shape after transformations
