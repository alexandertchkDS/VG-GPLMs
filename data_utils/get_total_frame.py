import numpy as np
import cv2
import os

def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return total_frames

# Directory containing your .mp4 files
video_directory = '/home/ubuntu/Project/VG-GPLMs/source_data/sample_youtube_videos_500/'
video_file_name = '[lKUsGymRRSdY]جمعہ تین مئی کا سیربین، حصۂ سوم.mp4'
video_path = os.path.join(video_directory, video_file_name)
print(video_path)
result = get_total_frames(video_path)
print(result)