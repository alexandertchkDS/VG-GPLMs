import os
import cv2

def get_total_frames(video_path):
    cap = cv2.VideoCapture(video_path)
    # Get the total number of frames in the video
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    return total_frames

# Directory containing your .mp4 files
video_directory = '/home/ubuntu/Project/VG-GPLMs/source_data/sample_youtube_videos_500/'

# Get list of all .mp4 files in the directory
video_files = [f for f in os.listdir(video_directory) if f.endswith('.mp4')]

# Sort the video files by frame count
sorted_video_files = sorted(video_files, key=lambda x: get_total_frames(os.path.join(video_directory, x)))

video_file_name = sorted_video_files[0]
print(f"Video file name is:\n{video_file_name}")

n_frames = get_total_frames(os.path.join(video_directory, video_file_name))
print(f"n_frames: {n_frames}")