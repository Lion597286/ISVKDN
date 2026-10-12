import cv2
import os
from pathlib import Path


video_root = Path("G:/Dataset/Multi-modalActionRecognition/UESTC-MMEA/video_test")
frames_root = Path("G:/Dataset/Multi-modalActionRecognition/UESTC-MMEA/frames_test")


os.makedirs(frames_root, exist_ok=True)

print(f"目标帧文件夹已创建：{frames_root}")


video_folders = sorted(video_root.iterdir(), key=lambda x: int(x.name.split('_')[0]))


for video_folder in video_folders:
    if not video_folder.is_dir():
        continue

    
    frame_folder = frames_root / video_folder.name
    os.makedirs(frame_folder, exist_ok=True)

    print(f"已创建帧文件夹：{frame_folder}")

    
    for video_file in video_folder.iterdir():
        if not video_file.is_file() or not video_file.suffix.lower() in ['.mp4', '.avi', '.mov']:
            continue

        
        video_frames_folder = frame_folder / video_file.stem
        os.makedirs(video_frames_folder, exist_ok=True)

        print(f"已创建视频帧文件夹：{video_frames_folder}")

        
        cap = cv2.VideoCapture(str(video_file))
        frame_count = 1
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            
            frame_path = video_frames_folder / f"{frame_count:06d}.jpg"
            cv2.imwrite(str(frame_path), frame)
            frame_count += 1

        
        cap.release()
        print(f"视频帧提取完成：{video_file}")

print("所有视频帧提取完成。")