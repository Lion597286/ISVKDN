

import os
from pathlib import Path





os.environ.setdefault("OPENCV_FFMPEG_THREADS", "1")

import cv2
import numpy as np
from PIL import Image
from torch.utils.data import Dataset


GAF_ROOT = Path(r"F:\Dataset2\UESTC-MMEA_GAF")
VIDEO_ROOT = Path(r"G:\Dataset\Multi-modalActionRecognition\UESTC-MMEA-CL")
VIDEO_SUFFIXES = {".mp4", ".avi", ".mov"}

cv2.setNumThreads(1)
cv2.ocl.setUseOpenCL(False)


def _gaf_key(path):
    name = path.stem
    for suffix in ("_GASF_a", "_GASF_g", "_GADF_a", "_GADF_g"):
        if name.endswith(suffix):
            return name[:-len(suffix)]
    raise ValueError(f"Unrecognized GAF filename: {path}")


def _indexed_files(directory, key_fn, suffixes):
    if not directory.is_dir():
        raise FileNotFoundError(directory)
    result = {}
    for path in directory.rglob("*"):
        if path.is_file() and path.suffix.lower() in suffixes:
            key = key_fn(path)
            if key in result:
                raise ValueError(f"Duplicate sample ID {key}: {result[key]}, {path}")
            result[key] = path
    return result


def aligned_records(split, gaf_root=GAF_ROOT, video_root=VIDEO_ROOT):
    if split not in {"train", "val", "test"}:
        raise ValueError(f"Unknown split: {split}")
    gaf_root, video_root = Path(gaf_root), Path(video_root)
    acc = _indexed_files(gaf_root / f"Acc_{split}", _gaf_key, {".jpg", ".jpeg", ".png"})
    gyro = _indexed_files(gaf_root / f"Gyro_{split}", _gaf_key, {".jpg", ".jpeg", ".png"})
    videos = _indexed_files(video_root / f"video_{split}", lambda p: p.stem, VIDEO_SUFFIXES)
    keys = set(videos)
    if keys != set(acc) or keys != set(gyro):
        raise ValueError(
            f"{split} sample IDs differ: video={len(videos)}, acc={len(acc)}, "
            f"gyro={len(gyro)}, missing acc={len(keys-set(acc))}, "
            f"missing gyro={len(keys-set(gyro))}, missing video={len((set(acc)|set(gyro))-keys)}"
        )
    records = []
    for key in sorted(keys):
        label = int(acc[key].parent.name[1:]) - 1
        if not 0 <= label < 32 or int(gyro[key].parent.name[1:]) - 1 != label:
            raise ValueError(f"GAF labels disagree for {key}")
        if int(key.split("_", 1)[0]) - 1 != label:
            raise ValueError(f"Video/GAF labels disagree for {key}")
        records.append((key, acc[key], gyro[key], videos[key], label))
    return records


class UESTCPairedDataset(Dataset):
    def __init__(self, split, gaf_transform, video_transform=None, num_segments=8,
                 gaf_root=GAF_ROOT, video_root=VIDEO_ROOT, return_id=False,
                 video_only=False):
        self.records = aligned_records(split, gaf_root, video_root)
        self.gaf_transform = gaf_transform
        self.video_transform = video_transform
        self.num_segments = num_segments
        self.training = split == "train"
        self.return_id = return_id
        self.video_only = video_only
        if video_only and video_transform is None:
            raise ValueError("video_only requires a video_transform")

    def __len__(self):
        return len(self.records)

    def _video_frames(self, path):
        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            raise OSError(f"Cannot open video: {path}")
        try:
            count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if count < 1:
                raise OSError(f"Cannot determine video frame count: {path}")
            
            if self.training:
                duration = count // self.num_segments
                if duration > 0:
                    indices = np.arange(self.num_segments) * duration + np.random.randint(
                        duration, size=self.num_segments)
                elif count > self.num_segments:
                    indices = np.sort(np.random.randint(count, size=self.num_segments))
                else:
                    indices = np.zeros(self.num_segments, dtype=int)
            else:
                tick = count / float(self.num_segments)
                indices = np.array([int(tick / 2 + tick * i) for i in range(self.num_segments)])
            images = []
            for index in indices:
                cap.set(cv2.CAP_PROP_POS_FRAMES, int(index))
                ok, frame = cap.read()
                if not ok:
                    raise OSError(f"Cannot decode frame {index} in {path}")
                images.append(Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)))
            return images
        finally:
            cap.release()

    def __getitem__(self, index):
        sample_id, acc_path, gyro_path, video_path, label = self.records[index]
        if self.video_only:
            video = self.video_transform(self._video_frames(video_path))
            result = (video, label)
            return result + (sample_id,) if self.return_id else result
        with Image.open(acc_path) as image:
            acc = self.gaf_transform(image.convert("RGB"))
        with Image.open(gyro_path) as image:
            gyro = self.gaf_transform(image.convert("RGB"))
        if self.video_transform is None:
            result = (gyro, acc, label)
            return result + (sample_id,) if self.return_id else result
        video = self.video_transform(self._video_frames(video_path))
        result = (gyro, acc, video, label)
        return result + (sample_id,) if self.return_id else result
