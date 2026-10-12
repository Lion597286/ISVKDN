






from __future__ import annotations

import csv
import json
from pathlib import Path
import random
import time

import cv2
import numpy as np
import torch
import torchvision
import torchvision.transforms as tv_transforms
from torch.utils.data import DataLoader

import model.backbone as backbone
from stage2_train_student import classification_metrics, load_state_dict_flexible
from transforms import GroupCenterCrop, GroupNormalize, GroupScale, Stack, ToTorchFormatTensor
from uestc_paired_dataset import UESTCPairedDataset


PROJECT_ROOT = Path(__file__).resolve().parent
OUTPUT_ROOT = PROJECT_ROOT / 'output' / 'stage2'
GAF_ROOT = Path(r'F:\Dataset2\UESTC-MMEA_GAF')
VIDEO_ROOT = Path(r'G:\Dataset\Multi-modalActionRecognition\UESTC-MMEA-CL')
BATCH_SIZE = 32
NUM_WORKERS = 8
SEED = 2025

EXPERIMENTS = {
    'A0': (OUTPUT_ROOT / 'students' / 'A0_video_baseline' / 'seed_2025',
           'best_video_acc.pth'),
    'A1': (OUTPUT_ROOT / 'students' / 'A1_soft_kd_only' / 'seed_2025',
           'best_video_acc.pth'),
    'A2': (OUTPUT_ROOT / 'students' / 'A2_sgsm_only' / 'seed_2025',
           'best_video_acc.pth'),
    'A3': (OUTPUT_ROOT / 'students' / 'A3_full_isvkdn' / 'seed_2025',
           'best_video_acc.pth'),
    'LEGACY': (
        PROJECT_ROOT / 'output' /
        'st_1_MMEA_a_g_v_teacher_SemanticFusionVGG16_student_TRN_arch_'
        'BNInception_seg8_epochs70_batch16_lr0.001_dropout0.8',
        'best_acc.pth'),
}


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)
    cv2.setNumThreads(1)


def save_json(path, value):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def save_outputs(output_dir, split, sample_ids, labels, predictions, logits, metrics):
    prefix = f'{split}_deterministic'
    with open(output_dir / f'{prefix}_predictions.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sample_id', 'true_label', 'predicted_label'])
        writer.writerows(zip(sample_ids, labels.tolist(), predictions.tolist()))
    np.save(output_dir / f'{prefix}_logits.npy', logits.astype(np.float32))
    save_json(output_dir / f'{prefix}_metrics.json', metrics)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    logits_all, labels_all, sample_ids_all = [], [], []
    torch.cuda.synchronize()
    start = time.perf_counter()
    for videos, labels, sample_ids in loader:
        videos = videos.to(device, non_blocking=True)
        with torch.cuda.amp.autocast(enabled=True):
            logits = model(videos, return_features=False)
        logits_all.append(logits.float().cpu())
        labels_all.append(labels.cpu())
        sample_ids_all.extend(list(sample_ids))
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    logits = torch.cat(logits_all).numpy()
    labels = torch.cat(labels_all).numpy()
    predictions = logits.argmax(axis=1)
    metrics = classification_metrics(labels, predictions)
    metrics['end_to_end_seconds'] = elapsed
    metrics['end_to_end_samples_per_second'] = len(labels) / max(elapsed, 1e-12)
    return sample_ids_all, labels, predictions, logits, metrics


def main():
    if not torch.cuda.is_available():
        raise RuntimeError('Deterministic evaluation requires a CUDA GPU')
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    device = torch.device('cuda')

    probe = backbone.TRN(
        32, 8, 'RGB', base_model='BNInception', consensus_type='TRNmultiscale',
        dropout=0.8, img_feature_dim=768, partial_bn=True)
    normalize = GroupNormalize(probe.input_mean, probe.input_std)
    video_transform = torchvision.transforms.Compose([
        GroupScale(int(probe.scale_size)),
        GroupCenterCrop(probe.crop_size),
        Stack(roll=True),
        ToTorchFormatTensor(div=False),
        normalize,
    ])
    del probe
    gaf_transform = tv_transforms.Compose([
        tv_transforms.Resize((224, 224)),
        tv_transforms.ToTensor(),
    ])

    def make_loader(split):
        return DataLoader(
            UESTCPairedDataset(
                split, gaf_transform, video_transform, num_segments=8,
                gaf_root=GAF_ROOT, video_root=VIDEO_ROOT,
                return_id=True, video_only=True),
            batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS,
            pin_memory=True, persistent_workers=True, prefetch_factor=4,
            worker_init_fn=seed_worker)

    val_loader = make_loader('val')
    test_loader = make_loader('test')
    rows = []
    for experiment_id, (output_dir, checkpoint_name) in EXPERIMENTS.items():
        cached_path = output_dir / 'metrics_deterministic.json'
        if cached_path.is_file():
            with open(cached_path, encoding='utf-8') as handle:
                split_metrics = json.load(handle)
            rows.append({
                'experiment': experiment_id,
                'validation_accuracy': split_metrics['validation']['accuracy'],
                'test_accuracy': split_metrics['test']['accuracy'],
                'test_macro_f1': split_metrics['test']['macro_f1'],
            })
            print(f'{experiment_id}: using cached deterministic metrics', flush=True)
            continue
        checkpoint = output_dir / checkpoint_name
        if not checkpoint.is_file():
            raise FileNotFoundError(checkpoint)
        model = backbone.TRN(
            32, 8, 'RGB', base_model='BNInception', consensus_type='TRNmultiscale',
            dropout=0.8, img_feature_dim=768, partial_bn=True).to(device)
        load_state_dict_flexible(model, checkpoint)
        split_metrics = {}
        for split, loader in (('validation', val_loader), ('test', test_loader)):
            sample_ids, labels, predictions, logits, metrics = evaluate(model, loader, device)
            save_outputs(output_dir, split, sample_ids, labels, predictions, logits, metrics)
            split_metrics[split] = metrics
            print(f'{experiment_id} {split}: acc={metrics["accuracy"]:.4f} '
                  f'macroF1={metrics["macro_f1"]:.4f}', flush=True)
        save_json(output_dir / 'metrics_deterministic.json', split_metrics)
        with open(output_dir / 'result_deterministic.txt', 'w', encoding='utf-8') as handle:
            handle.write('Deterministic Validation Video Acc: %.4f\n' %
                         split_metrics['validation']['accuracy'])
            handle.write('Deterministic Test Video Acc: %.4f\n' %
                         split_metrics['test']['accuracy'])
            handle.write('Deterministic Test Macro F1: %.4f\n' %
                         split_metrics['test']['macro_f1'])
        rows.append({
            'experiment': experiment_id,
            'validation_accuracy': split_metrics['validation']['accuracy'],
            'test_accuracy': split_metrics['test']['accuracy'],
            'test_macro_f1': split_metrics['test']['macro_f1'],
        })
        del model
        torch.cuda.empty_cache()

    summary_path = OUTPUT_ROOT / 'deterministic_summary.csv'
    with open(summary_path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print('Saved:', summary_path, flush=True)


if __name__ == '__main__':
    main()
