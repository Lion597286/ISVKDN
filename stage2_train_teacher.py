

from __future__ import annotations

import argparse
import json
import logging
import os
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

from model.backbone import SemanticFusionVGG16
from uestc_paired_dataset import UESTCPairedDataset


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaf_root', default=r'F:\Dataset2\UESTC-MMEA_GAF')
    parser.add_argument('--video_root', default=r'G:\Dataset\Multi-modalActionRecognition\UESTC-MMEA-CL')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--fusion_mode', choices=['mean', 'concat', 'concat_gate', 'full'], default='full')
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=80)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--prefetch_factor', type=int, default=4)
    parser.add_argument('--print_freq', type=int, default=20)
    parser.add_argument('--early_stop_patience', type=int, default=10)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--smoke', action='store_true')
    return parser.parse_args()


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


def state_dict_cpu(model):
    source = model.module if isinstance(model, nn.DataParallel) else model
    return {key: value.detach().cpu() for key, value in source.state_dict().items()}


def save_json(path, value):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def main():
    args = parse_args()
    if args.early_stop_patience < 1:
        raise ValueError('early_stop_patience must be at least 1')
    if args.prefetch_factor < 1 or args.print_freq < 1:
        raise ValueError('prefetch_factor and print_freq must be at least 1')
    if not torch.cuda.is_available():
        raise RuntimeError('Stage-2 training requires a CUDA GPU')

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format='%(message)s',
        handlers=[logging.StreamHandler(sys.stdout),
                  logging.FileHandler(output_dir / 'log.txt', encoding='utf-8')])

    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_amp = not args.no_amp
    device = torch.device('cuda')

    config = vars(args).copy()
    config.update(dataset='UESTC-MMEA-CL', num_classes=32,
                  split={'train': 4553, 'val': 653, 'test': 1316},
                  selection_metric='combined_teacher_validation_accuracy')
    save_json(output_dir / 'config.json', config)

    gaf_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    generator = torch.Generator().manual_seed(args.seed)

    def make_loader(split, shuffle, workers):
        loader_options = dict(
            batch_size=args.batch,
            shuffle=shuffle,
            num_workers=workers,
            pin_memory=True,
            persistent_workers=workers > 0,
            worker_init_fn=seed_worker,
            generator=generator if shuffle else None)
        if workers > 0:
            loader_options['prefetch_factor'] = args.prefetch_factor
        return DataLoader(
            UESTCPairedDataset(split, gaf_transform, gaf_root=args.gaf_root,
                               video_root=args.video_root),
            **loader_options)

    train_loader = make_loader('train', True, args.workers)
    val_loader = make_loader('val', False, max(1, args.workers // 2) if args.workers else 0)
    test_loader = make_loader('test', False, 0)
    actual_counts = {
        'train': len(train_loader.dataset),
        'val': len(val_loader.dataset),
        'test': len(test_loader.dataset),
    }
    if actual_counts != config['split']:
        raise ValueError('Official split count mismatch: %s' % actual_counts)

    semantic_vectors = torch.from_numpy(
        np.load(Path(__file__).resolve().parent / 'data' / 'UESTC-MMEA-CL_BERT.npy')
    ).float().to(device)

    model = SemanticFusionVGG16(n_classes=32, fusion_mode=args.fusion_mode).to(device)
    parameter_count = sum(p.numel() for p in model.parameters())
    logging.info('Fusion mode: %s', args.fusion_mode)
    logging.info('Teacher parameters: %.3f M', parameter_count / 1e6)
    logging.info('Aligned train/val/test samples: %d/%d/%d',
                 actual_counts['train'], actual_counts['val'], actual_counts['test'])
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    criterion_cls = nn.CrossEntropyLoss().to(device)
    criterion_semantic = nn.MSELoss().to(device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[40], gamma=0.5)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=1024)

    def train_epoch(epoch):
        model.train()
        total_loss = 0.0
        seen = 0
        correct_gyro = 0
        correct_acc = 0
        epoch_start = time.perf_counter()
        window_start = epoch_start
        window_seen = 0
        for batch_index, (gyro, acc, labels) in enumerate(train_loader, start=1):
            gyro = gyro.to(device, non_blocking=True)
            acc = acc.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            target_semantics = semantic_vectors[labels]
            with torch.cuda.amp.autocast(enabled=use_amp):
                outputs = model(gyro, acc, True)
                gyro_semantic, acc_semantic = outputs[12], outputs[13]
                gyro_logits, acc_logits = outputs[14], outputs[15]
                cls_loss = 0.5 * (criterion_cls(gyro_logits.float(), labels) +
                                  criterion_cls(acc_logits.float(), labels))
                semantic_loss = 0.5 * (
                    criterion_semantic(gyro_semantic.float(), target_semantics) +
                    criterion_semantic(acc_semantic.float(), target_semantics))
                loss = cls_loss + semantic_loss
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_size = labels.size(0)
            total_loss += loss.item() * batch_size
            seen += batch_size
            window_seen += batch_size
            correct_gyro += (gyro_logits.argmax(1) == labels).sum().item()
            correct_acc += (acc_logits.argmax(1) == labels).sum().item()
            if batch_index % args.print_freq == 0 or batch_index == len(train_loader):
                torch.cuda.synchronize()
                now = time.perf_counter()
                window_seconds = now - window_start
                average_seconds = now - epoch_start
                batches_left = len(train_loader) - batch_index
                eta_minutes = (average_seconds / batch_index) * batches_left / 60.0
                logging.info(
                    '[Teacher][Epoch %d][%d/%d] %.1f samples/s ETA=%.1f min',
                    epoch, batch_index, len(train_loader),
                    window_seen / max(window_seconds, 1e-12), eta_minutes)
                window_start = now
                window_seen = 0
            if args.smoke:
                break
        logging.info('[Teacher][Epoch %d] loss=%.5f gyro_acc=%.4f acc_acc=%.4f',
                     epoch, total_loss / max(seen, 1),
                     100 * correct_gyro / max(seen, 1),
                     100 * correct_acc / max(seen, 1))
        logging.info('[Teacher][Epoch %d] elapsed=%.1f min average=%.1f samples/s',
                     epoch, (time.perf_counter() - epoch_start) / 60.0,
                     seen / max(time.perf_counter() - epoch_start, 1e-12))

    @torch.no_grad()
    def evaluate(loader, split, epoch, one_batch=False):
        model.eval()
        seen = 0
        gyro_correct = 0
        acc_correct = 0
        combined_correct = 0
        for gyro, acc, labels in loader:
            gyro = gyro.to(device, non_blocking=True)
            acc = acc.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                gyro_logits, acc_logits = model(gyro, acc)
            seen += labels.size(0)
            gyro_correct += (gyro_logits.argmax(1) == labels).sum().item()
            acc_correct += (acc_logits.argmax(1) == labels).sum().item()
            combined_correct += ((gyro_logits + acc_logits).argmax(1) == labels).sum().item()
            if one_batch:
                break
        metrics = {
            'gyro_accuracy': 100 * gyro_correct / max(seen, 1),
            'accelerometer_accuracy': 100 * acc_correct / max(seen, 1),
            'combined_accuracy': 100 * combined_correct / max(seen, 1),
            'samples': seen,
        }
        logging.info('[Teacher][%s][Epoch %d] gyro=%.4f acc=%.4f combined=%.4f',
                     split, epoch, metrics['gyro_accuracy'],
                     metrics['accelerometer_accuracy'], metrics['combined_accuracy'])
        return metrics

    if args.smoke:
        train_epoch(1)
        evaluate(val_loader, 'validation-smoke', 1, one_batch=True)
        logging.info('Stage-2 teacher smoke test passed.')
        return

    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    best_metric = -1.0
    epochs_without_improvement = 0
    best_path = output_dir / 'best_teacher_acc.pth'
    last_epoch = 0

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        train_epoch(epoch)
        scheduler.step()
        val_metrics = evaluate(val_loader, 'validation', epoch)
        current = val_metrics['combined_accuracy']
        torch.save(state_dict_cpu(model), output_dir / 'last.pth')
        if current > best_metric:
            best_metric = current
            epochs_without_improvement = 0
            state = state_dict_cpu(model)
            torch.save(state, best_path)
            torch.save(state, output_dir / 'st_best_acc.pth')
        else:
            epochs_without_improvement += 1
        logging.info('Teacher early stopping: %d/%d epochs without improvement',
                     epochs_without_improvement, args.early_stop_patience)
        if epochs_without_improvement >= args.early_stop_patience:
            break

    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.load_state_dict(torch.load(best_path, map_location='cpu'))
    val_metrics = evaluate(val_loader, 'validation-best', last_epoch)
    test_metrics = evaluate(test_loader, 'test', last_epoch)
    metrics = {'best_validation': val_metrics, 'held_out_test': test_metrics}
    save_json(output_dir / 'metrics.json', metrics)
    save_json(output_dir / 'runtime.json', {
        'training_seconds': time.perf_counter() - start_time,
        'peak_cuda_memory_mb': torch.cuda.max_memory_allocated() / (1024 ** 2),
        'epochs_completed': last_epoch,
    })
    with open(output_dir / 'result.txt', 'w', encoding='utf-8') as handle:
        handle.write('Best Validation Combined Acc: %.4f\n' % val_metrics['combined_accuracy'])
        handle.write('Held-out Test Gyro Acc: %.4f\n' % test_metrics['gyro_accuracy'])
        handle.write('Held-out Test Accelerometer Acc: %.4f\n' % test_metrics['accelerometer_accuracy'])
        handle.write('Held-out Test Combined Acc: %.4f\n' % test_metrics['combined_accuracy'])


if __name__ == '__main__':
    main()
