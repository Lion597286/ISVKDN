

from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path
import random
import sys
import time

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as tv_transforms
from torch.utils.data import DataLoader

import model.backbone as backbone
from metric.loss import SoftTarget
from transforms import GroupCenterCrop, GroupNormalize, GroupScale, Stack, ToTorchFormatTensor
from uestc_paired_dataset import UESTCPairedDataset
from uestc_semantic_mapping import (
    feature_matching_loss,
    raw_attention_map,
    sgsm_map,
    sgsm_semantic_slope,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gaf_root', default=r'F:\Dataset2\UESTC-MMEA_GAF')
    parser.add_argument('--video_root', default=r'G:\Dataset\Multi-modalActionRecognition\UESTC-MMEA-CL')
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--experiment_id', required=True)
    parser.add_argument('--teacher_load', default='')
    parser.add_argument('--fusion_mode', choices=['mean', 'concat', 'concat_gate', 'full'], default='full')
    parser.add_argument('--sgsm_mode', choices=['off', 'raw', 'shuffled', 'no_graph', 'full'], default='full')
    parser.add_argument('--soft_kd_weight', type=float, default=1.0)
    parser.add_argument('--feature_weight', type=float, default=1.0)
    parser.add_argument('--semantic_weight', type=float, default=1.0)
    parser.add_argument('--temperature', type=float, default=4.0)
    parser.add_argument('--sgsm_k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=2025)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=70)
    parser.add_argument('--lr', type=float, default=1e-3)
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
    cv2.setNumThreads(1)


def save_json(path, value):
    with open(path, 'w', encoding='utf-8') as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2)


def state_dict_cpu(model):
    source = model.module if isinstance(model, nn.DataParallel) else model
    return {key: value.detach().cpu() for key, value in source.state_dict().items()}


def load_state_dict_flexible(model, path):
    state = torch.load(path, map_location='cpu')
    if not isinstance(state, dict):
        raise TypeError('Checkpoint must contain a state dictionary: %s' % path)
    if state and all(key.startswith('module.') for key in state):
        state = {key[len('module.'):]: value for key, value in state.items()}
    model.load_state_dict(state)


def classification_metrics(labels, predictions, num_classes=32):
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    for true_label, prediction in zip(labels, predictions):
        confusion[int(true_label), int(prediction)] += 1
    true_counts = confusion.sum(axis=1)
    pred_counts = confusion.sum(axis=0)
    tp = np.diag(confusion).astype(np.float64)
    precision = np.divide(tp, pred_counts, out=np.zeros_like(tp), where=pred_counts > 0)
    recall = np.divide(tp, true_counts, out=np.zeros_like(tp), where=true_counts > 0)
    f1 = np.divide(2 * precision * recall, precision + recall,
                   out=np.zeros_like(tp), where=(precision + recall) > 0)
    return {
        'accuracy': float(100.0 * tp.sum() / max(confusion.sum(), 1)),
        'macro_precision': float(100.0 * precision.mean()),
        'macro_recall': float(100.0 * recall.mean()),
        'macro_f1': float(100.0 * f1.mean()),
        'samples': int(confusion.sum()),
        'per_class_recall': [float(100.0 * value) for value in recall],
        'confusion_matrix': confusion.tolist(),
    }


def save_predictions(output_dir, split, sample_ids, labels, predictions, logits, metrics):
    with open(output_dir / f'{split}_predictions.csv', 'w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['sample_id', 'true_label', 'predicted_label'])
        writer.writerows(zip(sample_ids, labels.tolist(), predictions.tolist()))
    np.save(output_dir / f'{split}_logits.npy', logits.astype(np.float32))
    save_json(output_dir / f'{split}_metrics.json', metrics)


def student_training_step(student, teacher, batch, bert_vectors, args,
                          optimizer, scaler, use_amp, criteria):
    





    ce_criterion, kd_criterion, semantic_criterion = criteria
    use_semantic_mapping = args.feature_weight > 0 and args.sgsm_mode in {
        'full', 'shuffled', 'no_graph'}
    if teacher is None:
        videos, labels = batch
        with torch.cuda.amp.autocast(enabled=use_amp):
            video_logits = student(videos, return_features=False)
            ce_loss = ce_criterion(video_logits.float(), labels)
            soft_kd_loss = video_logits.new_zeros(())
            feature_loss = video_logits.new_zeros(())
            semantic_loss = video_logits.new_zeros(())
            loss = ce_loss
    else:
        gyro, acc, videos, labels = batch
        batch_size = labels.size(0)
        if use_semantic_mapping:
            gyro_input = torch.cat((gyro, torch.zeros_like(gyro)), dim=0)
            acc_input = torch.cat((acc, torch.zeros_like(acc)), dim=0)
            video_input = torch.cat((videos, torch.zeros_like(videos)), dim=0)
        else:
            gyro_input, acc_input, video_input = gyro, acc, videos

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            teacher_outputs = teacher(gyro_input, acc_input, True)
        with torch.cuda.amp.autocast(enabled=use_amp):
            student_outputs = student(video_input, return_features='distill')

        gyro_features = [teacher_outputs[i] for i in (0, 2, 4, 6, 8)]
        acc_features = [teacher_outputs[i] for i in (1, 3, 5, 7, 9)]
        gyro_semantic, acc_semantic = teacher_outputs[12], teacher_outputs[13]
        gyro_logits, acc_logits = teacher_outputs[14], teacher_outputs[15]
        student_features = list(student_outputs[:5])
        video_semantic, video_logits = student_outputs[5], student_outputs[6]

        if use_semantic_mapping:
            propagate_graph = args.sgsm_mode != 'no_graph'
            gyro_slope = sgsm_semantic_slope(
                gyro_logits, bert_vectors, args.sgsm_k, propagate_graph)
            acc_slope = sgsm_semantic_slope(
                acc_logits, bert_vectors, args.sgsm_k, propagate_graph)
            video_slope = sgsm_semantic_slope(
                video_logits, bert_vectors, args.sgsm_k, propagate_graph)

        gyro_features = [value[:batch_size] for value in gyro_features]
        acc_features = [value[:batch_size] for value in acc_features]
        student_features = [value[:batch_size] for value in student_features]
        gyro_semantic = gyro_semantic[:batch_size]
        acc_semantic = acc_semantic[:batch_size]
        gyro_logits = gyro_logits[:batch_size]
        acc_logits = acc_logits[:batch_size]
        video_semantic = video_semantic[:batch_size]
        video_logits = video_logits[:batch_size]

        with torch.cuda.amp.autocast(enabled=use_amp):
            ce_loss = ce_criterion(video_logits.float(), labels)
            soft_kd_loss = video_logits.new_zeros(())
            feature_loss = video_logits.new_zeros(())
            semantic_loss = video_logits.new_zeros(())
            if args.soft_kd_weight > 0:
                soft_kd_loss = 0.5 * args.soft_kd_weight * (
                    kd_criterion(video_logits.float(), gyro_logits.float()) +
                    kd_criterion(video_logits.float(), acc_logits.float()))
            if args.feature_weight > 0:
                mapped_pairs = []
                for video_feature, gyro_feature, acc_feature in zip(
                        student_features, gyro_features, acc_features):
                    spatial_size = tuple(min(v, g, a) for v, g, a in zip(
                        video_feature.shape[-2:], gyro_feature.shape[-2:],
                        acc_feature.shape[-2:]))
                    if args.sgsm_mode == 'raw':
                        mapped_pairs.append((
                            raw_attention_map(video_feature, spatial_size),
                            raw_attention_map(gyro_feature, spatial_size),
                            raw_attention_map(acc_feature, spatial_size)))
                    else:
                        mapped_pairs.append((
                            sgsm_map(video_feature, video_slope, spatial_size),
                            sgsm_map(gyro_feature, gyro_slope, spatial_size),
                            sgsm_map(acc_feature, acc_slope, spatial_size)))
                feature_loss = args.feature_weight * feature_matching_loss(mapped_pairs)
            if args.semantic_weight > 0:
                semantic_loss = 0.5 * args.semantic_weight * (
                    semantic_criterion(video_semantic.float(), gyro_semantic.float()) +
                    semantic_criterion(video_semantic.float(), acc_semantic.float()))
            loss = ce_loss + soft_kd_loss + feature_loss + semantic_loss

    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()
    return video_logits, dict(loss=loss, ce=ce_loss, soft_kd=soft_kd_loss,
                              feature=feature_loss, semantic=semantic_loss)


def main():
    args = parse_args()
    if args.early_stop_patience < 1:
        raise ValueError('early_stop_patience must be at least 1')
    if args.prefetch_factor < 1 or args.print_freq < 1:
        raise ValueError('prefetch_factor and print_freq must be at least 1')
    if args.feature_weight > 0 and args.sgsm_mode == 'off':
        raise ValueError('feature_weight must be 0 when sgsm_mode=off')
    if args.feature_weight == 0 and args.sgsm_mode != 'off':
        raise ValueError('sgsm_mode must be off when feature_weight=0')
    if not torch.cuda.is_available():
        raise RuntimeError('Stage-2 training requires a CUDA GPU')

    teacher_required = any(weight > 0 for weight in (
        args.soft_kd_weight, args.feature_weight, args.semantic_weight))
    if teacher_required and not args.smoke and not Path(args.teacher_load).is_file():
        raise FileNotFoundError('Teacher checkpoint not found: %s' % args.teacher_load)

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
    config.update(
        dataset='UESTC-MMEA-CL',
        num_classes=32,
        num_segments=8,
        student='TRN/BN-Inception',
        split={'train': 4553, 'val': 653, 'test': 1316},
        selection_metric='video_student_validation_accuracy',
        inference_modality='RGB video only',
        teacher_required=teacher_required,
    )
    save_json(output_dir / 'config.json', config)

    student = backbone.TRN(
        32, 8, 'RGB', base_model='BNInception', consensus_type='TRNmultiscale',
        dropout=0.8, img_feature_dim=768, partial_bn=True).to(device)
    policies = student.get_optim_policies()

    teacher = None
    if teacher_required:
        teacher = backbone.SemanticFusionVGG16(
            n_classes=32, fusion_mode=args.fusion_mode).to(device)
        if args.teacher_load and Path(args.teacher_load).is_file():
            load_state_dict_flexible(teacher, args.teacher_load)
        teacher.requires_grad_(False)
        teacher.eval()

    logging.info('Experiment: %s', args.experiment_id)
    logging.info('Loss weights: CE=1 softKD=%g feature=%g semantic=%g',
                 args.soft_kd_weight, args.feature_weight, args.semantic_weight)
    logging.info('SMMRF=%s SGSM=%s seed=%d', args.fusion_mode, args.sgsm_mode, args.seed)
    logging.info('Student parameters: %.3f M',
                 sum(p.numel() for p in student.parameters()) / 1e6)

    crop_size = student.crop_size
    scale_size = student.scale_size
    normalize = GroupNormalize(student.input_mean, student.input_std)
    train_augmentation = student.get_augmentation()
    gaf_transform = tv_transforms.Compose([
        tv_transforms.Resize((224, 224)),
        tv_transforms.ToTensor(),
    ])
    train_video_transform = torchvision.transforms.Compose([
        train_augmentation,
        Stack(roll=True),
        ToTorchFormatTensor(div=False),
        normalize,
    ])
    eval_video_transform = torchvision.transforms.Compose([
        GroupScale(int(scale_size)),
        GroupCenterCrop(crop_size),
        Stack(roll=True),
        ToTorchFormatTensor(div=False),
        normalize,
    ])
    generator = torch.Generator().manual_seed(args.seed)

    def make_loader(split, transform, shuffle=False, workers=0, return_id=False,
                    video_only=False):
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
            UESTCPairedDataset(
                split, gaf_transform, transform, num_segments=8,
                gaf_root=args.gaf_root, video_root=args.video_root,
                return_id=return_id, video_only=video_only),
            **loader_options)

    train_loader = make_loader('train', train_video_transform, True, args.workers,
                               video_only=not teacher_required)
    val_loader = make_loader('val', eval_video_transform, False,
                             max(1, args.workers // 2) if args.workers else 0,
                             True, True)
    test_loader = make_loader('test', eval_video_transform, False, 0, True, True)
    actual_counts = {
        'train': len(train_loader.dataset),
        'val': len(val_loader.dataset),
        'test': len(test_loader.dataset),
    }
    if actual_counts != config['split']:
        raise ValueError('Official split count mismatch: %s' % actual_counts)
    logging.info('Aligned train/val/test samples: %d/%d/%d',
                 actual_counts['train'], actual_counts['val'], actual_counts['test'])

    bert_vectors = torch.from_numpy(
        np.load(Path(__file__).resolve().parent / 'data' / 'UESTC-MMEA-CL_BERT.npy')
    ).float().to(device)
    sgsm_vectors = bert_vectors
    if args.sgsm_mode == 'shuffled':
        shuffle_generator = torch.Generator().manual_seed(args.seed + 991)
        permutation = torch.randperm(bert_vectors.size(0), generator=shuffle_generator).to(device)
        sgsm_vectors = bert_vectors[permutation]
        config['semantic_permutation'] = permutation.cpu().tolist()
        save_json(output_dir / 'config.json', config)

    if torch.cuda.device_count() > 1:
        student = nn.DataParallel(student)
        if teacher is not None:
            teacher = nn.DataParallel(teacher)

    optimizer = torch.optim.SGD(
        policies, args.lr, momentum=0.9, weight_decay=5e-4)
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=1024)
    ce_criterion = nn.CrossEntropyLoss().to(device)
    kd_criterion = SoftTarget(args.temperature).to(device)
    semantic_criterion = nn.MSELoss().to(device)

    def train_epoch(epoch):
        student.train()
        if teacher is not None:
            teacher.eval()
        seen = 0
        correct = 0
        totals = {'loss': 0.0, 'ce': 0.0, 'soft_kd': 0.0, 'feature': 0.0, 'semantic': 0.0}
        epoch_start = time.perf_counter()
        window_start = epoch_start
        window_seen = 0

        for batch_index, batch in enumerate(train_loader, start=1):
            if teacher_required:
                gyro, acc, videos, labels = batch
                gyro = gyro.to(device, non_blocking=True)
                acc = acc.to(device, non_blocking=True)
            else:
                videos, labels = batch
            videos = videos.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            batch_size = labels.size(0)

            device_batch = ((gyro, acc, videos, labels) if teacher_required
                            else (videos, labels))
            video_logits, losses = student_training_step(
                student, teacher, device_batch, sgsm_vectors, args,
                optimizer, scaler, use_amp,
                (ce_criterion, kd_criterion, semantic_criterion))

            seen += batch_size
            window_seen += batch_size
            correct += (video_logits.argmax(1) == labels).sum().item()
            for name, value in losses.items():
                totals[name] += float(value.detach()) * batch_size
            if batch_index % args.print_freq == 0 or batch_index == len(train_loader):
                torch.cuda.synchronize()
                now = time.perf_counter()
                window_seconds = now - window_start
                average_seconds = now - epoch_start
                batches_left = len(train_loader) - batch_index
                eta_minutes = (average_seconds / batch_index) * batches_left / 60.0
                logging.info(
                    '[Student][Epoch %d][%d/%d] %.1f samples/s train_acc=%.2f%% ETA=%.1f min',
                    epoch, batch_index, len(train_loader),
                    window_seen / max(window_seconds, 1e-12),
                    100 * correct / max(seen, 1), eta_minutes)
                window_start = now
                window_seen = 0
            if args.smoke:
                break

        logging.info(
            '[Student][Epoch %d] loss=%.5f ce=%.5f softKD=%.5f feature=%.5f '
            'semantic=%.5f train_acc=%.4f',
            epoch, *(totals[name] / max(seen, 1) for name in
                     ('loss', 'ce', 'soft_kd', 'feature', 'semantic')),
            100 * correct / max(seen, 1))
        logging.info('[Student][Epoch %d] elapsed=%.1f min average=%.1f samples/s',
                     epoch, (time.perf_counter() - epoch_start) / 60.0,
                     seen / max(time.perf_counter() - epoch_start, 1e-12))

    @torch.no_grad()
    def evaluate(loader, split, epoch, save=False, one_batch=False):
        student.eval()
        logits_all = []
        labels_all = []
        sample_ids_all = []
        torch.cuda.synchronize()
        start = time.perf_counter()
        for videos, labels, sample_ids in loader:
            videos = videos.to(device, non_blocking=True)
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = student(videos, return_features=False)
            logits_all.append(logits.float().cpu())
            labels_all.append(labels.cpu())
            sample_ids_all.extend(list(sample_ids))
            if one_batch:
                break
        torch.cuda.synchronize()
        elapsed = time.perf_counter() - start
        logits = torch.cat(logits_all).numpy()
        labels = torch.cat(labels_all).numpy()
        predictions = logits.argmax(axis=1)
        metrics = classification_metrics(labels, predictions)
        metrics['end_to_end_seconds'] = elapsed
        metrics['end_to_end_samples_per_second'] = len(labels) / max(elapsed, 1e-12)
        logging.info('[Student][%s][Epoch %d] acc=%.4f macroF1=%.4f samples=%d',
                     split, epoch, metrics['accuracy'], metrics['macro_f1'], metrics['samples'])
        if save:
            save_predictions(output_dir, split, sample_ids_all, labels,
                             predictions, logits, metrics)
        return metrics

    if args.smoke:
        train_epoch(1)
        evaluate(val_loader, 'validation_smoke', 1, one_batch=True)
        logging.info('Stage-2 student smoke test passed: %s', args.experiment_id)
        return

    start_time = time.perf_counter()
    torch.cuda.reset_peak_memory_stats()
    best_accuracy = -1.0
    epochs_without_improvement = 0
    best_path = output_dir / 'best_video_acc.pth'
    last_epoch = 0

    for epoch in range(1, args.epochs + 1):
        last_epoch = epoch
        if epoch in {35}:
            for group in optimizer.param_groups:
                group['lr'] *= 0.5
        train_epoch(epoch)
        val_metrics = evaluate(val_loader, 'validation', epoch)
        torch.save(state_dict_cpu(student), output_dir / 'last.pth')
        if val_metrics['accuracy'] > best_accuracy:
            best_accuracy = val_metrics['accuracy']
            epochs_without_improvement = 0
            state = state_dict_cpu(student)
            torch.save(state, best_path)
            torch.save(state, output_dir / 'best_acc.pth')
        else:
            epochs_without_improvement += 1
        logging.info('Student early stopping: %d/%d epochs without video validation improvement',
                     epochs_without_improvement, args.early_stop_patience)
        if epochs_without_improvement >= args.early_stop_patience:
            break

    raw_student = student.module if isinstance(student, nn.DataParallel) else student
    raw_student.load_state_dict(torch.load(best_path, map_location='cpu'))
    validation_metrics = evaluate(val_loader, 'validation', last_epoch, save=True)
    test_metrics = evaluate(test_loader, 'test', last_epoch, save=True)
    save_json(output_dir / 'metrics.json', {
        'best_validation': validation_metrics,
        'held_out_test': test_metrics,
    })
    save_json(output_dir / 'runtime.json', {
        'training_seconds': time.perf_counter() - start_time,
        'peak_cuda_memory_mb': torch.cuda.max_memory_allocated() / (1024 ** 2),
        'epochs_completed': last_epoch,
    })
    with open(output_dir / 'result.txt', 'w', encoding='utf-8') as handle:
        handle.write('Best Validation Video Acc: %.4f\n' % validation_metrics['accuracy'])
        handle.write('Held-out Test Video Acc: %.4f\n' % test_metrics['accuracy'])
        handle.write('Held-out Test Macro Precision: %.4f\n' % test_metrics['macro_precision'])
        handle.write('Held-out Test Macro Recall: %.4f\n' % test_metrics['macro_recall'])
        handle.write('Held-out Test Macro F1: %.4f\n' % test_metrics['macro_f1'])


if __name__ == '__main__':
    main()
