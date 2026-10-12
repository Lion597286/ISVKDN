





from pathlib import Path
import csv
from datetime import datetime
import hashlib
import json
import statistics
import subprocess
import sys
import time
from types import SimpleNamespace



PROJECT_ROOT = Path(__file__).resolve().parent
PREFERRED_PYTHON = Path(r'F:\DeepLearningSoftware\Anaconda\envs\SAKDN\python.exe')
GAF_ROOT = Path(r'F:\Dataset2\UESTC-MMEA_GAF')
VIDEO_ROOT = Path(r'G:\Dataset\Multi-modalActionRecognition\UESTC-MMEA-CL')
TEACHER_CHECKPOINT = (PROJECT_ROOT / 'output' /
    'MMEA_a_g_ISVKDN_SMMRF_epochs80_batch16_lr0.0001' / 'st_best_acc.pth')
STUDENT_CHECKPOINT = (PROJECT_ROOT / 'output' / 'stage2' / 'students' /
    'A3_full_isvkdn' / 'seed_2025' / 'best_video_acc.pth')
BATCH_SIZE = 16
SEED = 42
USE_AMP = True
SGSM_K = 10
WARMUP_STEPS = 10
MEASURED_STEPS = 30  
ROUNDS = 3           
CACHED_BATCHES = 2
CPU_THREADS = 4
GPU_INDEX = 0
TIMEOUT_SECONDS = 1800
OUTPUT_ROOT = PROJECT_ROOT / 'output' / 'graph_step_benchmark'



def timing_statistics(values):
    ordered = sorted(values)
    return dict(mean_seconds=statistics.mean(values),
                median_seconds=statistics.median(values),
                std_seconds=statistics.stdev(values) if len(values) > 1 else 0.0,
                min_seconds=min(values), max_seconds=max(values),
                p95_seconds=ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))],
                measured_steps=len(values))


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    import numpy as np
    import torch
    import torchvision.transforms as tv
    from torch.utils.data import DataLoader, Subset
    import model.backbone as backbone
    from metric.loss import SoftTarget
    from stage2_train_student import load_state_dict_flexible, set_seed, student_training_step
    from transforms import GroupNormalize, Stack, ToTorchFormatTensor
    from uestc_paired_dataset import UESTCPairedDataset

    if min(BATCH_SIZE, CACHED_BATCHES, MEASURED_STEPS, ROUNDS, CPU_THREADS) < 1 or WARMUP_STEPS < 1:
        raise ValueError('批次大小、预热、计时步数、轮数和线程数必须为正数')
    if not torch.cuda.is_available():
        raise RuntimeError('训练步基准测试需要CUDA GPU；请使用SAKDN环境运行')
    for path in (GAF_ROOT, VIDEO_ROOT, TEACHER_CHECKPOINT, STUDENT_CHECKPOINT,
                 PROJECT_ROOT / 'data' / 'UESTC-MMEA-CL_BERT.npy'):
        if not path.exists():
            raise FileNotFoundError(path)
    torch.set_num_threads(CPU_THREADS)
    torch.cuda.set_device(GPU_INDEX)
    device = torch.device('cuda', GPU_INDEX)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    set_seed(SEED)
    stamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir = OUTPUT_ROOT / stamp
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    config = dict(dataset='UESTC-MMEA-CL', batch=BATCH_SIZE, seed=SEED,
                  frames=8, resolution=[224, 224], amp=USE_AMP, sgsm_k=SGSM_K,
                  warmup_steps_per_block=WARMUP_STEPS, measured_steps_per_block=MEASURED_STEPS,
                  rounds=ROUNDS, cached_batches=CACHED_BATCHES,
                  teacher_checkpoint=str(TEACHER_CHECKPOINT),
                  student_checkpoint=str(STUDENT_CHECKPOINT),
                  gpu=torch.cuda.get_device_name(device), torch_version=torch.__version__,
                  cuda_version=torch.version.cuda, cudnn_version=torch.backends.cudnn.version(),
                  gpu_index=GPU_INDEX, python=sys.version, cpu_threads=CPU_THREADS,
                  cudnn_benchmark=True, tf32=True,
                  scope='Current local reconstruction; score-weighted class embeddings',
                  timing_scope='Synchronized wall time: teacher/student forwards, losses, student backward, SGD and AMP update; GPU-resident real batches; excludes loading, decoding and H2D transfer',
                  control='Same semantics, masked inputs and loss weights; only graph construction and propagation disabled in no_graph',
                  source_sha256={name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
                                 for name in ('stage2_train_student.py', 'uestc_semantic_mapping.py',
                                              'benchmark_UESTC_graph_step.py')})
    write_json(output_dir / 'config.json', config)
    print('输出目录:', output_dir, flush=True)
    print('GPU:', config['gpu'], '| batch:', BATCH_SIZE, '| AMP:', USE_AMP, flush=True)
    print('正在加载检查点和真实训练批次（此阶段不计时）...', flush=True)
    student = backbone.TRN(32, 8, 'RGB', base_model='BNInception',
                           consensus_type='TRNmultiscale', dropout=0.8,
                           img_feature_dim=768, partial_bn=True, print_spec=False).to(device)
    load_state_dict_flexible(student, STUDENT_CHECKPOINT)
    initial_state = {key: value.detach().cpu().clone() for key, value in student.state_dict().items()}
    teacher = backbone.SemanticFusionVGG16(n_classes=32, fusion_mode='full').to(device)
    load_state_dict_flexible(teacher, TEACHER_CHECKPOINT)
    teacher.requires_grad_(False)
    teacher.eval()
    vectors = torch.from_numpy(np.load(PROJECT_ROOT / 'data' / 'UESTC-MMEA-CL_BERT.npy')).float().to(device)
    video_transform = tv.Compose([student.get_augmentation(), Stack(roll=True),
        ToTorchFormatTensor(div=False), GroupNormalize(student.input_mean, student.input_std)])
    dataset = UESTCPairedDataset('train', tv.Compose([tv.Resize((224, 224)), tv.ToTensor()]),
                                 video_transform, num_segments=8, gaf_root=GAF_ROOT,
                                 video_root=VIDEO_ROOT, return_id=True)
    if len(dataset) != 4553:
        raise ValueError('官方训练集应为4553条，实际为%s' % len(dataset))
    if CACHED_BATCHES * BATCH_SIZE > len(dataset):
        raise ValueError('缓存批次数超过训练集大小')
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(SEED))[
        :CACHED_BATCHES * BATCH_SIZE].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=BATCH_SIZE, num_workers=0,
                        pin_memory=True, drop_last=True)
    batches = []
    sample_ids = []
    for gyro, acc, videos, labels, ids in loader:
        batches.append(tuple(value.to(device, non_blocking=True) for value in (gyro, acc, videos, labels)))
        sample_ids.append(list(ids))
    torch.cuda.synchronize(device)
    config['cached_sample_ids'] = sample_ids
    config['student_parameters'] = sum(parameter.numel() for parameter in student.parameters())
    write_json(output_dir / 'config.json', config)
    criteria = (torch.nn.CrossEntropyLoss().to(device), SoftTarget(4.0).to(device),
                torch.nn.MSELoss().to(device))
    rows = []
    blocks = []
    fields = ['round', 'mode', 'step', 'batch_index', 'wall_seconds', 'cuda_interval_seconds',
              'loss', 'ce', 'soft_kd', 'feature', 'semantic', 'scale_before', 'scale_after']
    try:
        with (output_dir / 'steps.csv').open('w', newline='', encoding='utf-8') as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for round_index in range(ROUNDS):
                order = ('no_graph', 'full') if round_index % 2 == 0 else ('full', 'no_graph')
                for mode in order:
                    student.load_state_dict(initial_state, strict=True)
                    student.zero_grad(set_to_none=True)
                    student.train()
                    teacher.eval()
                    set_seed(SEED + round_index)
                    optimizer = torch.optim.SGD(student.get_optim_policies(), lr=1e-3,
                                                momentum=0.9, weight_decay=5e-4)
                    scaler = torch.cuda.amp.GradScaler(enabled=USE_AMP, init_scale=1024)
                    args = SimpleNamespace(sgsm_mode=mode, sgsm_k=SGSM_K,
                                           soft_kd_weight=1.0, feature_weight=1.0, semantic_weight=1.0)
                    print('第%d/%d轮 %s: 预热%d步' % (round_index + 1, ROUNDS, mode, WARMUP_STEPS), flush=True)
                    for index in range(WARMUP_STEPS):
                        if time.perf_counter() - started > TIMEOUT_SECONDS:
                            raise TimeoutError('基准测试超时')
                        _, losses = student_training_step(student, teacher, batches[index % len(batches)],
                            vectors, args, optimizer, scaler, USE_AMP, criteria)
                        if not torch.isfinite(losses['loss']).item():
                            raise FloatingPointError('预热损失非有限值')
                    torch.cuda.synchronize(device)
                    
                    set_seed(SEED + round_index + 10000)
                    torch.cuda.reset_peak_memory_stats(device)
                    block = []
                    for index in range(MEASURED_STEPS):
                        if time.perf_counter() - started > TIMEOUT_SECONDS:
                            raise TimeoutError('基准测试超时')
                        scale_before = scaler.get_scale()
                        start_event = torch.cuda.Event(enable_timing=True)
                        end_event = torch.cuda.Event(enable_timing=True)
                        torch.cuda.synchronize(device)
                        tick = time.perf_counter()
                        start_event.record()
                        logits, losses = student_training_step(student, teacher, batches[index % len(batches)],
                            vectors, args, optimizer, scaler, USE_AMP, criteria)
                        end_event.record()
                        torch.cuda.synchronize(device)
                        elapsed = time.perf_counter() - tick
                        scale_after = scaler.get_scale()
                        if not torch.isfinite(losses['loss']).item() or scale_after < scale_before:
                            raise FloatingPointError('计时步出现非有限损失或AMP跳过更新；不将其作为有效训练步')
                        row = dict(round=round_index + 1, mode=mode, step=index + 1,
                                   batch_index=index % len(batches), wall_seconds=elapsed,
                                   cuda_interval_seconds=start_event.elapsed_time(end_event) / 1000,
                                   scale_before=scale_before, scale_after=scale_after,
                                   **{name: float(value.detach()) for name, value in losses.items()})
                        rows.append(row)
                        block.append(elapsed)
                        writer.writerow(row)
                        if (index + 1) % 10 == 0 or index + 1 == MEASURED_STEPS:
                            handle.flush()
                            print('  已测%d/%d步，均值%.5f秒/步' %
                                  (index + 1, MEASURED_STEPS, statistics.mean(block)), flush=True)
                        del logits, losses
                    blocks.append(dict(round=round_index + 1, mode=mode,
                                       peak_allocated_mb=torch.cuda.max_memory_allocated(device) / 1024 ** 2,
                                       **timing_statistics(block)))
                    del optimizer, scaler
        modes = {mode: timing_statistics([row['wall_seconds'] for row in rows if row['mode'] == mode])
                 for mode in ('no_graph', 'full')}
        deltas = [next(block['mean_seconds'] for block in blocks if block['round'] == index and block['mode'] == 'full') -
                  next(block['mean_seconds'] for block in blocks if block['round'] == index and block['mode'] == 'no_graph')
                  for index in range(1, ROUNDS + 1)]
        report = dict(status='completed', config=config, modes=modes, blocks=blocks,
                      paired_round_deltas_seconds=deltas,
                      mean_graph_overhead_seconds=statistics.mean(deltas),
                      mean_graph_overhead_percent=100 * statistics.mean(deltas) / modes['no_graph']['mean_seconds'],
                      elapsed_total_seconds=time.perf_counter() - started,
                      checkpoint_files_modified=False,
                      note='Timing experiment only; no accuracy evaluation. Does not validate remote/original experiments.')
        write_json(output_dir / 'summary.json', report)
        lines = ['# UESTC-MMEA-CL 图传播训练步计时', '',
                 '当前本地重建实现；真实训练数据提前驻留GPU。教师/学生前向、损失、反向、SGD和AMP更新均计时；数据加载、解码及传输不计时。', '',
                 '| 配置 | 均值（秒/步） | 标准差 | 中位数 | 计时步数 |',
                 '| --- | ---: | ---: | ---: | ---: |']
        for mode, label in (('no_graph', '正确语义，无图构建/传播'), ('full', '正确语义，有图构建/传播')):
            value = modes[mode]
            lines.append('| %s | %.5f | %.5f | %.5f | %d |' % (label, value['mean_seconds'],
                value['std_seconds'], value['median_seconds'], value['measured_steps']))
        lines.extend(['', '有图减无图：%.5f 秒/步（%.2f%%）。差值保留符号，不人为调整结果。' %
                      (report['mean_graph_overhead_seconds'], report['mean_graph_overhead_percent']), '',
                      'GPU：%s；batch=%d；8帧224×224；AMP=%s；k=%d；计时种子=%d。' %
                      (config['gpu'], BATCH_SIZE, USE_AMP, SGSM_K, SEED), '',
                      '每轮恢复同一学生检查点和随机状态，交替测量先后顺序。该实现保留现有概率加权BERT语义；原论文按预测类别取嵌入，二者须分别说明。', '',
                      '原始逐步记录：steps.csv；环境、采样ID、源码哈希和各轮统计：summary.json。'])
        (output_dir / '结果说明.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        write_json(OUTPUT_ROOT / 'latest.json', dict(output_dir=str(output_dir), summary=str(output_dir / 'summary.json')))
        print('\n无图: %.5f s/step; 有图: %.5f s/step; 差值: %+.5f s' %
              (modes['no_graph']['mean_seconds'], modes['full']['mean_seconds'], statistics.mean(deltas)), flush=True)
        print('结果:', output_dir / '结果说明.md', flush=True)
    except Exception as error:
        write_json(output_dir / 'failure.json', dict(status='failed', error=repr(error),
                   measured_steps=len(rows), elapsed_seconds=time.perf_counter() - started))
        raise


if __name__ == '__main__':
    if PREFERRED_PYTHON.is_file() and Path(sys.executable).resolve() != PREFERRED_PYTHON.resolve():
        subprocess.run([str(PREFERRED_PYTHON), '-X', 'utf8', '-u', str(Path(__file__).resolve())],
                       cwd=str(PROJECT_ROOT), check=True)
    else:
        main()
