




from pathlib import Path
import csv
from datetime import datetime
import hashlib
import json
import statistics
import subprocess
import sys
import time



PROJECT_ROOT = Path(__file__).resolve().parent
PREFERRED_PYTHON = Path(r'F:\DeepLearningSoftware\Anaconda\envs\SAKDN\python.exe')
CHECKPOINT = (PROJECT_ROOT / 'output' / 'stage2' / 'students' /
              'A3_full_isvkdn' / 'seed_2025' / 'best_video_acc.pth')
GAF_ROOT = Path(r'F:\Dataset2\UESTC-MMEA_GAF')
VIDEO_ROOT = Path(r'G:\Dataset\Multi-modalActionRecognition\UESTC-MMEA-CL')
SEED = 42
CPU_THREADS = 4
GPU_INDEX = 0
CACHED_VIDEOS = 4
WARMUP_STEPS = 30
MEASURED_STEPS = 100
ROUNDS = 3
PRECISIONS = ('amp', 'fp32_tf32')
TIMEOUT_SECONDS = 900
OUTPUT_ROOT = PROJECT_ROOT / 'output' / 'student_forward_benchmark'



def gpu_snapshot():
    try:
        result = subprocess.run(['nvidia-smi', '--query-gpu=index,name,utilization.gpu,memory.used',
                                 '--format=csv,noheader'], capture_output=True, timeout=10)
        return result.stdout.decode('utf-8', errors='replace').strip()
    except (OSError, subprocess.TimeoutExpired) as error:
        return repr(error)


def summarize(values):
    return dict(mean_ms=statistics.mean(values), median_ms=statistics.median(values),
                std_ms=statistics.stdev(values), min_ms=min(values), max_ms=max(values),
                count=len(values))


def main():
    import numpy as np
    import torch
    import torchvision.transforms as tv
    from torch.utils.data import DataLoader, Subset
    from model.backbone.TRN import TRN
    from stage2_train_student import load_state_dict_flexible, set_seed
    from transforms import GroupCenterCrop, GroupNormalize, GroupScale, Stack, ToTorchFormatTensor
    from uestc_paired_dataset import UESTCPairedDataset

    if min(CPU_THREADS, CACHED_VIDEOS, WARMUP_STEPS, MEASURED_STEPS, ROUNDS) < 1 or MEASURED_STEPS < 2:
        raise ValueError('Invalid benchmark dimensions')
    if not PRECISIONS or set(PRECISIONS) - {'amp', 'fp32_tf32'}:
        raise ValueError('PRECISIONS must contain amp and/or fp32_tf32')
    if not CHECKPOINT.is_file() or not torch.cuda.is_available():
        raise RuntimeError('需要已有学生检查点和CUDA GPU')
    torch.set_num_threads(CPU_THREADS)
    torch.cuda.set_device(GPU_INDEX)
    device = torch.device('cuda', GPU_INDEX)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cuda.matmul.allow_tf32 = True
    set_seed(SEED)
    output_dir = OUTPUT_ROOT / datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    output_dir.mkdir(parents=True, exist_ok=False)
    started = time.perf_counter()
    print('输出目录:', output_dir, flush=True)
    snapshot_before = gpu_snapshot()
    print('测试前GPU状态:', snapshot_before, flush=True)
    student = TRN(32, 8, 'RGB', base_model='BNInception', consensus_type='TRNmultiscale',
                  dropout=0.8, img_feature_dim=768, partial_bn=True, print_spec=False).to(device).eval()
    load_state_dict_flexible(student, CHECKPOINT)
    parameters = sum(value.numel() for value in student.parameters())
    per_class_parameters = sum(module[-1].in_features + int(module[-1].bias is not None)
                               for module in student.consensus.fc_fusion_scales)
    param_by_classes = {name: parameters + (classes - 32) * per_class_parameters
                        for name, classes in [('UTD-MHAD', 27), ('Berkeley-MHAD', 11), ('UESTC-MMEA-CL', 32)]}
    video_transform = tv.Compose([GroupScale(int(student.scale_size)),
        GroupCenterCrop(student.crop_size), Stack(roll=True), ToTorchFormatTensor(div=False),
        GroupNormalize(student.input_mean, student.input_std)])
    dataset = UESTCPairedDataset('val', tv.Compose([tv.Resize((224, 224)), tv.ToTensor()]),
        video_transform, num_segments=8, gaf_root=GAF_ROOT, video_root=VIDEO_ROOT,
        video_only=True, return_id=True)
    if len(dataset) != 653 or CACHED_VIDEOS > len(dataset):
        raise ValueError('验证集数量或缓存视频数量不匹配')
    indices = torch.randperm(len(dataset), generator=torch.Generator().manual_seed(SEED))[:CACHED_VIDEOS]
    loader = DataLoader(Subset(dataset, indices.tolist()), batch_size=1, num_workers=0, pin_memory=True)
    inputs, sample_ids = [], []
    for videos, labels, ids in loader:
        inputs.append(videos.to(device, non_blocking=True))
        sample_ids.extend(ids)
    torch.cuda.synchronize(device)
    config = dict(scope='Current local reconstructed student; not original remote experiment',
                  checkpoint=str(CHECKPOINT), gpu=torch.cuda.get_device_name(device),
                  torch=torch.__version__, cuda=torch.version.cuda, cudnn=torch.backends.cudnn.version(),
                  python=sys.version, cpu_threads=CPU_THREADS, batch=1, frames=8,
                  resolution=[224, 224], seed=SEED, warmup_per_block=WARMUP_STEPS,
                  measured_per_block=MEASURED_STEPS, rounds=ROUNDS, precisions=list(PRECISIONS),
                  cudnn_benchmark=True, tf32=True, return_features=False,
                  mode='eval with torch.no_grad; per-forward autocast context as in current evaluation',
                  input_shape=list(inputs[0].shape),
                  sample_ids=sample_ids, gpu_status_before=snapshot_before,
                  timing_scope='Synchronized wall-clock forward time; excludes decoding, transforms, H2D, softmax and CPU output transfer',
                  checkpoint_size=CHECKPOINT.stat().st_size,
                  code_sha256={name: hashlib.sha256((PROJECT_ROOT / name).read_bytes()).hexdigest()
                               for name in ('benchmark_UESTC_student_forward.py', 'model/backbone/TRN.py')})
    (output_dir / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding='utf-8')
    rows, blocks = [], []
    fields = ['round', 'precision', 'step', 'input_index', 'wall_ms', 'cuda_interval_ms']
    try:
        with (output_dir / 'steps.csv').open('w', newline='', encoding='utf-8') as handle, torch.no_grad():
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for round_index in range(ROUNDS):
                order = PRECISIONS if round_index % 2 == 0 else tuple(reversed(PRECISIONS))
                for precision in order:
                    before = gpu_snapshot()
                    use_amp = precision == 'amp'
                    print('第%d/%d轮 %s: 预热%d次' % (round_index + 1, ROUNDS, precision, WARMUP_STEPS), flush=True)
                    for index in range(WARMUP_STEPS):
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            output = student(inputs[index % len(inputs)], return_features=False)
                    torch.cuda.synchronize(device)
                    if output.shape != (1, 32) or not torch.isfinite(output).all().item():
                        raise FloatingPointError('学生输出形状错误或出现非有限值')
                    begin_event = torch.cuda.Event(enable_timing=True)
                    end_event = torch.cuda.Event(enable_timing=True)
                    block_rows = []
                    for index in range(MEASURED_STEPS):
                        if time.perf_counter() - started > TIMEOUT_SECONDS:
                            raise TimeoutError('前向计时超时')
                        torch.cuda.synchronize(device)
                        tick = time.perf_counter()
                        begin_event.record()
                        with torch.cuda.amp.autocast(enabled=use_amp):
                            output = student(inputs[index % len(inputs)], return_features=False)
                        end_event.record()
                        torch.cuda.synchronize(device)
                        elapsed_ms = (time.perf_counter() - tick) * 1000
                        row = dict(round=round_index + 1, precision=precision, step=index + 1,
                                   input_index=index % len(inputs), wall_ms=elapsed_ms,
                                   cuda_interval_ms=begin_event.elapsed_time(end_event))
                        writer.writerow(row)
                        rows.append(row)
                        block_rows.append(row)
                    handle.flush()
                    if not torch.isfinite(output).all().item():
                        raise FloatingPointError('计时后学生输出出现非有限值')
                    block = dict(round=round_index + 1, precision=precision,
                                 wall=summarize([value['wall_ms'] for value in block_rows]),
                                 cuda_interval=summarize([value['cuda_interval_ms'] for value in block_rows]),
                                 gpu_status_before=before, gpu_status_after=gpu_snapshot())
                    blocks.append(block)
                    print('  均值%.4f ms，中位数%.4f ms' %
                          (block['wall']['mean_ms'], block['wall']['median_ms']), flush=True)
        results = {precision: dict(wall=summarize([row['wall_ms'] for row in rows if row['precision'] == precision]),
                                  cuda_interval=summarize([row['cuda_interval_ms'] for row in rows if row['precision'] == precision]))
                   for precision in PRECISIONS}
        report = dict(status='completed', config=config, parameters=parameters,
                      parameters_millions=parameters / 1e6,
                      parameter_counts_for_same_architecture_class_heads=param_by_classes,
                      class_head_parameters_per_class=per_class_parameters,
                      results=results, blocks=blocks, gpu_status_after=gpu_snapshot(),
                      elapsed_seconds=time.perf_counter() - started,
                      note='Other dataset counts change class heads only; not measured latencies or verification of their original architectures.')
        (output_dir / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
        lines = ['# UESTC-MMEA-CL 视频学生前向实测', '',
                 'RTX 4090 D；batch=1；8帧224×224；真实验证视频预处理后驻留GPU；eval及no_grad；每次前向独立进入autocast，与当前评估代码相同；仅学生分类前向。', '',
                 '| 精度 | 墙钟均值（ms） | 中位数（ms） | 标准差（ms） | CUDA区间均值（ms） | 次数 |',
                 '| --- | ---: | ---: | ---: | ---: | ---: |']
        for precision in PRECISIONS:
            wall = results[precision]['wall']
            lines.append('| %s | %.4f | %.4f | %.4f | %.4f | %d |' %
                (precision, wall['mean_ms'], wall['median_ms'], wall['std_ms'],
                 results[precision]['cuda_interval']['mean_ms'], wall['count']))
        lines.extend(['', '精确参数量：%s；即%.6fM；保留两位小数为%.2fM。' %
                      (format(parameters, ','), parameters / 1e6, parameters / 1e6), '',
                      'amp使用自动混合精度；fp32_tf32使用FP32张量并允许TF32。此处以GPU同步的墙钟时间为主要前向延迟，CUDA事件区间为参照。', '',
                      '不包含视频解码、抽帧、预处理、CPU到GPU传输或输出转CPU。每组每轮预热30次，计时100次，共3轮；硬件其他任务的状态记录于summary.json。', '',
                      '这是当前本地模型与当前运行环境的测量，不据此认定另一实现或另一环境的18.57ms为相同结果。未修改论文或回复信。'])
        (output_dir / '结果说明.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
        (OUTPUT_ROOT / 'latest.json').write_text(json.dumps(dict(output_dir=str(output_dir)), indent=2), encoding='utf-8')
        print('参数量:', parameters, '|', parameters / 1e6, 'M', flush=True)
        print('结果:', output_dir / '结果说明.md', flush=True)
    except Exception as error:
        (output_dir / 'failure.json').write_text(json.dumps(dict(error=repr(error), measured_forwards=len(rows))), encoding='utf-8')
        raise


if __name__ == '__main__':
    if PREFERRED_PYTHON.is_file() and Path(sys.executable).resolve() != PREFERRED_PYTHON.resolve():
        subprocess.run([str(PREFERRED_PYTHON), '-X', 'utf8', '-u', str(Path(__file__).resolve())],
                       cwd=str(PROJECT_ROOT), check=True)
    else:
        main()
