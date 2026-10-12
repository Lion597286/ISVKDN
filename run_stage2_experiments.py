






from __future__ import annotations

from pathlib import Path
import subprocess
import sys



GAF_ROOT = Path(r'F:\Dataset2\UESTC-MMEA_GAF')
VIDEO_ROOT = Path(r'G:\Dataset\Multi-modalActionRecognition\UESTC-MMEA-CL')
PREFERRED_PYTHON = Path(r'F:\DeepLearningSoftware\Anaconda\envs\SAKDN\python.exe')
EXISTING_FULL_TEACHER = (
    Path(__file__).resolve().parent / 'output' /
    'MMEA_a_g_ISVKDN_SMMRF_epochs80_batch16_lr0.0001' / 'st_best_acc.pth'
)



SELECTED_EXPERIMENTS = ['C0']
SEEDS = [2025]


RUN_MODE = 'train'


NUM_WORKERS = 8
PREFETCH_FACTOR = 4
PRINT_FREQ = 20
EARLY_STOP_PATIENCE = 10
USE_AMP = True
RUN_SMOKE_BEFORE_TRAIN = True
RETRAIN_TEACHERS = False
REUSE_EXISTING_FULL_TEACHER = True
FORCE_RERUN_STUDENTS = False



PROJECT_ROOT = Path(__file__).resolve().parent
TEACHER_SCRIPT = PROJECT_ROOT / 'stage2_train_teacher.py'
STUDENT_SCRIPT = PROJECT_ROOT / 'stage2_train_student.py'
OUTPUT_ROOT = PROJECT_ROOT / 'output' / 'stage2'



EXPERIMENTS = {
    'A0': dict(name='video_baseline', fusion='full', sgsm='off', soft=0, feature=0, semantic=0),
    'A1': dict(name='soft_kd_only', fusion='full', sgsm='off', soft=1, feature=0, semantic=0),
    'A2': dict(name='sgsm_only', fusion='full', sgsm='full', soft=0, feature=1, semantic=0),
    'A3': dict(name='full_isvkdn', fusion='full', sgsm='full', soft=1, feature=1, semantic=1),
    
    
    'B0': dict(name='smmrf_mean', fusion='mean', sgsm='full', soft=1, feature=1, semantic=1),
    'B1': dict(name='smmrf_concat', fusion='concat', sgsm='full', soft=1, feature=1, semantic=1),
    'B2': dict(name='smmrf_concat_gate', fusion='concat_gate', sgsm='full', soft=1, feature=1, semantic=1),
    'B3': dict(name='smmrf_full_seeded', fusion='full', sgsm='full', soft=1, feature=1, semantic=1),
    
    'C0': dict(name='without_sgsm', fusion='full', sgsm='off', soft=1, feature=0, semantic=1),
    'C1': dict(name='raw_feature_matching', fusion='full', sgsm='raw', soft=1, feature=1, semantic=1),
    'C2': dict(name='shuffled_semantics', fusion='full', sgsm='shuffled', soft=1, feature=1, semantic=1),
    'C4': dict(name='correct_semantics_no_graph', fusion='full', sgsm='no_graph', soft=1, feature=1, semantic=1),
}


def python_executable():
    return PREFERRED_PYTHON if PREFERRED_PYTHON.is_file() else Path(sys.executable)


def run(command):
    print('\n>', ' '.join(str(value) for value in command), flush=True)
    subprocess.run([str(value) for value in command], cwd=PROJECT_ROOT, check=True)


def common_args(seed):
    result = [
        '--gaf_root', GAF_ROOT,
        '--video_root', VIDEO_ROOT,
        '--seed', seed,
        '--workers', NUM_WORKERS,
        '--prefetch_factor', PREFETCH_FACTOR,
        '--print_freq', PRINT_FREQ,
        '--early_stop_patience', EARLY_STOP_PATIENCE,
    ]
    if not USE_AMP:
        result.append('--no_amp')
    return result


def teacher_dir(fusion_mode, seed):
    return OUTPUT_ROOT / 'teachers' / f'fusion_{fusion_mode}' / f'seed_{seed}'


def student_dir(experiment_id, definition, seed):
    return OUTPUT_ROOT / 'students' / f'{experiment_id}_{definition["name"]}' / f'seed_{seed}'


def ensure_teacher(fusion_mode, seed):
    if (fusion_mode == 'full' and REUSE_EXISTING_FULL_TEACHER and
            EXISTING_FULL_TEACHER.is_file() and not RETRAIN_TEACHERS):
        print(f'\nReusing existing full-SMMRF teacher: {EXISTING_FULL_TEACHER}', flush=True)
        return EXISTING_FULL_TEACHER
    output_dir = teacher_dir(fusion_mode, seed)
    checkpoint = output_dir / 'best_teacher_acc.pth'
    if checkpoint.is_file() and not RETRAIN_TEACHERS:
        print(f'\nReusing teacher: {checkpoint}', flush=True)
        return checkpoint

    if RUN_SMOKE_BEFORE_TRAIN:
        smoke_dir = OUTPUT_ROOT / 'smoke' / 'teachers' / f'fusion_{fusion_mode}' / f'seed_{seed}'
        run([
            python_executable(), '-u', TEACHER_SCRIPT,
            '--output_dir', smoke_dir,
            '--fusion_mode', fusion_mode,
            *common_args(seed),
            '--smoke',
        ])

    run([
        python_executable(), '-u', TEACHER_SCRIPT,
        '--output_dir', output_dir,
        '--fusion_mode', fusion_mode,
        *common_args(seed),
    ])
    if not checkpoint.is_file():
        raise FileNotFoundError('Teacher training finished without a checkpoint: %s' % checkpoint)
    return checkpoint


def run_student(experiment_id, definition, seed):
    output_dir = student_dir(experiment_id, definition, seed)
    if RUN_MODE == 'train' and (output_dir / 'metrics.json').is_file() and not FORCE_RERUN_STUDENTS:
        print(f'\nSkipping completed experiment: {experiment_id}, seed={seed}', flush=True)
        return

    teacher_required = any(definition[key] > 0 for key in ('soft', 'feature', 'semantic'))
    checkpoint = (ensure_teacher(definition['fusion'], seed)
                  if teacher_required and RUN_MODE == 'train' else None)
    base_command = [
        python_executable(), '-u', STUDENT_SCRIPT,
        '--experiment_id', experiment_id,
        '--output_dir', output_dir,
        '--fusion_mode', definition['fusion'],
        '--sgsm_mode', definition['sgsm'],
        '--soft_kd_weight', definition['soft'],
        '--feature_weight', definition['feature'],
        '--semantic_weight', definition['semantic'],
        *common_args(seed),
    ]
    if checkpoint is not None:
        base_command.extend(['--teacher_load', checkpoint])
    if RUN_MODE == 'smoke':
        smoke_dir = OUTPUT_ROOT / 'smoke' / 'students' / f'{experiment_id}_{definition["name"]}' / f'seed_{seed}'
        output_index = base_command.index('--output_dir') + 1
        base_command[output_index] = smoke_dir
        base_command.append('--smoke')
        run(base_command)
        return
    if RUN_SMOKE_BEFORE_TRAIN:
        smoke_dir = OUTPUT_ROOT / 'smoke' / 'students' / f'{experiment_id}_{definition["name"]}' / f'seed_{seed}'
        smoke_command = list(base_command)
        output_index = smoke_command.index('--output_dir') + 1
        smoke_command[output_index] = smoke_dir
        smoke_command.append('--smoke')
        run(smoke_command)
    run(base_command)


def main():
    missing = [path for path in (GAF_ROOT, VIDEO_ROOT, TEACHER_SCRIPT, STUDENT_SCRIPT)
               if not path.exists()]
    if missing:
        raise FileNotFoundError('Missing required paths:\n' + '\n'.join(str(path) for path in missing))
    unknown = [name for name in SELECTED_EXPERIMENTS if name not in EXPERIMENTS]
    if unknown:
        raise ValueError('Unknown experiment IDs: %s' % unknown)
    if not SEEDS:
        raise ValueError('SEEDS cannot be empty')
    if RUN_MODE not in {'train', 'smoke'}:
        raise ValueError('RUN_MODE must be train or smoke')

    print('Python:', python_executable(), flush=True)
    print('GAF:', GAF_ROOT, flush=True)
    print('Video:', VIDEO_ROOT, flush=True)
    print('Experiments:', SELECTED_EXPERIMENTS, flush=True)
    print('Seeds:', SEEDS, flush=True)
    print('Mode:', RUN_MODE, flush=True)

    for seed in SEEDS:
        for experiment_id in SELECTED_EXPERIMENTS:
            run_student(experiment_id, EXPERIMENTS[experiment_id], seed)

    print('\nAll selected stage-2 experiments completed. Results:', OUTPUT_ROOT, flush=True)


if __name__ == '__main__':
    main()
