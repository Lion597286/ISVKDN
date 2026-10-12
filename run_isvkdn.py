





from pathlib import Path
import os
import subprocess
import sys




RUN_MODE = "train"

GAF_ROOT = Path(r"F:\Dataset2\UESTC-MMEA_GAF")
VIDEO_ROOT = Path(r"G:\Dataset\Multi-modalActionRecognition\UESTC-MMEA-CL")


PREFERRED_PYTHON = Path(r"F:\DeepLearningSoftware\Anaconda\envs\SAKDN\python.exe")
NUM_WORKERS = 4
EARLY_STOP_PATIENCE = 10  


RETRAIN_TEACHER = False



PROJECT_ROOT = Path(__file__).resolve().parent
TEACHER_SCRIPT = PROJECT_ROOT / "train_teacher.py"
STUDENT_SCRIPT = PROJECT_ROOT / "train_student.py"
TEACHER_CHECKPOINT = (
    PROJECT_ROOT
    / "output"
    / "MMEA_a_g_ISVKDN_SMMRF_epochs80_batch16_lr0.0001"
    / "st_best_acc.pth"
)


def _python() -> Path:
    return PREFERRED_PYTHON if PREFERRED_PYTHON.is_file() else Path(sys.executable)


def _check_inputs() -> None:
    required = [GAF_ROOT, VIDEO_ROOT, TEACHER_SCRIPT, STUDENT_SCRIPT]
    missing = [str(path) for path in required if not path.exists()]
    if missing:
        raise FileNotFoundError("以下必要路径不存在：\n" + "\n".join(missing))
    if RUN_MODE not in {"train", "smoke"}:
        raise ValueError('RUN_MODE 只能是 "train" 或 "smoke"')
    if EARLY_STOP_PATIENCE < 1:
        raise ValueError("EARLY_STOP_PATIENCE 必须至少为 1")


def _run(script: Path, *extra: str) -> None:
    command = [
        str(_python()),
        "-u",
        str(script),
        "--gaf_root",
        str(GAF_ROOT),
        "--video_root",
        str(VIDEO_ROOT),
        "--workers",
        str(NUM_WORKERS),
        "--early_stop_patience",
        str(EARLY_STOP_PATIENCE),
        *extra,
    ]
    environment = os.environ.copy()
    environment["PYTHONUNBUFFERED"] = "1"
    subprocess.run(command, cwd=PROJECT_ROOT, env=environment, check=True)


def main() -> None:
    _check_inputs()
    print(f"Python: {_python()}", flush=True)
    print(f"GAF:    {GAF_ROOT}", flush=True)
    print(f"Video:  {VIDEO_ROOT}", flush=True)

    if RUN_MODE == "smoke":
        print("\n[1/2] 检查 SMMRF 教师网络……", flush=True)
        _run(TEACHER_SCRIPT, "--smoke")
        print("\n[2/2] 检查 SGSM 视频学生网络……", flush=True)
        _run(STUDENT_SCRIPT, "--smoke")
        print("\n快速检查通过，可以把 RUN_MODE 改回 train 开始完整训练。", flush=True)
        return

    if RETRAIN_TEACHER or not TEACHER_CHECKPOINT.is_file():
        print("\n[1/2] 训练 SMMRF 教师网络……", flush=True)
        _run(TEACHER_SCRIPT)
    else:
        print(f"\n[1/2] 复用已有教师权重：{TEACHER_CHECKPOINT}", flush=True)

    if not TEACHER_CHECKPOINT.is_file():
        raise FileNotFoundError(
            "教师训练结束后没有生成所需权重：\n" + str(TEACHER_CHECKPOINT)
        )

    print("\n[2/2] 训练带 SGSM 蒸馏的 TRN 视频学生网络……", flush=True)
    _run(STUDENT_SCRIPT, "--teacher_load", str(TEACHER_CHECKPOINT))
    print("\nISVKDN 训练与最终测试已完成，结果位于 output 文件夹。", flush=True)


if __name__ == "__main__":
    main()
