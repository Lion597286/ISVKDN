

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import transforms

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.backbone.TRN import TRN
from model.backbone.vggnet import SemanticFusionVGG16
from metric.loss import SoftTarget
from transforms import Stack, ToTorchFormatTensor, GroupNormalize
from uestc_paired_dataset import UESTCPairedDataset
from uestc_semantic_mapping import sgsm_semantic_slope, sgsm_map, feature_matching_loss


def worker_init(_worker_id):
    cv2.setNumThreads(1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=0)
    parser.add_argument('--batch', type=int, default=2)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA GPU is required for this smoke test')
    torch.backends.cudnn.benchmark = True
    student = TRN(32, 8, 'RGB', base_model='BNInception',
                  consensus_type='TRNmultiscale', dropout=0.8,
                  img_feature_dim=768, print_spec=False).cuda().train()
    teacher = SemanticFusionVGG16(32).cuda().eval()
    video_transform = transforms.Compose([
        student.get_augmentation(), Stack(roll=True),
        ToTorchFormatTensor(div=False),
        GroupNormalize(student.input_mean, student.input_std)])
    gaf_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    dataset = UESTCPairedDataset('train', gaf_transform, video_transform, num_segments=8)
    loader = DataLoader(dataset, batch_size=args.batch, shuffle=False,
                        num_workers=args.workers, pin_memory=True,
                        persistent_workers=args.workers > 0,
                        worker_init_fn=worker_init)
    gyro, acc, video, labels = next(iter(loader))
    gyro, acc = gyro.cuda(non_blocking=True), acc.cuda(non_blocking=True)
    video, labels = video.cuda(non_blocking=True), labels.cuda(non_blocking=True)
    bert = torch.from_numpy(np.load('data/UESTC-MMEA-CL_BERT.npy')).float().cuda()
    batch = labels.size(0)
    optimizer = torch.optim.SGD(student.parameters(), lr=0.001)
    scaler = torch.cuda.amp.GradScaler(init_scale=1024)

    with torch.no_grad(), torch.cuda.amp.autocast():
        teacher_out = teacher(torch.cat((gyro, torch.zeros_like(gyro))),
                              torch.cat((acc, torch.zeros_like(acc))), True)
    with torch.cuda.amp.autocast():
        student_out = student(torch.cat((video, torch.zeros_like(video))),
                              return_features='distill')
    g_layers = [teacher_out[i][:batch] for i in (0, 2, 4, 6, 8)]
    a_layers = [teacher_out[i][:batch] for i in (1, 3, 5, 7, 9)]
    s_layers = [student_out[i][:batch] for i in range(5)]
    w_g = sgsm_semantic_slope(teacher_out[14], bert, 10)
    w_a = sgsm_semantic_slope(teacher_out[15], bert, 10)
    w_s = sgsm_semantic_slope(student_out[6], bert, 10)
    mapped = []
    for s, g, a in zip(s_layers, g_layers, a_layers):
        size = tuple(min(x, y) for x, y in zip(s.shape[-2:], g.shape[-2:]))
        mapped.append((sgsm_map(s, w_s, size), sgsm_map(g, w_g, size), sgsm_map(a, w_a, size)))
    fmap_loss = feature_matching_loss(mapped)
    logits = student_out[6][:batch].float()
    semantic = student_out[5][:batch].float()
    distill = (SoftTarget(4)(logits, teacher_out[14][:batch].float()) +
               SoftTarget(4)(logits, teacher_out[15][:batch].float())) / 2
    semantic_loss = (F.mse_loss(semantic, teacher_out[12][:batch].float()) +
                     F.mse_loss(semantic, teacher_out[13][:batch].float())) / 2
    loss = F.cross_entropy(logits, labels) + distill + fmap_loss + semantic_loss
    print('loss components', float(F.cross_entropy(logits, labels)), float(distill),
          float(fmap_loss), float(semantic_loss), flush=True)
    if not bool(torch.isfinite(loss)):
        raise ArithmeticError(f'Non-finite loss: {loss}')
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    gradient = sum(float(p.grad.float().abs().sum()) for p in student.parameters() if p.grad is not None)
    if not np.isfinite(gradient) or gradient <= 0:
        raise ArithmeticError(f'Invalid student gradient: {gradient}')
    scaler.step(optimizer)
    scaler.update()
    torch.cuda.synchronize()
    print('real-data AMP step OK:', 'batch', batch, 'loss', round(float(loss), 4),
          'SGSM', round(float(fmap_loss), 4), 'student gradient', round(gradient, 3),
          'peak GPU GiB', round(torch.cuda.max_memory_allocated() / 2**30, 2))


if __name__ == '__main__':
    main()
