from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import os
import sys
import time
import argparse
import logging
from uestc_paired_dataset import UESTCPairedDataset
from uestc_semantic_mapping import sgsm_semantic_slope, sgsm_map, feature_matching_loss
import model.backbone as backbone
import metric.pairsampler as pair
from metric.batchsampler import NPairs
import numpy as np
import torch
import torchvision
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
import cv2

from torch.utils.data import DataLoader
import torch.nn.functional as F
import matplotlib.pyplot as plt
from metric.utils import recall, count_parameters_in_MB, accuracy, AverageMeter
from metric.batchsampler import NPairs
from metric.loss import HardDarkRank, RkdDistance, RKdAngle, L2Triplet, AttentionTransfer, SP, SE_Fusion,SoftTarget
from model.embedding import LinearEmbedding
from transforms import *
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))
parser.add_argument('--dataset', type=str, default='UTD', choices=['UTD', 'MMAct'])
parser.add_argument('--gaf_root', type=str, default=r"F:\Dataset2\UESTC-MMEA_GAF")
parser.add_argument('--video_root', type=str, default=r"G:\Dataset\Multi-modalActionRecognition\UESTC-MMEA-CL")
parser.add_argument('--modality', type=str, default='a', choices=['a', 'g'])
parser.add_argument('--student_base',
                        choices=dict(googlenet=backbone.GoogleNet,
                                    inception_v1bn=backbone.InceptionV1BN,
                                    resnet18=backbone.ResNet18,
                                    resnet50=backbone.ResNet50,
                                    vggnet16=backbone.VggNet16,
                                    Sevggnet16=backbone.SeVggNet16,
                                    SeFusionVGG16=backbone.SeFusionVGG16,
                                    SemanticFusionVGG16=backbone.SemanticFusionVGG16,
                                    TSN=backbone.TSN,
                                    TRN=backbone.TRN,
                                    ),
                        default=backbone.ResNet50,
                        action=LookupChoices)
parser.add_argument('--teacher_base',
                        choices=dict(googlenet=backbone.GoogleNet,
                                    inception_v1bn=backbone.InceptionV1BN,
                                    resnet18=backbone.ResNet18,
                                    resnet50=backbone.ResNet50,
                                    Semantic_ResNet18=backbone.Semantic_ResNet18,
                                    Semantic_ResNet50=backbone.Semantic_ResNet50,
                                    vggnet16=backbone.VggNet16,
                                    Sevggnet16=backbone.SeVggNet16,
                                    SeFusionVGG16=backbone.SeFusionVGG16,
                                    ),
                        default=backbone.VggNet16,
                        action=LookupChoices)

parser.add_argument('--consensus_type', type=str, default='avg',
                    choices=['avg', 'max', 'topk', 'identity', 'rnn', 'cnn','TRN', 'TRNmultiscale'])
parser.add_argument('--arch', type=str, default="resnet101")    

parser.add_argument('--img_feature_dim', default=256, type=int, help="the feature dimension for each frame")

parser.add_argument('--dropout', '--do', default=0.5, type=float,
                    metavar='DO', help='dropout ratio (default: 0.5)')
parser.add_argument('--num_classes', default=25, type=int)

parser.add_argument('--triplet_ratio', default=0, type=float)      
parser.add_argument('--dist_ratio', default=0, type=float)         
parser.add_argument('--angle_ratio', default=0, type=float)        
parser.add_argument('--sp_ratio', default=0, type=float)           
parser.add_argument('--st_ratio', default=0, type=float)           
parser.add_argument('--GCAM_ratio', default=0, type=float)         
parser.add_argument('--semantic_ratio', default=0, type=float)     
parser.add_argument('--dark_ratio', default=0, type=float)         
parser.add_argument('--dark_alpha', default=2, type=float)         
parser.add_argument('--dark_beta', default=3, type=float)          
parser.add_argument('--at_ratio', default=0, type=float)           

parser.add_argument('--tea_layer_name', type=str, default=None,help='last convolutional layer name')
parser.add_argument('--stu_layer_name', type=str, default=None,help='last convolutional layer name')


parser.add_argument('--triplet_sample',
                    choices=dict(random=pair.RandomNegative,
                                 hard=pair.HardNegative,
                                 all=pair.AllPairs,
                                 semihard=pair.SemiHardNegative,
                                 distance=pair.DistanceWeighted),
                    default=pair.DistanceWeighted,
                    action=LookupChoices)

parser.add_argument('--triplet_margin', type=float, default=0.2)



parser.add_argument('--teacher_load', type=str,
                    default='output/MMEA_a_g_ISVKDN_SMMRF_epochs80_batch16_lr0.0001/st_best_acc.pth')
parser.add_argument('--sgsm_k', type=int, default=10)
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--no_amp', action='store_true', help='disable CUDA mixed precision')
parser.add_argument('--smoke', action='store_true', help='one training and validation batch, no checkpoint')
parser.add_argument('--early_stop_patience', type=int, default=10,
                    help='stop after this many epochs without improved video validation accuracy')




parser.add_argument('--lr', default=1e-4, type=float)
parser.add_argument('--data', default='data')
parser.add_argument('--epochs', default=80, type=int)
parser.add_argument('--batch', default=64, type=int)
parser.add_argument('--iter_per_epoch', default=100, type=int)                  
parser.add_argument('--lr_decay_epochs', type=int, default=[40, 60], nargs='+')   
parser.add_argument('--lr_decay_gamma', type=float, default=0.1)    
parser.add_argument('--recall', default=[1], type=int, nargs='+')
parser.add_argument('--save_dir', default=None)
parser.add_argument('--load', default=None)

parser.add_argument('--print_freq', type=int, default=1, help='frequency of showing training results on console')

opts = parser.parse_args()
opts.dataset='MMEA'
opts.modality='a_g_v'
opts.num_classes=32
opts.student_base=backbone.TRN
opts.consensus_type='TRNmultiscale'
opts.arch='BNInception' 
opts.num_segments=8
opts.teacher_base=backbone.SemanticFusionVGG16

opts.epochs=70
opts.lr=0.001
opts.dropout=0.8
opts.lr_decay_epochs=[35]
opts.lr_decay_gamma=0.5
opts.batch=16
opts.img_feature_dim=768

opts.sp_ratio=0    
opts.st_ratio=1
opts.GCAM_ratio=1
opts.semantic_ratio=1

opts.print_freq=1
opts.output_dir='output/'



opts.save_dir= opts.output_dir+'_'.join(map(str, ['st',str(opts.st_ratio),opts.dataset, opts.modality,'teacher','SemanticFusionVGG16','student','TRN', 
            'arch',str(opts.arch),'seg'+str(opts.num_segments),'epochs'+str(opts.epochs),'batch'+str(opts.batch), 'lr'+str(opts.lr),'dropout'+str(opts.dropout)]))
if opts.smoke:
    opts.save_dir = os.path.join(opts.output_dir, 'ISVKDN_SMOKE_student')

if not os.path.exists(opts.save_dir):
    os.makedirs(opts.save_dir)
log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(opts.save_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)

def video_worker_init(_worker_id):
    cv2.setNumThreads(1)

def main():
    if not opts.smoke and not os.path.isfile(opts.teacher_load):
        raise FileNotFoundError(
            f"Teacher checkpoint not found: {opts.teacher_load}. "
            "Run train_teacher.py first or provide --teacher_load."
        )
    torch.backends.cudnn.benchmark = True  
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_amp = not opts.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=1024)

    print("CUDA Available:", torch.cuda.is_available())
    print("Current CUDA Device:", torch.cuda.current_device())
    print("Device Name:", torch.cuda.get_device_name(torch.cuda.current_device()))

    logging.info('----------- Network Initialization --------------')
    
    
    
    student = opts.student_base(opts.num_classes, opts.num_segments, 'RGB', 
    base_model=opts.arch, consensus_type=opts.consensus_type,  dropout=opts.dropout, img_feature_dim=opts.img_feature_dim, partial_bn=True).cuda()
    
    teacher = opts.teacher_base(n_classes=opts.num_classes)
    
    logging.info('Teacher param size = %fMB', count_parameters_in_MB(teacher))   
    logging.info('Student param size = %fMB', count_parameters_in_MB(student))
    logging.info('-----------------------------------------------')
    
    UESTC_MMEA_BERT=np.load('data/UESTC-MMEA-CL_BERT.npy')
    UESTC_MMEA_BERT=torch.from_numpy(UESTC_MMEA_BERT)
    UESTC_MMEA_BERT=UESTC_MMEA_BERT.float().cuda()
    
    crop_size = student.crop_size           
    scale_size = student.scale_size         
    input_mean = student.input_mean         
    input_std = student.input_std           
    policies = student.get_optim_policies() 
    train_augmentation = student.get_augmentation()     
    normalize = GroupNormalize(input_mean, input_std)   

    gaf_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_video_transform = torchvision.transforms.Compose([
        train_augmentation, Stack(roll=opts.arch == 'BNInception'),
        ToTorchFormatTensor(div=opts.arch != 'BNInception'), normalize])
    eval_video_transform = torchvision.transforms.Compose([
        GroupScale(int(scale_size)), GroupCenterCrop(crop_size),
        Stack(roll=opts.arch == 'BNInception'),
        ToTorchFormatTensor(div=opts.arch != 'BNInception'), normalize])
    def paired_loader(split, video_transform, shuffle=False, workers=None):
        workers = opts.workers if workers is None else workers
        return DataLoader(
            UESTCPairedDataset(split, gaf_transform, video_transform,
                               num_segments=opts.num_segments, gaf_root=opts.gaf_root,
                               video_root=opts.video_root),
            batch_size=opts.batch, shuffle=shuffle, num_workers=workers,
            pin_memory=True, persistent_workers=workers > 0 and split != 'test',
            worker_init_fn=video_worker_init)

    train_loader = paired_loader('train', train_video_transform, shuffle=True)
    val_loader = paired_loader('val', eval_video_transform,
                               workers=max(1, opts.workers // 2) if opts.workers else 0)
    test_loader = paired_loader('test', eval_video_transform, workers=0)
    teacher_val_loader = DataLoader(
        UESTCPairedDataset('val', gaf_transform, gaf_root=opts.gaf_root,
                           video_root=opts.video_root),
        batch_size=opts.batch, shuffle=False, num_workers=0, pin_memory=True)

    torch.cuda.empty_cache()            


    if torch.cuda.device_count() > 1:                     
        student = torch.nn.DataParallel(student, device_ids=[0,1]).cuda()   
        teacher = torch.nn.DataParallel(teacher, device_ids=[0,1]).cuda()   
    
    logging.info("Aligned train/val/test samples: %d/%d/%d",
                 len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))

    if opts.load is not None:             
        student.load_state_dict(torch.load(opts.load))     
        logging.info("Loaded Model from %s" % opts.load)
    
    if not opts.smoke:
        teacher.load_state_dict(torch.load(opts.teacher_load))
    student = student.cuda()
    teacher = teacher.cuda()
    teacher.requires_grad_(False)

    
    optimizer = torch.optim.SGD(policies,          
                                opts.lr,
                                momentum=0.9,
                                weight_decay=5e-4)
    

    
    
    
    sp_criterion=SP().cuda()               
    st_criterion = SoftTarget(4).cuda()    
    cls_criterion=torch.nn.CrossEntropyLoss().cuda()
    criterion_semantic=torch.nn.MSELoss().cuda()
     
    

    
    


    def train(loader, ep):
        K = opts.recall
        batch_time = AverageMeter()
        data_time = AverageMeter()
        epoch_time = AverageMeter()
        
        
        loss_sp= AverageMeter()
        loss_gcam = AverageMeter()
        loss_st= AverageMeter()
        loss_cls= AverageMeter()
        loss_semantic= AverageMeter()
        top1_recall = AverageMeter()
        top1_prec = AverageMeter()
        
        teacher.eval()
        student.train()

        
        
        semantic_loss_all = []
        cls_loss_all=[]
        sp_loss_all = []
        st_loss_all = []
        gcam_loss_all = []
        loss_all = []
        train_acc=0.
        
        end = time.time()
        i=1
        for t_images_two, t_images_one, s_videos, s_labels in loader:
            t_labels_one = t_labels_two = s_labels
            

            
            
            t_images_one = t_images_one.cuda(non_blocking=True)
            t_images_two = t_images_two.cuda(non_blocking=True)
            s_videos = s_videos.cuda(non_blocking=True)
            s_labels = s_labels.cuda(non_blocking=True)
            t_labels_one = t_labels_two = s_labels
            t_images_one_ablation = torch.zeros_like(t_images_one)
            t_images_two_ablation = torch.zeros_like(t_images_two)
            s_videos_ablation = torch.zeros_like(s_videos)
            
            t_images_one_combined=torch.cat((t_images_one,t_images_one_ablation),0)
            t_images_two_combined=torch.cat((t_images_two,t_images_two_ablation),0)
            s_videos_combined=torch.cat((s_videos,s_videos_ablation),0)

            with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
                s_out1,t_out1,s_out2,t_out2,s_out3,t_out3,s_out4,t_out4,s_out5,t_out5,s_out6,t_out6,s_out7,t_out7,s_out8,t_out8 = teacher(t_images_two_combined,t_images_one_combined, True)

            with torch.cuda.amp.autocast(enabled=use_amp):
                conv_out_conv2,conv_out_3a,conv_out_4b,conv_out_4d,conv_out_5b,v_semantic,video_output = student(s_videos_combined, return_features='distill')


            t_side=len(t_labels_one)
            s_side=len(s_labels)

            
            
            w_gyro = sgsm_semantic_slope(s_out8, UESTC_MMEA_BERT, opts.sgsm_k)
            w_acc = sgsm_semantic_slope(t_out8, UESTC_MMEA_BERT, opts.sgsm_k)
            w_video = sgsm_semantic_slope(video_output, UESTC_MMEA_BERT, opts.sgsm_k)

            video_output=video_output[0:s_side,:]
            v_semantic=v_semantic[0:s_side,:]
            conv_out_conv2=conv_out_conv2[0:s_side,:]
            conv_out_5b=conv_out_5b[0:s_side,:]
            s_out1=s_out1[0:t_side,:]
            t_out1=t_out1[0:t_side,:]
            s_out2=s_out2[0:t_side,:]
            t_out2=t_out2[0:t_side,:]
            s_out3=s_out3[0:t_side,:]
            t_out3=t_out3[0:t_side,:]
            s_out4=s_out4[0:t_side,:]
            t_out4=t_out4[0:t_side,:]
            s_out5=s_out5[0:t_side,:]
            t_out5=t_out5[0:t_side,:]
            s_out7=s_out7[0:t_side,:]
            t_out7=t_out7[0:t_side,:]
            s_out8=s_out8[0:t_side,:]
            t_out8=t_out8[0:t_side,:]

            
            
            feature_pairs = [
                (conv_out_conv2, s_out1, t_out1),
                (conv_out_3a[:s_side], s_out2, t_out2),
                (conv_out_4b[:s_side], s_out3, t_out3),
                (conv_out_4d[:s_side], s_out4, t_out4),
                (conv_out_5b, s_out5, t_out5),
            ]
            mapped_pairs = []
            for student_feature, gyro_feature, acc_feature in feature_pairs:
                spatial_size = tuple(min(x, y) for x, y in zip(
                    student_feature.shape[-2:], gyro_feature.shape[-2:]))
                mapped_pairs.append((
                    sgsm_map(student_feature, w_video, spatial_size),
                    sgsm_map(gyro_feature, w_gyro, spatial_size),
                    sgsm_map(acc_feature, w_acc, spatial_size)))

            pred=torch.max(video_output,1)[1]
            train_correct=(pred==s_labels).sum()
            train_acc+=train_correct.item()

            

            

            

            st_loss=opts.st_ratio * (st_criterion(video_output.float(),s_out8.float())+st_criterion(video_output.float(),t_out8.float()))/2.0

            
            gcam_loss = opts.GCAM_ratio * feature_matching_loss(mapped_pairs)

            cls_loss=cls_criterion(video_output.float(),s_labels)

            semantic_loss=opts.semantic_ratio*(criterion_semantic(v_semantic.float(), s_out7.float())+criterion_semantic(v_semantic.float(), t_out7.float()))/2.0

            loss = st_loss+gcam_loss+cls_loss+semantic_loss 
            

            
            
            
            loss_gcam.update(gcam_loss.item(), s_videos.size(0))
            loss_st.update(st_loss.item(), s_videos.size(0))
            loss_cls.update(cls_loss.item(), s_videos.size(0))
            loss_semantic.update(semantic_loss.item(), s_videos.size(0))

            rec = recall(video_output, s_labels, K=K)
            prec = accuracy(video_output, s_labels, topk=(1,))
            top1_recall.update(rec[0], s_videos.size(0))
            top1_prec.update(prec[0]/100, s_videos.size(0))
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            st_loss_all.append(st_loss.item())
            gcam_loss_all.append(gcam_loss.item())
            cls_loss_all.append(cls_loss.item())
            semantic_loss_all.append(semantic_loss.item())
            loss_all.append(loss.item())
            
            
            if i % opts.print_freq == 0:
                log_str=('Epoch[{0}]:[{1:03}/{2:03}] '
                        
                        
                        
                        
                        
                        'St:{st_loss.val:.5f} ({st_loss.avg:.4f}) '
                        'Gcam:{gcam_loss.val:.5f} ({gcam_loss.avg:.4f}) '
                        'Cls:{cls_loss.val:.5f}({cls_loss.avg:.4f}) '
                        'Se:{semantic_loss.val:.5f} ({semantic_loss.avg:.4f}) '
                        'recall@1:{top1_recall.val:.2f}({top1_recall.avg:.2f}) '
                        'pre@1:{top1_prec.val:.2f}({top1_prec.avg:.2f}) '.format(
                        ep, i, len(loader),st_loss=loss_st,gcam_loss=loss_gcam,cls_loss=loss_cls,semantic_loss=loss_semantic,top1_recall=top1_recall,top1_prec=top1_prec))
                logging.info(log_str)
            
            
            i=i+1
            if opts.smoke:
                break
        epoch_time.update(time.time() - end)
        logging.info('[Epoch %d] Time:  %.5f, Loss: %.5f, St: %.5f, Gcam: %.5f, Se: %.5f, Acc: %.5f\n' %\
            (ep, epoch_time.val, torch.Tensor(loss_all).mean(), torch.Tensor(st_loss_all).mean(), torch.Tensor(gcam_loss_all).mean(),torch.Tensor(semantic_loss_all).mean(),100*train_acc/len(loader.dataset)))


    def eval_teacher(net, loader, ep):
        K = opts.recall
        net.eval()
        
        s_embeddings_all, t_embeddings_all,s_labels_all,t_labels_all = [], [], [], []
        s_correct = 0
        t_correct = 0
        
        with torch.no_grad():
            for s_images, t_images, labels in loader:
                t_labels = s_labels = labels
                t_images, t_labels = t_images.cuda(non_blocking=True), t_labels.cuda(non_blocking=True)
                s_images, s_labels = s_images.cuda(non_blocking=True), s_labels.cuda(non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    s_embedding, t_embedding = net(s_images,t_images)
                s_pred=torch.max(s_embedding,1)[1]
                t_pred=torch.max(t_embedding,1)[1]
                s_num_correct=(s_pred==s_labels).sum()
                t_num_correct=(t_pred==t_labels).sum()
                s_correct+=s_num_correct.item()
                t_correct+=t_num_correct.item()
                s_embeddings_all.append(s_embedding.float().detach())
                t_embeddings_all.append(t_embedding.float().detach())
                s_labels_all.append(s_labels.data)  
                t_labels_all.append(t_labels.data)
            s_embeddings_all = torch.cat(s_embeddings_all).cpu()
            t_embeddings_all = torch.cat(t_embeddings_all).cpu()
            s_labels_all = torch.cat(s_labels_all).cpu()
            t_labels_all = torch.cat(t_labels_all).cpu()
            
            s_prec = accuracy(s_embeddings_all, s_labels_all, topk=(1,))
            s_acc = s_correct/len(loader.dataset)
            t_acc = t_correct/len(loader.dataset)
            logging.info('[Epoch %d] Teacher 1 acc: [%.4f] Teacher 2 acc: [%.4f]' % (ep, s_acc*100, t_acc*100))    
        return s_embeddings_all, t_embeddings_all
    
    def eval_student(net, loader, ep):
        K=opts.recall
        net.eval()
        correct = 0
        embeddings_all= []
        labels_all= []
        with torch.no_grad():
            for i,(_, _, images, labels) in enumerate(loader, start=1):
                images, labels = images.cuda(non_blocking=True), labels.cuda(non_blocking=True)
                with torch.cuda.amp.autocast(enabled=use_amp):
                    video_output = net(images, return_features=False)
                pred=torch.max(video_output,1)[1]
                num_correct=(pred==labels).sum()
                correct+=num_correct.item()
                embeddings_all.append(video_output.float().detach())
                labels_all.append(labels.data)
                if opts.smoke:
                    break
            embeddings_all = torch.cat(embeddings_all).cpu()
            labels_all = torch.cat(labels_all).cpu()
            rec = recall(embeddings_all, labels_all, K=K)
            
            acc = correct/(len(loader.dataset))
            logging.info('[Epoch %d] recall@1: [%.4f]' % (ep, 100 * rec[0]))
            logging.info('[Epoch %d] acc: [%.4f]' % (ep, acc*100))    
        return rec[0], acc, embeddings_all, labels_all

    if opts.smoke:
        train(train_loader, 1)
        eval_student(student, val_loader, 1)
        logging.info('Student smoke test passed; no checkpoint was saved.')
        return

    if opts.early_stop_patience < 1:
        raise ValueError('early_stop_patience must be at least 1')

    logging.info('----------- Teacher Network performance --------------')

    one_prec, two_prec=eval_teacher(teacher, teacher_val_loader, 0)

    logging.info('----------- Student Network performance  --------------')

    best_val_recall, best_val_acc, s_prec, labels_all  = eval_student(student, val_loader, 0)
    
    torch.save(student.state_dict(), os.path.join(opts.save_dir, 'best_acc.pth'))

    combined_acc=accuracy((one_prec+two_prec+s_prec), labels_all, topk=(1,))
    best_combined_acc=combined_acc[0]
    combined_rec = recall((one_prec+two_prec+s_prec), labels_all, K=[1])
    best_combined_rec=combined_rec[0]
    logging.info('----------- Combined Network performance  --------------')
    logging.info('Combined acc: [%.4f]\n' % (best_combined_acc)) 
    logging.info('Combined rec: [%.4f]\n' % (best_combined_rec)) 
    epochs_without_improvement = 0
    last_epoch = 0
    for epoch in range(1, opts.epochs+1):
        last_epoch = epoch
        adjust_learning_rate(optimizer, epoch, opts.lr_decay_epochs)
        train(train_loader, epoch)
        val_recall, val_acc, s_prec, labels_all = eval_student(student, val_loader, epoch)
        val_combined_acc=accuracy((one_prec+two_prec+s_prec), labels_all, topk=(1,))
        val_comb_acc=val_combined_acc[0]
        val_combined_rec = recall((one_prec+two_prec+s_prec), labels_all, K=[1])
        val_comb_rec=val_combined_rec[0]
        logging.info('[Epoch %d] combined acc: [%.4f]' % (epoch, val_comb_acc)) 
        logging.info('[Epoch %d] combined rec: [%.4f]\n' % (epoch, val_comb_rec*100)) 
        if best_val_recall < val_recall:
            best_val_recall = val_recall
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(student.state_dict(), "%s/%s"%(opts.save_dir, "best_recall.pth"))

        if best_val_acc < val_acc:
            best_val_acc = val_acc
            epochs_without_improvement = 0
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(student.state_dict(), "%s/%s"%(opts.save_dir, "best_acc.pth"))
        else:
            epochs_without_improvement += 1

        if best_combined_acc < val_comb_acc:
            best_combined_acc = val_comb_acc
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(student.state_dict(), "%s/%s"%(opts.save_dir, "best_combined.pth"))

        if best_combined_rec < val_comb_rec:
            best_combined_rec = val_comb_rec
            if opts.save_dir is not None:
                if not os.path.isdir(opts.save_dir):
                    os.mkdir(opts.save_dir)
                torch.save(student.state_dict(), "%s/%s"%(opts.save_dir, "best_combined_rec.pth"))        
   
        F_measure=(2*best_val_acc*best_val_recall)/max(best_val_acc+best_val_recall, 1e-12)
        combined_F_measure=(2*best_combined_acc/100*best_combined_rec)/max(best_combined_acc/100+best_combined_rec, 1e-12)
        if opts.save_dir is not None:
            if not os.path.isdir(opts.save_dir):
                os.mkdir(opts.save_dir)
            torch.save(student.state_dict(), "%s/%s" % (opts.save_dir, "last.pth"))
            with open("%s/result.txt" % opts.save_dir, 'w') as f:
                f.write("Best Test recall@1: %.4f\n" % (best_val_recall * 100))
                f.write("Final Recall@1: %.4f\n" % (val_recall * 100))
                f.write("Best Test Acc: %.4f\n" % (best_val_acc * 100))
                f.write("Final Acc: %.4f\n" % (val_acc * 100))
                f.write("Best Combined Acc: %.4f\n" % (best_combined_acc))
                f.write("Final Combined Acc: %.4f\n" % (val_comb_acc ))
                f.write("F-measure: %.4f\n" % (F_measure*100))
                f.write("Combined F-measure: %.4f\n" % (combined_F_measure*100))

        logging.info("Best Eval Recall@1: %.4f" % (best_val_recall*100))
        logging.info("Best Eval Acc: %.4f" % (best_val_acc*100))
        logging.info("Best Eval Combined Acc: %.4f" % (best_combined_acc))
        logging.info("Eval F-measure: %.4f" % (F_measure*100))
        logging.info("Eval Combined F-measure: %.4f\n" % (combined_F_measure*100))
        logging.info("Student early stopping: %d/%d epochs without video validation accuracy improvement",
                     epochs_without_improvement, opts.early_stop_patience)
        if epochs_without_improvement >= opts.early_stop_patience:
            logging.info("Student early stopped at epoch %d; best video validation accuracy: %.4f%%",
                         epoch, best_val_acc * 100)
            break

    best_student = os.path.join(opts.save_dir, 'best_acc.pth')
    if os.path.isfile(best_student):
        student.load_state_dict(torch.load(best_student, map_location='cpu'))
    logging.info('----------- Final held-out test --------------')
    eval_student(student, test_loader, last_epoch)


def adjust_learning_rate(optimizer, epoch, lr_steps):
    
    decay = opts.lr_decay_gamma ** (sum(epoch >= np.array(lr_steps)))
    lr = opts.lr * decay
    decay = 5e-4
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr * param_group['lr_mult']
        param_group['weight_decay'] = decay * param_group['decay_mult']

def get_last_conv_name(net):
    




    layer_name = None
    for name, m in net.named_modules():
        if isinstance(m, nn.Conv2d):
            layer_name = name
    return layer_name        

class ConvFeatureExtraction(object):
    




    def __init__(self, net, layer_name):
        self.net = net
        self.layer_name = layer_name
        self.feature = None
        self.handlers = []
        self._register_hook()

    def _get_features_hook(self, module, input, output):
        self.feature = output

    def _register_hook(self):
        for (name, module) in self.net.named_modules():
            if name == self.layer_name:
                self.handlers.append(module.register_forward_hook(self._get_features_hook))

    def remove_handlers(self):
        for handle in self.handlers:
            handle.remove()

    def __call__(self, inputs):
        




        self.net.zero_grad()
        output = self.net(inputs)  

        feature = self.feature  

        return feature  

def norm_image(image):
    




    image = image.copy()
    image -= np.max(np.min(image), 0)
    image /= np.max(image)
    image *= 255.
    return np.uint8(image)

def gen_cam(image, mask):
    





    
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    heatmap = heatmap[..., ::-1]  
    
    cam = heatmap + np.float32(image)
        
    
    
    
    plt.imshow(norm_image(heatmap))
    plt.show()
    plt.imshow(norm_image(cam))
    plt.show()

    
    

    return norm_image(cam)

if __name__ == '__main__':
    main()
