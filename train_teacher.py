import os
import sys
import time
import argparse
import random
import logging
import torch
import torchvision
from torchvision import models
import torch.optim as optim
import torchvision.transforms as transforms
from uestc_paired_dataset import UESTCPairedDataset
import model.backbone as backbone
import numpy as np
import metric.loss as loss
import metric.pairsampler as pair
import torch.nn.functional as F


from torch.utils.data import DataLoader

from metric.utils import recall, count_parameters_in_MB, accuracy, AverageMeter
from metric.batchsampler import NPairs, BalancedBatchSampler
from model.embedding import LinearEmbedding

import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()  
LookupChoices = type('', (argparse.Action, ), dict(__call__=lambda a, p, n, v, o: setattr(n, a.dest, a.choices[v])))



parser.add_argument('--dataset', type=str, default='MMEA', choices=['MMEA', 'MMAct'])

parser.add_argument('--gaf_root', type=str, default=r"F:\Dataset2\UESTC-MMEA_GAF")
parser.add_argument('--video_root', type=str, default=r"G:\Dataset\Multi-modalActionRecognition\UESTC-MMEA-CL")

parser.add_argument('--modality', type=str, default='a', choices=['a', 'g'])


parser.add_argument('--output_dir', type=str, default='output/')
parser.add_argument('--workers', type=int, default=4)
parser.add_argument('--no_amp', action='store_true', help='disable CUDA mixed precision')
parser.add_argument('--smoke', action='store_true', help='one training and validation batch, no checkpoint')
parser.add_argument('--early_stop_patience', type=int, default=10,
                    help='stop after this many epochs without improved combined validation accuracy')
parser.add_argument('--mode',
                        choices=["train", "eval"],
                        default="train")


parser.add_argument('--load',
                        default=None)



parser.add_argument('--base',
                        choices=dict(googlenet=backbone.GoogleNet,
                                    inception_v1bn=backbone.InceptionV1BN,
                                    resnet18=backbone.ResNet18,
                                    resnet50=backbone.ResNet50,
                                    vggnet16=backbone.VggNet16,
                                    Sevggnet16=backbone.SeVggNet16,
                                    SeFusionVGG16=backbone.SeFusionVGG16,
                                    SemanticFusionVGG16=backbone.SemanticFusionVGG16,
                                    ),
                        default=backbone.VggNet16,
                        action=LookupChoices)


parser.add_argument('--sample',
                        choices=dict(random=pair.RandomNegative,
                                    hard=pair.HardNegative,
                                    all=pair.AllPairs,
                                    semihard=pair.SemiHardNegative,
                                    distance=pair.DistanceWeighted),
                        default=pair.AllPairs,
                        action=LookupChoices)


parser.add_argument('--loss',
                        choices=dict(l1_triplet=loss.L1Triplet,
                                    l2_triplet=loss.L2Triplet,
                                    contrastive=loss.ContrastiveLoss),
                        default=loss.L2Triplet,
                        action=LookupChoices)

parser.add_argument('--num_classes', default=32, type=int)  
parser.add_argument('--margin', type=float, default=0.2)  
parser.add_argument('--embedding_size', type=int, default=128)  
parser.add_argument('--l2normalize', choices=['true', 'false'], default='true')  

parser.add_argument('--lr', default=1e-5, type=float)  
parser.add_argument('--lr_decay_epochs', type=int, default=[25, 30, 35], nargs='+')  
parser.add_argument('--lr_decay_gamma', default=0.5, type=float)  

parser.add_argument('--batch', default=64, type=int)  
parser.add_argument('--num_image_per_class', default=5, type=int)

parser.add_argument('--epochs', default=40, type=int)  
parser.add_argument('--iter_per_epoch', type=int, default=100)  
parser.add_argument('--recall', default=[1], type=int, nargs='+')  


parser.add_argument('--seed', default=random.randint(1, 1000), type=int)  
parser.add_argument('--data', default='data')  
parser.add_argument('--save_dir', default=None)  


parser.add_argument('--print_freq', type=int, default=1, help='frequency of showing training results on console')


opts = parser.parse_args()  


opts.num_classes=32   
opts.mode='train'   

opts.modality='a_g'   

opts.base=backbone.SemanticFusionVGG16
opts.sample=pair.DistanceWeighted
opts.loss=loss.L2Triplet

opts.lr=0.0001
opts.margin=0.2
opts.batch=16
opts.epochs=80
opts.lr_decay_epochs=[40]
opts.lr_decay_gamma=0.5
opts.print_freq=1
opts.output_dir='output/'

opts.save_dir= opts.output_dir+'_'.join(map(str, [opts.dataset, opts.modality, 'ISVKDN_SMMRF',
            'epochs'+str(opts.epochs),'batch'+str(opts.batch), 'lr'+str(opts.lr)]))
if opts.smoke:
    opts.save_dir = os.path.join(opts.output_dir, 'ISVKDN_SMOKE_teacher')


if not os.path.exists(opts.save_dir):
    os.makedirs(opts.save_dir)


log_format = '%(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO, format=log_format)
fh = logging.FileHandler(os.path.join(opts.save_dir, 'log.txt'))
fh.setFormatter(logging.Formatter(log_format))
logging.getLogger().addHandler(fh)


def main():

    for set_random_seed in [random.seed, torch.manual_seed, torch.cuda.manual_seed_all]:   
        set_random_seed(opts.seed)
    torch.backends.cudnn.benchmark = True  
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    use_amp = not opts.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=use_amp, init_scale=1024)
    logging.info("args = %s", opts)   

    gaf_transform = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    train_loader = DataLoader(
        UESTCPairedDataset('train', gaf_transform, gaf_root=opts.gaf_root,
                           video_root=opts.video_root),
        batch_size=opts.batch, shuffle=True, num_workers=opts.workers,
        pin_memory=True, persistent_workers=opts.workers > 0)
    val_loader = DataLoader(
        UESTCPairedDataset('val', gaf_transform, gaf_root=opts.gaf_root,
                           video_root=opts.video_root),
        batch_size=opts.batch, shuffle=False,
        num_workers=max(1, opts.workers // 2) if opts.workers else 0,
        pin_memory=True, persistent_workers=opts.workers > 0)
    test_loader = DataLoader(
        UESTCPairedDataset('test', gaf_transform, gaf_root=opts.gaf_root,
                           video_root=opts.video_root),
        batch_size=opts.batch, shuffle=False, num_workers=0, pin_memory=True)

    UESTC_BERT=np.load('data/UESTC-MMEA-CL_BERT.npy')   
    UESTC_BERT=torch.from_numpy(UESTC_BERT)   
    UESTC_BERT=UESTC_BERT.float().cuda()   
    
    torch.cuda.empty_cache()   
    logging.info('----------- Network Initialization --------------')   
    model = opts.base(n_classes=opts.num_classes).cuda()   
    logging.info('Acce param size = %fMB', count_parameters_in_MB(model))   
    logging.info('-----------------------------------------------')   
    
    if torch.cuda.device_count() > 1:
        model = torch.nn.DataParallel(model, device_ids=[0,1]).cuda()
    if opts.load is not None:   
        model.load_state_dict(torch.load(opts.load))   
        print("Loaded Model from %s" % opts.load)   


    
    logging.info("Aligned train/val/test samples: %d/%d/%d",
                 len(train_loader.dataset), len(val_loader.dataset), len(test_loader.dataset))


    if opts.load is not None:   
        model.load_state_dict(torch.load(opts.load))   
        logging.info("Loaded Model from %s" % opts.load)

    
    
    
    criterion_cls=torch.nn.CrossEntropyLoss().cuda()
    
    criterion_semantic=torch.nn.MSELoss().cuda()
    
    
    optimizer = optim.Adam(model.parameters(), lr=opts.lr, weight_decay=1e-5)
    
    
    
    
    
    
    lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=opts.lr_decay_epochs, gamma=opts.lr_decay_gamma)
    

    def train(net, loader, ep):
        K = opts.recall   
        batch_time = AverageMeter()      
        data_time = AverageMeter()       
        
        cls_loss = AverageMeter()        
        semantic_loss = AverageMeter()   
        
        
        s_train_acc=0.
        t_train_acc=0.
        net.train()                      
        loss_all = []                    
        
        end = time.time()                
        i=1

        for s_images, t_images, labels in loader:
            t_labels = s_labels = labels
            data_time.update(time.time() - end)                     
            t_images, t_labels = t_images.cuda(non_blocking=True), t_labels.cuda(non_blocking=True)
            s_images, s_labels = s_images.cuda(non_blocking=True), s_labels.cuda(non_blocking=True)
            
            t_semantic=UESTC_BERT[t_labels]                          
            s_semantic=UESTC_BERT[s_labels]                          
            
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                s_out1,t_out1,s_out2,t_out2,s_out3,t_out3,s_out4,t_out4,s_out5,t_out5,s_out6,t_out6,s_out7,t_out7,s_out8,t_out8 = net(s_images,t_images,True)

            
            s_pred=torch.max(s_out8,1)[1]
            t_pred=torch.max(t_out8,1)[1]
            
            s_train_correct=(s_pred==s_labels).sum()
            t_train_correct=(t_pred==t_labels).sum()
            
            s_train_acc+=s_train_correct.item()
            t_train_acc+=t_train_correct.item()

            
            
            s_loss_cls=criterion_cls(s_out8.float(), s_labels)
            t_loss_cls=criterion_cls(t_out8.float(), t_labels)
            
            s_semantic_loss=criterion_semantic(s_out7.float(), s_semantic)
            t_semantic_loss=criterion_semantic(t_out7.float(), t_semantic)
            loss=(s_loss_cls+t_loss_cls)/2.0+(s_semantic_loss+t_semantic_loss)/2.0 
            loss_all.append(loss.item())

            
            cls_loss.update((s_loss_cls.item()+t_loss_cls.item()), s_images.size(0)+t_images.size(0))
            
            semantic_loss.update((s_semantic_loss.item()+t_semantic_loss.item()), s_images.size(0)+t_images.size(0))

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            batch_time.update(time.time() - end)   
            end = time.time()
            if i % opts.print_freq == 0:      
                log_str=('Epoch[{0}]:[{1:03}/{2:03}] '
                        'Batch:{batch_time.val:.4f} '
                        'Data:{data_time.val:.4f}  '
                        'Cls Loss:{loss_cls.val:.4f}({loss_cls.avg:.4f})  '
                        'Semantic Loss:{loss_semantic.val:.4f}({loss_semantic.avg:.4f})  '
                        
                        
                        
                        .format(ep, i, len(loader), batch_time=batch_time, data_time=data_time,
                        loss_cls=cls_loss, loss_semantic=semantic_loss))
                logging.info(log_str)
            i=i+1
            if opts.smoke:
                break
            
            
        logging.info('[Epoch %d] Loss: %.5f  Gyro Acc: %.5f Acce Acc: %.5f' % (ep, torch.Tensor(loss_all).mean(), 100*s_train_acc/len(loader.dataset), 100*t_train_acc/len(loader.dataset)))


    def eval(net, loader, ep):
        
        net.eval()             
        
        
        s_embeddings_all, t_embeddings_all,s_labels_all,t_labels_all = [], [], [], []
        
        s_correct = 0
        t_correct = 0
        st_correct = 0
        
        with torch.no_grad():     
            for s_images, t_images, labels in loader:
                t_labels = s_labels = labels
                t_images, t_labels = t_images.cuda(non_blocking=True), t_labels.cuda(non_blocking=True)
                s_images, s_labels = s_images.cuda(non_blocking=True), s_labels.cuda(non_blocking=True)
                t_semantic=UESTC_BERT[t_labels]                                
                s_semantic=UESTC_BERT[s_labels]
                with torch.cuda.amp.autocast(enabled=use_amp):
                    s_embedding, t_embedding = net(s_images,t_images)
                
                s_pred=torch.max(s_embedding,1)[1]
                t_pred=torch.max(t_embedding,1)[1]
                st_pred=torch.max((s_embedding+t_embedding),1)[1]        
                
                s_num_correct=(s_pred==s_labels).sum()
                t_num_correct=(t_pred==t_labels).sum()
                st_num_correct=(st_pred==t_labels).sum()
                s_correct+=s_num_correct.item()
                t_correct+=t_num_correct.item()
                st_correct+=st_num_correct.item()
                
                s_embeddings_all.append(s_embedding.data)
                t_embeddings_all.append(t_embedding.data)
                s_labels_all.append(s_labels.data)  
                t_labels_all.append(t_labels.data)
                if opts.smoke:
                    break
            
            s_embeddings_all = torch.cat(s_embeddings_all).cpu()
            t_embeddings_all = torch.cat(t_embeddings_all).cpu()
            s_labels_all = torch.cat(s_labels_all).cpu()
            t_labels_all = torch.cat(t_labels_all).cpu()
            
            
            
            s_acc = s_correct/len(loader.dataset)
            t_acc = t_correct/len(loader.dataset)
            st_acc= st_correct/len(loader.dataset)
            logging.info('[Epoch %d] gyro acc: [%.4f] acce acc: [%.4f] combined acc: [%.4f]' % (ep, s_acc*100, t_acc*100, st_acc*100))
        return s_acc, t_acc ,st_acc


    if opts.smoke:
        train(model, train_loader, 1)
        eval(model, val_loader, 1)
        logging.info('Teacher smoke test passed; no checkpoint was saved.')
        return

    if opts.mode == "eval":
        eval(model, test_loader, 0)
    else:
        if opts.early_stop_patience < 1:
            raise ValueError('early_stop_patience must be at least 1')
        
        gyro_val_acc, acce_val_acc , st_val_acc= eval(model, val_loader, 0)
        gyro_best_acc =gyro_val_acc
        acce_best_acc =acce_val_acc
        st_best_acc =st_val_acc
        
        torch.save(model.state_dict(), os.path.join(opts.save_dir, 'st_best_acc.pth'))
        epochs_without_improvement = 0
        last_epoch = 0
        
        for epoch in range(1, opts.epochs+1):
            last_epoch = epoch
            
            train(model, train_loader, epoch)
            lr_scheduler.step()
            
            gyro_val_acc, acce_val_acc ,st_val_acc= eval(model, val_loader, epoch)
            if gyro_best_acc < gyro_val_acc:      
                gyro_best_acc = gyro_val_acc
                if opts.save_dir is not None:   
                    if not os.path.isdir(opts.save_dir):
                        os.mkdir(opts.save_dir)
                    torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "gyro_best_acc.pth"))
            if acce_best_acc < acce_val_acc:      
                acce_best_acc = acce_val_acc
                if opts.save_dir is not None:
                    if not os.path.isdir(opts.save_dir):
                        os.mkdir(opts.save_dir)
                    torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "acce_best_acc.pth"))
            if st_best_acc < st_val_acc:        
                st_best_acc = st_val_acc
                epochs_without_improvement = 0
                if opts.save_dir is not None:
                    if not os.path.isdir(opts.save_dir):
                        os.mkdir(opts.save_dir)
                    torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "st_best_acc.pth"))
            else:
                epochs_without_improvement += 1
            
            if opts.save_dir is not None:                        
                if not os.path.isdir(opts.save_dir):             
                    os.mkdir(opts.save_dir)                      
                torch.save(model.state_dict(), "%s/%s"%(opts.save_dir, "last.pth"))       
                with open("%s/result.txt"%opts.save_dir, 'w') as f:       
                    
                    
                    f.write("Best gyro acc: %.4f\n" % (gyro_best_acc*100))  
                    f.write("Best acce acc: %.4f\n" % (acce_best_acc*100))  
                    f.write("Best combined acc: %.4f\n" % (st_best_acc*100))  
                    
                    
                    f.write("Final gyro acc: %.4f\n" % (gyro_val_acc*100))  
                    f.write("Final acce acc: %.4f\n" % (acce_val_acc*100))  
                    f.write("Final combined acc: %.4f\n" % (st_val_acc*100))  
                    

            
            
            
            logging.info("Best Gyro Acc: %.4f" % (gyro_best_acc*100))
            logging.info("Best Acce Acc: %.4f\n" % (acce_best_acc*100))
            logging.info("Best combined Acc: %.4f\n" % (st_best_acc*100))
            logging.info("Teacher early stopping: %d/%d epochs without combined validation accuracy improvement",
                         epochs_without_improvement, opts.early_stop_patience)
            if epochs_without_improvement >= opts.early_stop_patience:
                logging.info("Teacher early stopped at epoch %d; best combined validation accuracy: %.4f%%",
                             epoch, st_best_acc * 100)
                break
            




        best_teacher = os.path.join(opts.save_dir, 'st_best_acc.pth')
        if os.path.isfile(best_teacher):
            model.load_state_dict(torch.load(best_teacher, map_location='cpu'))
        logging.info('----------- Final held-out test --------------')
        eval(model, test_loader, last_epoch)


if __name__ == '__main__':
    main()
