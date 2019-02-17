import argparse, os
import torch
import random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torchvision.utils as utils
import data
from model.edsr import EDSR
from option import args
import copy
import numpy as np
import math
# Training settings
import scipy.misc
import torch.nn.functional as F
from srresnet import _NetG_DOWN, _NetD
import PIL
from torchvision import transforms
import matplotlib.pyplot as plt
from loss.tvloss import TVLoss
from utility import calc_psnr_pixsh
opt = args
opt.gpus = opt.n_GPUs
opt.start_epoch = 0

print(opt)

opt.cuda = not opt.cpu

criterion1 = nn.L1Loss()#size_average=False)
criterion = nn.MSELoss()#size_average=False)
criterion_ = nn.MSELoss(size_average=False)
criterionD= nn.MSELoss()#size_average=False)
tvloss = TVLoss()

torch.set_num_threads(4)



def main():
    global opt, model
    if opt.cuda:
        print("=> use gpu id: '{}'".format(opt.gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.gpus)
        if not torch.cuda.is_available():
                raise Exception("No GPU found or Wrong gpu id, please run without --cuda")

    opt.seed = random.randint(1, 10000)
    print("Random Seed: ", opt.seed)
    torch.manual_seed(opt.seed)
    if opt.cuda:
        torch.cuda.manual_seed(opt.seed)

    cudnn.benchmark = True
    scale = int(args.scale[0])
    print("===> Loading datasets")
    
    opt.n_train = 400
    loader = data.Data(opt)
    opt_high = copy.deepcopy(opt)
    opt_high.offset_train = 400
    opt_high.n_train = 400

    loader_high = data.Data(opt_high)

    training_data_loader = loader.loader_train
    training_high_loader = loader_high.loader_train
    test_data_loader = loader.loader_test

    print("===> Building model")
    GLR = _NetG_DOWN(stride=2)
    GHR = EDSR(args)
    GDN = _NetG_DOWN(stride=1)
    DLR =_NetD(stride=1)
    DHR = _NetD(stride=2)
    GNO = _NetG_DOWN(stride=1)


    Loaded = torch.load('../experiment/model/EDSR_baseline_x{}.pt'.format(scale))
    GHR.load_state_dict(Loaded) 
    
    model = nn.ModuleList()

    model.append(GDN) #DN
    model.append(GHR)
    model.append(GLR) #LR
    model.append(DLR)
    model.append(DHR)
    model.append(GNO) #
    
    optG = torch.optim.Adam(list(model[0].parameters())+list(model[1].parameters())+ list(model[2].parameters())+list(model[5].parameters()), lr=opt.lr, weight_decay=0)
    optD = torch.optim.Adam(list(model[3].parameters())+list(model[4].parameters()), lr=opt.lr, weight_decay=0)
    
    optimizer =  [optG, optD]

    print(model)

    cudnn.benchmark = True
    # optionally resume from a checkpoint
    opt.resume = 'model_total_{}.pth'.format(scale)
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            
            optimizer = checkpoint['optimizer']
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
            
    step = 2 if opt.start_epoch > opt.epochs else 1
            

    print("===> Setting GPU")
    if opt.cuda:
        model = model.cuda()

    # print("===> Setting Optimizer")
    
    
    if opt.test_only:
        print('===> Testing')
        test(test_data_loader, model ,opt.start_epoch)
        return

    if step == 1:
        print("===> Training Step 1.")
        for epoch in range(opt.start_epoch, opt.epochs + 1):
            train(training_data_loader, training_high_loader, model, optimizer, epoch, False)
            save_checkpoint(model, optimizer, epoch, scale)
            test(test_data_loader, model, epoch)
        torch.save(model.state_dict(),'backup.pt')
    elif step == 2:
        print("===> Training Step 2.")
        opt.lr = 1e-4
        for epoch in range(opt.start_epoch + 1, opt.epochs*2 + 1):
            train(training_data_loader, training_high_loader, model, optimizer, epoch, True)
            save_checkpoint(model, optimizer, epoch, scale)
            test(test_data_loader, model, epoch)

def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (opt.gamma ** ((epoch%opt.epochs) // opt.lr_decay))
    return lr

def train(training_data_loader, training_high_loader, model, optimizer, epoch, joint=False):
    
    lr = adjust_learning_rate(epoch)
    step_weight = 1 if joint else 0
    
    optG, optD = optimizer

    for param_group in optG.param_groups:
        param_group["lr"] = lr
    for param_group in optD.param_groups:
        param_group["lr"] = lr

    print("Epoch = {}, lr = {}".format(epoch, lr))

    model.train()

    for iteration, (batch0, batch1) in enumerate(zip(training_data_loader, training_high_loader)):

        input_v, target_v = batch0[0], batch0[1] # input domain dataset
        input, target, bicubic = batch1[0], batch1[1], batch1[2] # target domain dataset

        # we do not know target_v and input
        # input : unknown input, target : high resolution image, bicubic : low resolution image downsampled by bicubic kernel

        y_real = torch.ones(input_v.size(0),1,1,1)
        y_fake = torch.zeros(input_v.size(0),1,1,1)


        if opt.cuda:
            target = target.cuda() /args.rgb_range
            input = input.cuda()/args.rgb_range
            bicubic= bicubic.cuda()/args.rgb_range
            target_v = target_v.cuda()/args.rgb_range
            input_v = input_v.cuda()/args.rgb_range
            y_real = y_real.cuda()
            y_fake = y_fake.cuda()
        
        
        optG.zero_grad()

        ########### ordinray D lr ##############
        optD.zero_grad()

        dn_ = model[0](input_v)
        real_lr = model[3](bicubic)
        fake_lr = model[3](dn_.detach())

        D_lr_loss = \
                        criterionD(fake_lr, y_fake.expand_as(fake_lr)) \
                        + criterionD(real_lr, y_real.expand_as(real_lr))

        (D_lr_loss).backward(retain_graph=True)
        optD.step()
        


        fake_lr_g = model[3](dn_)
        DG_lr_loss = \
                        criterionD(fake_lr_g, y_real.expand_as(fake_lr_g))

        DG_lr_loss.backward(retain_graph=True)

        # TV lr loss
        dn_ = model[0](input_v)    

        TV_loss_lr = 0.5 * tvloss(dn_)
        TV_loss_lr.backward(retain_graph=True)

        ######### cycle & idt lr ###########

        bi_l = model[0](bicubic)
        idt_loss_l = \
                     criterion1(bi_l, bicubic)
        idt_loss_l = idt_loss_l * 5
        idt_loss_l.backward(retain_graph=True)

        
        dn_ = model[0](input_v)
        no_ = model[5](dn_)
        
        cyc_loss_l = criterion(no_, input_v) 
        cyc_loss_l = cyc_loss_l * 10
        cyc_loss_l.backward(retain_graph=True)

        # update G(lr)
        optG.step()

        optG.zero_grad()
        ########### ordinray D hr ##############
        optD.zero_grad()

        dn_ = model[0](input_v)
        hr_ = model[1](dn_)


        real_hr = model[4](target)
        fake_hr = model[4](hr_.detach())
        D_hr_loss = \
                        (criterionD(fake_hr, y_fake.expand_as(fake_hr))\
                         + criterionD(real_hr, y_real.expand_as(real_hr)) )*step_weight

        (D_hr_loss).backward(retain_graph=True)
        optD.step()
        

        fake_hr_g = model[4](hr_)
        DG_hr_loss = \
                        criterionD(fake_hr_g, y_real.expand_as(fake_hr_g)) * step_weight

        DG_hr_loss.backward(retain_graph=True)
        
        # TV hr loss

        dn_ = model[0](input_v)    
        hr_ = model[1](dn_)

        TV_loss_hr = 2 * step_weight * tvloss(hr_)
        TV_loss_hr.backward(retain_graph=True)

        ########## cycle & idt hr ###########        

        bi_ = model[1](bicubic)

        idt_loss = \
                    criterion_(bi_, target) * step_weight


        idt_loss = idt_loss * 5
        idt_loss.backward(retain_graph=True)
        
        dn_ = model[0](input_v)
        hr_ = model[1](dn_)
        lr_ = model[2](hr_)

        cyc_loss = \
                   criterion(lr_, input_v) * step_weight
        cyc_loss = cyc_loss * 10
        
        cyc_loss.backward()

        # update G(hr)
        optG.step()
        
        
        
        if iteration%10 == 0:
            with torch.no_grad():
                sr_ = model[1](model[0](input))
                sr_r = model[2](sr_)
                sr = model[2](target)
                srr = model[1](model[0](sr))
                psnr_ = -20 *((sr_ - target).pow(2).mean().pow(0.5)).log10()
                psnr = -20*((sr - input).pow(2).mean().pow(0.5)).log10()

            image = torch.cat([target[0], sr_[0], model[1](bicubic)[0]], -1)
            image_ = torch.cat([input[0], bicubic[0], sr[0], sr_r[0], model[0](input)[0], model[5](model[0](input))[0]],-1)
            utils.save_image(image, '_resulth.png')
            utils.save_image(image_, '_resultl.png')

            print("===> Epoch[{}]({}/{}): Loss: idt {:.6f} {:.6f} cyc {:.6f}  {:.6f} D {:.6f} {:.6f}, G: {:.6f} {:.6f}, psnr_hr: {:.6f}, psnr_lr {:.6f} "\
                    .format(epoch, iteration, len(training_data_loader), idt_loss.data[0], idt_loss_l.data[0], cyc_loss.data[0], cyc_loss_l.data[0],\
                        D_hr_loss.data[0], D_lr_loss.data[0], DG_hr_loss.data[0], DG_lr_loss.data[0], psnr_, psnr,))


def test(test_data_loader, model, epoch):
    avg_ = 0
    avg = 0
    n = len(test_data_loader)
    model.eval()
    for iteration, batch in enumerate(test_data_loader):
        input, target = batch[0], batch[1]
        if opt.cuda: 
            target = target.cuda()/args.rgb_range
            input = input.cuda()/args.rgb_range

        with torch.no_grad():
            sr_ = model[1](model[0](input))
            sr = model[0](input)
            utils.save_image(sr_, 'result/h_{}.png'.format(iteration))
            utils.save_image(sr, 'result/l_{}.png'.format(iteration))
            psnr_ = calc_psnr_pixsh(sr_, target, args.scale[0], 1)
            psnr = calc_psnr_pixsh(sr, input, args.scale[0], 1)
        avg += psnr#.data
        avg_ += psnr_#.data

        
        print("===> ({}/{}): psnr lr: {:.10f} hr: {:.10f} "\
                    .format(iteration, len(test_data_loader), psnr, psnr_,))
    print('lr psnr', avg/n, 'hr psnr', avg_/n)
    with open('test.txt', 'a') as f:
        f.write('{} {} {} {} {}\n'.format(epoch,'lr', avg/n,'hr', avg_/n))

def to_numpy(var):
    return var.data.cpu().numpy()

        
def save_checkpoint(model, optimizer, epoch, scale=2):
    model_out_path =  "model_total_{}.pth".format(scale)
    state = {"epoch": epoch ,"model": model, 'optimizer': optimizer}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
