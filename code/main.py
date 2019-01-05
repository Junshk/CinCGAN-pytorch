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
    GLR = _NetG_DOWN(stride=2)#EDSR(args)
    GHR = EDSR(args)#_NetG_UP()#Generator(G_input_dim, num_filters, G_output_dim)
    GDN = _NetG_DOWN(stride=1)#EDSR(args)
    DLR =_NetD(stride=1)# True)# _NetD(3)#Generator(G_input_dim, num_filters, G_output_dim)
    DHR = _NetD(stride=2)#Generator(G_input_dim, num_filters, G_output_dim)
    GNO = _NetG_DOWN(stride=1)#EDSR(args)


    Loaded = torch.load('../experiment/model/EDSR_baseline_x{}.pt'.format(scale))
    GHR.load_state_dict(Loaded) 
    
    model = nn.ModuleList()

    model.append(GDN) #DN
    model.append(GHR)
    model.append(GLR) #LR
    model.append(DLR)
    model.append(DHR)
    model.append(GNO) #
    
    print(model)

    cudnn.benchmark = True
    # optionally resume from a checkpoint
    opt.resume = 'model_total_{}.pth'.format(scale)
    if opt.resume:
        if os.path.isfile(opt.resume):
            print("=> loading checkpoint '{}'".format(opt.resume))
            checkpoint = torch.load(opt.resume)
            opt.start_epoch = checkpoint["epoch"] + 1
            model.load_state_dict(checkpoint["model"].state_dict())
        else:
            print("=> no checkpoint found at '{}'".format(opt.resume))
            
    # model[4] = _NetD_(4, True)#, True, 4)


    print("===> Setting GPU")
    if opt.cuda:
        model = model.cuda()

    print("===> Setting Optimizer")
    # optimizer = optim.Adam(model.parameters(), lr=opt.lr)#, momentum=opt.momentum, weight_decay=opt.weight_decay)
    
    
    if opt.test_only:
        print('===> Testing')
        test(test_data_loader, model ,opt.start_epoch)
        return

    print("===> Training Step 1.")
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        train(training_data_loader, training_high_loader, model, epoch, False)
        save_checkpoint(model, epoch, scale)
        test(test_data_loader, model, epoch)

    print("===> Training Step 2.")
    opt.lr = 1e-4
    for epoch in range(opt.start_epoch, opt.epochs + 1):
        train(training_data_loader, training_high_loader, model, epoch, True)
        save_checkpoint(model, epoch, scale)
        test(test_data_loader, model, epoch)

def adjust_learning_rate(epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 10 epochs"""
    lr = opt.lr * (opt.gamma ** (epoch // opt.lr_decay))
    return lr
def train(training_data_loader, training_high_loader, model, epoch, joint=False):
    
    lr = adjust_learning_rate(epoch)
    step_weight = 1 if joint else 0
    '''
        for param_group in optimizer.param_groups:
        param_group["lr"] = lr
    '''
    

    print("Epoch = {}, lr = {}".format(epoch, lr))

    model.train()
    optG = torch.optim.Adam(list(model[0].parameters())+list(model[1].parameters())+ list(model[2].parameters())+list(model[5].parameters()), lr=lr, weight_decay=0)
    optD = torch.optim.Adam(list(model[3].parameters())+list(model[4].parameters()), lr=lr, weight_decay=0)

    for iteration, (batch0, batch1) in enumerate(zip(training_data_loader, training_high_loader)):
        input, target, bicubic, input_v, target_v = batch1[0], batch1[1], batch1[2], batch0[0], batch0[1]
            
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
        
        
        # TV LOSS  weight 2
        dn_ = model[0](input_v)    
        hr_ = model[1](dn_)

        TV_loss = 0.5 * tvloss(dn_) + 2 * step_weight * tvloss(hr_)
        optG.zero_grad()
        TV_loss.backward()
        optG.step()
        
        patch_size = args.patch_size//args.scale[0]  

        ########### ordinray D lr ##############
        dn_ = model[0](input_v)

        tarv2 = model[4](bicubic)
        lr__v2 = model[4](dn_.detach())
        D_hlr_loss_l = \
                        criterionD(lr__v2, y_fake.expand_as(lr__v2)) + criterionD(tarv2, y_real.expand_as(lr__v2))

        optD.zero_grad()
        (D_hlr_loss_l).backward(retain_graph=True)
        optD.step()
        

        nf = model[4](dn_)#dn_)#, True)
        # update G
        DG_hlr_loss_l = \
                        criterionD(nf, y_real.expand_as(nf))
        optG.zero_grad()

        DG_hlr_loss_l.backward()
        optG.step()

        ########### ordinray D hr ##############
        dn_ = model[0](input_v)
        hr_ = model[1](dn_)


        tarv2 = model[3](target)#starget)
        lr__v2 = model[3](hr_.detach())#dn_.detach())
        D_hlr_loss_h = \
                        (criterionD(lr__v2, y_fake.expand_as(lr__v2)) + criterionD(tarv2, y_real.expand_as(lr__v2)) )*step_weight

        optD.zero_grad()
        (D_hlr_loss_h).backward(retain_graph=True)
        optD.step()
        

        hf = model[3](hr_)#dn_)#, True)
        # update G
        DG_hlr_loss_h = \
                        criterionD(hf, y_real.expand_as(hf)) * step_weight
        optG.zero_grad()

        DG_hlr_loss_h.backward()
        optG.step()



        ########## cycle & idt #############        

        bi_l = model[0](bicubic)
        bi_ = model[1](bicubic)

    
        idt_loss = \
                    criterion_(bi_, target) * step_weight\
                    + criterion1(bi_l, bicubic)
        optG.zero_grad()

        idt_loss = idt_loss * 5
        idt_loss.backward()
        optG.step()
        
        dn_ = model[0](input_v)
        hr_ = model[1](dn_)
        no_ = model[5](dn_)
        lr_ = model[2](hr_)
        cyc_loss =  criterion(no_, input_v) \
                   + criterion(lr_, input_v) * step_weight
        cyc_loss = cyc_loss * 10

        optG.zero_grad()
        
        cyc_loss.backward()
        optG.step()
        # print(dn_.max().item(), no_.max().item(), lr_.max().item(), bicubic.max().item(), input_v.max().item(), target.max().item())
        
        
        
        
        if iteration%10 == 0:
            utils.save_image(torch.cat([target, target_v], -1)[0], 'targets.png')
            utils.save_image(torch.cat([input, input_v, bicubic], -1)[0], 'inputs.png')
            with torch.no_grad():
                sr_ = model[1](model[0](input))
                sr_r = model[2](sr_)
                sr = model[2](target)# model[1](model[0](target))
                srr = model[1](model[0](sr))# model[1](model[0](target))
                psnr_ = -20 *((sr_ - target).pow(2).mean().pow(0.5)).log10()
                psnr = -20*((sr - input).pow(2).mean().pow(0.5)).log10()
            image = torch.cat([target[0], sr_[0], model[1](bicubic)[0]], -1)
            image_ = torch.cat([input[0], bicubic[0], sr[0], sr_r[0], model[0](input)[0], model[5](model[0](input))[0]],-1)
            utils.save_image(image, '_resulth.png')
            utils.save_image(image_, '_resultl.png')

            print("===> Epoch[{}]({}/{}): Loss: idt {:.6f} cyc {:.6f} D {:.6f} {:.6f}, G: {:.6f} {:.6f}, psnr_hr: {:.6f}, psnr_lr {:.6f} "\
                    .format(epoch, iteration, len(training_data_loader), idt_loss.data[0], cyc_loss.data[0], \
                        D_hlr_loss_h.data[0], D_hlr_loss_l.data[0], DG_hlr_loss_h.data[0], DG_hlr_loss_l.data[0], psnr_, psnr,))

def calc_psnr(sr, hr, scale, rgb_range, benchmark=True):
    psnr = 0
    # diff = (sr - hr).data.div(rgb_range)
    sr.data.div(rgb_range)
    hr.data.div(rgb_range)
    
    shave = scale + 6
    sr = sr[:, :, shave:-shave, shave:-shave]
    for i in range(2*shave-1):
        for j in range(2*shave-1):
            valid = (sr-hr[:, :, i:-2*shave+i, j:-2*shave+j])
            mse = valid.pow(2).mean()
            if psnr < -10 * math.log10(mse):
                psnr = -10* math.log10(mse)


    return psnr

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
            sr_r = model[2](sr_)
            sr = model[2](target)# model[1](model[0](target))
            srr = model[1](model[0](sr))# model[1](model[0](target))
            utils.save_image(sr_, 'result/h_{}.png'.format(iteration))
            utils.save_image(sr, 'result/l_{}.png'.format(iteration))
            psnr_ = calc_psnr(sr_, target, args.scale[0], 1)#-20 *((sr_ - target).pow(2).mean().pow(0.5)/1).log10()
            psnr = calc_psnr(sr, input, args.scale[0], 1)#-20*((sr - input).pow(2).mean().pow(0.5)/1).log10()
        avg += psnr#.data
        avg_ += psnr_#.data

        
        print("===> ({}/{}): psnr a: {:.10f} b: {:.10f} "\
                    .format(iteration, len(test_data_loader), psnr, psnr_,))
    print('a', avg/n, 'b', avg_/n)
    with open('test.txt', 'a') as f:
        f.write('{} {} {} {} {}\n'.format(epoch,'lr_retore', avg/n,'hr_restore', avg_/n))

def to_numpy(var):
    return var.data.cpu().numpy()

        
def save_checkpoint(model, epoch, scale=2):
    model_out_path =  "model_total_{}.pth".format(scale)
    state = {"epoch": epoch ,"model": model}
    if not os.path.exists("checkpoint/"):
        os.makedirs("checkpoint/")

    torch.save(state, model_out_path)

    print("Checkpoint saved to {}".format(model_out_path))

if __name__ == "__main__":
    main()
