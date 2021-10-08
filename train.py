# -*- coding: utf-8 -*-
# @Author: yirui
# @Date:   2021-09-11 10:08:03
# @Last Modified by:   blackkkkk
# @Last Modified time: 2021-09-22 23:17:27
import os
import torch.utils.data
from torch import nn
from torch.nn import DataParallel
import datetime
from training_config import BATCH_SIZE, SAVE_FREQ, RESUME, SAVE_DIR, TEST_FREQ, TOTAL_EPOCH, MODEL_PRE, GPU
from training_config import GLINT_DATA_DIR, LFW_DATA_DIR
from core import model
from core.utils import init_log
from dataloader.GLINT_loader import GLINT_Face
from dataloader.LFW_loader import LFW
from torch.optim import lr_scheduler
import torch.optim as optim
import time
from lfw_eval import parseList, evaluation_10_fold
import numpy as np
import scipy.io
import math

print('INFO:  init gpu')
# gpu init
gpu_list = ''
multi_gpus = False
if isinstance(GPU, int):
    gpu_list = str(GPU)
else:
    multi_gpus = True
    for i, gpu_id in enumerate(GPU):
        gpu_list += str(gpu_id)
        if i != len(GPU) - 1:
            gpu_list += ','
os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
print(f'INFO:  gpu used - {gpu_list}')

print('INFO:  init logging')
# other init
start_epoch = 1
save_dir = os.path.join(SAVE_DIR, MODEL_PRE + datetime.datetime.now().strftime('%Y%m%d_%H%M%S'))
if os.path.exists(save_dir):
    raise NameError('model dir exists!')
os.makedirs(save_dir)
logging = init_log(save_dir)
_print = logging.info


_print('INFO:  init training dataloader ...')
# define trainloader and testloader
trainset = GLINT_Face(data_folder=GLINT_DATA_DIR)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=BATCH_SIZE,
                                          shuffle=True, num_workers=8, drop_last=False)

_print('INFO:  init testing dataloader ...')
# nl: left_image_path
# nr: right_image_path
nl, nr, folds, flags = parseList(root=LFW_DATA_DIR)
testdataset = LFW(nl, nr)
testloader = torch.utils.data.DataLoader(testdataset, batch_size=32,
                                         shuffle=False, num_workers=8, drop_last=False)


# define model
net = model.MobileFacenet()
ArcMargin = model.ArcMarginProduct(128, trainset.class_nums)

if RESUME:
    ckpt = torch.load(RESUME)
    net.load_state_dict(ckpt['net_state_dict'])
    start_epoch = ckpt['epoch'] + 1
    _print(f'INFO: resume from {RESUME}')
    _print(f'INFO: start with epoch {start_epoch}')


# define optimizers
ignored_params = list(map(id, net.linear1.parameters()))
ignored_params += list(map(id, ArcMargin.weight))
prelu_params_id = []
prelu_params = []
for m in net.modules():
    if isinstance(m, nn.PReLU):
        ignored_params += list(map(id, m.parameters()))
        prelu_params += m.parameters()
base_params = filter(lambda p: id(p) not in ignored_params, net.parameters())

optimizer_ft = optim.SGD([
    {'params': base_params, 'weight_decay': 4e-5},
    {'params': net.linear1.parameters(), 'weight_decay': 4e-4},
    {'params': ArcMargin.weight, 'weight_decay': 4e-4},
    {'params': prelu_params, 'weight_decay': 0.0}
], lr=0.1, momentum=0.9, nesterov=True)

exp_lr_scheduler = lr_scheduler.MultiStepLR(optimizer_ft, milestones=[36, 52, 58], gamma=0.1)


net = net.cuda()
ArcMargin = ArcMargin.cuda()
if multi_gpus:
    net = DataParallel(net)
    ArcMargin = DataParallel(ArcMargin)
criterion = torch.nn.CrossEntropyLoss()


best_acc = 0.0
best_epoch = 0

# DATA_SIZE = 13309 # for testing only
DATA_SIZE = len(trainset)   

num_batches = math.ceil(DATA_SIZE / BATCH_SIZE)

def output_process (since, previous, epoch, i, duration_enumerate):
    now = time.time()
    duration_total = now - since
    duration_total_str = str(datetime.timedelta(seconds=duration_total))
    duration_batch = now - previous
    duration_batch_str = str(datetime.timedelta(seconds=duration_batch))
    duration_per_batch = duration_total / i
    time_end = now + duration_per_batch * (num_batches - i)
    time_end_str = time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time_end))
    duration_estimate = time_end - since
    duration_estimate_str = str(datetime.timedelta(seconds=duration_estimate))
    duration_remaining = time_end - now
    duration_remaining_str = str(datetime.timedelta(seconds=duration_remaining))
    duration_enumerate_str = str(datetime.timedelta(seconds=duration_enumerate))
    print(f'[epoch{epoch}] Processed batch: {i+1}/{num_batches}({100/num_batches*i}%). Duration: batch={duration_batch_str}, total={duration_total_str}, total(enum)={duration_enumerate_str}, estimate={duration_estimate_str}, remaining={duration_remaining_str}. Estimate end: {time_end_str}\r')
    return now
    
for epoch in range(start_epoch, TOTAL_EPOCH + 1):
    exp_lr_scheduler.step()
    # train model
    _print('Train Epoch: {}/{} ...'.format(epoch, TOTAL_EPOCH))
    net.train()

    train_total_loss = 0.0
    total = 0
    duration_enumerate = 0
    since = time.time()
    previous_output = since
    previous_end = since
    for i, data in enumerate(trainloader):
        duration_enumerate += time.time() - previous_end
        img, label = data[0].cuda(), data[1].cuda()
        batch_size = img.size(0)
        optimizer_ft.zero_grad()

        raw_logits = net(img)

        output = ArcMargin(raw_logits, label)
        total_loss = criterion(output, label)
        total_loss.backward()
        optimizer_ft.step()

        train_total_loss += total_loss.item() * batch_size
        total += batch_size
        if i % 200 == 0:
            previous_output = output_process(since, previous_output, epoch, i, duration_enumerate)
        previous_end = time.time()

    train_total_loss = train_total_loss / total
    time_elapsed = time.time() - since
    loss_msg = '    total_loss: {:.4f} time: {:.0f}m {:.0f}s'\
        .format(train_total_loss, time_elapsed // 60, time_elapsed % 60)
    _print(loss_msg)


    # save model
    if epoch % SAVE_FREQ == 0:
        msg = 'Saving checkpoint: {}. [Testing still outstanding]'.format(epoch)
        _print(msg)
        if multi_gpus:
            net_state_dict = net.module.state_dict()
        else:
            net_state_dict = net.state_dict()
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
        torch.save({
            'epoch': epoch,
            'net_state_dict': net_state_dict},
            os.path.join(save_dir, '%03d.ckpt' % epoch))
        _print('Saved')
            

    # test model on lfw
    if epoch % TEST_FREQ == 0:
        net.eval()
        featureLs = None
        featureRs = None
        _print('Testing Epoch - {} in progress...'.format(epoch))
        t2 = time.time()
        for data in testloader:
            for i in range(len(data)):
                data[i] = data[i].cuda()
            res = [net(d).data.cpu().numpy() for d in data]
            featureL = np.concatenate((res[0], res[1]), 1)
            featureR = np.concatenate((res[2], res[3]), 1)
            if featureLs is None:
                featureLs = featureL
            else:
                featureLs = np.concatenate((featureLs, featureL), 0)
            if featureRs is None:
                featureRs = featureR
            else:
                featureRs = np.concatenate((featureRs, featureR), 0)

        result = {'fl': featureLs, 'fr': featureRs, 'fold': folds, 'flag': flags}
        # save tmp_result
        scipy.io.savemat('./temp/tmp_result.mat', result)
        accs = evaluation_10_fold('./temp/tmp_result.mat')
        _print('    ave accuracy: {:.4f}'.format(np.mean(accs) * 100))
        time_diff = time.time()-t2
        time_dur_str = str(datetime.timedelta(seconds=time_diff))
        print(f'Testing took {time_dur_str}')

            
print('finishing training')
