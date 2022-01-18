import os
import sys
import argparse

# 没有什么要填的!!!
# ========================
# Parser from args/configs
# ========================
#------- parser arguments ------
parser = argparse.ArgumentParser()
parser.add_argument('--configs',required=True)
parser.add_argument('--device',default=None, help='--device 0,1,2,3')
parser.add_argument('--evaluate',default=False, action='store_true')
parser.add_argument('--eval_ckpt_pth',default=None,help='ckpt for evaluate')
parser.add_argument('--debug', default=False, action='store_true')
parser.add_argument('--newconfig', default=False, action='store_true')
parser.add_argument('--best_ckpt_to_test', default=None)
args = parser.parse_args()
#--------- Device ----------
if args.device != 'cpu':
    device_ids = args.device.strip()
    os.environ['CUDA_VISIBLE_DEVICES'] = device_ids
    device = 'cuda'
    print(f'--------- Device = cuda:{device_ids} ---------')
    device_ids = [int(idd) for idd in device_ids.split(',')]
else:
    device = 'cpu'
    print('--------- Device = cpu ---------')


from utils.config import Config, configs
import random
import torch
import torch.nn as nn
import numpy as np
import shutil
import torch.backends.cudnn as cudnn
from torch.utils.data import DataLoader
import tensorboardX
from tqdm import tqdm, trange
import logging
#------- load configs ----------
configs.update_from_modules(args.configs)
#-------------------------------
# ==============================
# Fix Random Seed, Creat logfile
# ==============================
print(f'------------ Fix Random Seed: {configs.seed}------------')
seed = configs.seed
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

# =====================
# Dataset, Dataloader
# =====================
#-------- Train/Test/Valid Data --------
print('---------- Dataset, Dataloader ------------')
loaders = {}
dataset = configs.dataset()
try:
    if dataset.collate_fn is not None:
        print('use custom train/valid collate function')
        collate_fn = dataset.collate_fn
except:
    from torch.utils.data.dataloader import default_collate
    collate_fn = default_collate
    print('use default train/valid collate function')

for i, s in enumerate(['train', 'valid']): # (s=='train'),
    loaders[s] = DataLoader(dataset[s], batch_size=configs.dataloader.batch_size[s],shuffle=(s=='train'),
                            num_workers=configs.dataloader.num_workers, 
                            collate_fn=collate_fn,
                            pin_memory=configs.dataloader.pin_memory,
                            worker_init_fn=lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id + (i + 1)))

if args.evaluate and 'other_dataset' in configs.evaluate:
    print('------------ Use Evaluate Dataset to Test -------------')
    dataset = configs.evaluate.other_dataset()
else:
    print('------------ Use Train Dataset to Test ------------')
    dataset = configs.dataset()
try:
    if dataset.collate_fn is not None:
        print('use custom test collate function')
        collate_fn = dataset.collate_fn
except:
    from torch.utils.data.dataloader import default_collate
    collate_fn = default_collate
    print('use default test collate function')
loaders['test'] = DataLoader(dataset['test'], batch_size=configs.dataloader.batch_size['test'],shuffle=(s=='train'),
                            num_workers=configs.dataloader.num_workers, 
                            collate_fn=collate_fn,
                            pin_memory=configs.dataloader.pin_memory,
                            worker_init_fn=lambda worker_id: np.random.seed(np.random.get_state()[1][0] + worker_id + 3))


# =======================================================
# Model & Device (parallel/non-parallel), cudnn Benchmark
# =======================================================
#--------- Model ----------
print('------------ model -----------')
model = configs.model()
#print(model)
#--------- Parallel ----------
if device != 'cpu':
    if not configs.deterministic:
        cudnn.benchmark = True
    else:
        cudnn.deterministic = True
        cudnn.benchmark = False

    #if configs.parallel and len(device_ids)>1:
    #    assert torch.cuda.device_count() > 1
    model = nn.DataParallel(model)
    print(f'Use Parallel: model on device {device_ids}')

model.to(device)

# =======================
# Train Tools (Criterion, Optimizer, Scheduler)
# =======================
print('------------ Train Tools -------------')
#------------------ Criterion ---------------------
criterion = configs.train.criterion() # loss class in pytorch or yourself
#------------------ Optimizer ---------------------
# always Adam or SGD+momentum
optimizer = configs.train.optimizer(model.parameters())
#------------------ Scheduler ---------------------
scheduler = configs.train.scheduler(optimizer)
# =======================
# Train One Epoch Kernel
# =======================
torch.autograd.set_detect_anomaly(True)
def train(model, dataloader, criterion, optimizer, scheduler, writer, current_step):
    model.train()
    for inputs, targets in tqdm(dataloader, desc='train', ncols=0):
        if not isinstance(inputs, (list, tuple)):
            inputs = inputs.to(device)
        if not isinstance(targets, (list, tuple)):
            targets = targets.to(device)
        #inputs = inputs.cuda(1)
        #targets = targets.cuda(1)
        #print(f'inputs = {inputs[:, :3, :].permute(0, 2, 1)}')
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        writer.add_scalar('loss/train',loss.item(), current_step)
        current_step += 1
    if scheduler is not None:
        scheduler.step()
        #print(loss.item())
# ======================
# Valid One Epoch Kernel
# ======================
# meters = {'meter1_{test/valid}':meter1, 'meter2_{test/valid}':meter2}
# best_results = {'meter1_{test/valid}':result1, 'meter2_{test/valid}':result2}
def valid(model, dataloader, meters, best_flags, best_results, writer, current_step):
    model.eval()
    results = {}
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='valid', ncols=0):
        #for inputs, targets in dataloader:
            if not isinstance(inputs, (list, tuple)):
                inputs = inputs.to(device)
            if not isinstance(targets, (list, tuple)):
                targets = targets.to(device)
            outputs = model(inputs)
            for meter in meters.values():
                meter.update(outputs, targets)
        
        for k, meter in meters.items():
            results[k] = meter.compute()
            if isinstance(results[k], dict):
                if not isinstance(best_results[k], dict):
                    best_results[k] = {}
                if not isinstance(best_flags[k], dict):
                    best_flags[k] = {}
                for name, value in results[k].items():
                    writer.add_scalar(f'{k}/{name}/valid', value, current_step)
                    try:
                        if value > best_results[k][name]:
                            best_results[k][name] = value
                            best_flags[k][name] = True
                        else:
                            best_flags[k][name] = False
                    except KeyError:
                        best_results[k][name] = 0
            else:
                writer.add_scalar(f'{k}/valid', results[k], current_step)
                #print(best_results)
                if results[k] > best_results[k]:
                    best_results[k] = results[k]
                    best_flags[k] = True
                else:
                    best_flags[k] = False
    
    return results

def test(model, dataloader, meters, configs):
    model.eval()
    results = {}
    with torch.no_grad():
        for inputs, targets in tqdm(dataloader, desc='test', ncols=0):
        #for inputs, targets in dataloader:
            if not isinstance(inputs, (list, tuple)):
                inputs = inputs.to(device)
            if not isinstance(targets, (list, tuple)):
                targets = targets.to(device)
            outputs = model(inputs)
            for meter in meters.values():
                meter.update(outputs, targets)
        
        for k, meter in meters.items():
            results[k] = meter.compute()
            if isinstance(results[k], dict):
                for name, value in results[k].items():
                    print(f'results[{k}][{name}] = {value}')
            else:
                print(f'results[{k}] = {results[k]}')
    return results
# ==========================================
# If Evaluate, turn to the evaluate function
# ==========================================
if args.evaluate:
    print('------------ Evaluate Begin ------------')
    if args.eval_ckpt_pth is not None and os.path.exists(args.eval_ckpt_pth):
        ckpt = torch.load(args.eval_ckpt_pth)
        model.load_state_dict(ckpt['model'])
    elif args.best_ckpt_to_test is not None:
        print(f'use {args.best_ckpt_to_test} best ckpt')
        ckpt = torch.load(configs.train.best_ckpt_paths[args.best_ckpt_to_test])
        model.load_state_dict(ckpt['model'])
    elif os.path.exists(configs.train.common_ckpt_path):
        print('use train common ckpt')
        ckpt = torch.load(configs.train.common_ckpt_path)
        model.load_state_dict(ckpt['model'])
    else:
        print('ERROR: No checkpoint file !')
    meters = {}
    for k, meter in configs.evaluate.meters.items():
        meters[k.format('test')] = meter()

    if configs.evaluate.fn is not None:
        evaluate = configs.evaluate.fn
    else:
        evaluate = test
    evaluate(model, loaders['test'], meters, configs)
    sys.exit()
# ===============
# Main Train
# ===============
num_epochs = configs.train.num_epochs
start_epoch = 0
# ---- create checkpoint save path for this experiment
os.makedirs(os.path.join(os.getcwd(), configs.train.ckpt_dir), exist_ok=True)
try:
    logfile = configs.train.logfile
except KeyError: 
    logfile = None
BASIC_FORMAT = '[%(levelname)s] %(asctime)s:%(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
logging.basicConfig(filename=logfile, format=BASIC_FORMAT, datefmt=DATE_FORMAT, level=logging.DEBUG)
logging.info(configs.exp_name)
# -------- Read CheckPoint if given -------------
train_common_ckpt_path = configs.train.common_ckpt_path
# Mulit-best for Mulit-Metrics
# {'meter1_valid':best_path1, 'meter2_valid':best_path2, ...}
train_best_ckpt_paths = configs.train.best_ckpt_paths

# -------- Initialize Tensorboard Writer, Meters, Result recorders -------------
writer = tensorboardX.SummaryWriter(f'runs/{configs.exp_name}')

# Meters, Result recorders
meters, best_flags, best_results = {}, {}, {}
current_step = 0
# Multi-Meter for Multi-Metrics
# meters = {'meter1_valid':meter1,...}
# best_results = {'meter1_valid':0,...}
for k, meter in configs.train.meters.items():
    meters[k.format('valid')] = meter()
    best_flags[k.format('valid')] = False
    best_results[k.format('valid')] = 0
    #print(best_results)
# -------- Just for recover the train progress ------------
if os.path.exists(train_common_ckpt_path) and not args.debug:
    print('------------ Use checkpoint to recover train process ----------')
    ckpt = torch.load(train_common_ckpt_path)
    model.load_state_dict(ckpt['model'])
    optimizer.load_state_dict(ckpt['optimizer'])
    scheduler.load_state_dict(ckpt['scheduler'])
    start_epoch = ckpt['last_epoch'] + 1
    best_results = ckpt['best_results']
    current_step = ckpt['current_step']
    if not args.newconfig:
        configs = ckpt['configs']

# -------- Main Train Step -------------
print('------------ Begin train process ------------')
try:
    if configs.train.fn is not None:
        print('use custom train function')
        train = configs.train.fn
except:
    print('use default train function')
try:
    if configs.train.valid_fn is not None:
        print('use custom valid function')
        valid = configs.train.valid_fn
except:
    print('use default valid function')

for epoch in range(start_epoch, configs.train.num_epochs):
    print(f'epoch = {epoch}/{configs.train.num_epochs}')
    logging.info(f'epoch = {epoch}/{configs.train.num_epochs}')
    np.random.seed(epoch)
    train(model, loaders['train'], criterion, optimizer, scheduler, writer, current_step)
    # 每个epoch换一个新的
    for k, meter in configs.train.meters.items():
        meters[k.format('valid')] = meter()
    
    if not (epoch+1)%configs.train.valid_interval:
        results = valid(model, loaders['valid'], meters, best_flags, best_results, writer, current_step)
        current_step += len(loaders['train'])
        #scheduler.step(results[''])
        for k in results.keys():
            if isinstance(results[k], dict):
                assert isinstance(best_results[k], dict)
                assert isinstance(best_flags[k], dict)
                for name, value in results[k].items():
                    logging.info(f'results[{k}][{name}] = {value}, best_results[{k}][{name}] = {best_results[k][name]}')
            else:
                logging.info(f'results[{k}] = {results[k]}, best_results[{k}] = {best_results[k]}')
        #--------- Save CheckPoint ----------
        if not args.debug:
            torch.save(
                {'model':model.state_dict(),
                'optimizer':optimizer.state_dict(),
                'scheduler':scheduler.state_dict(),
                'last_epoch':epoch,
                'best_results':best_results,
                'current_step':current_step,
                'configs':configs
                }, train_common_ckpt_path)
            for k, v in best_flags.items():
                if isinstance(best_flags[k], dict):
                    for name, _ in best_flags[k].items():
                        shutil.copyfile(train_common_ckpt_path, train_best_ckpt_paths.format(k + '_' + name))
                else:
                    if v:
                        shutil.copyfile(train_common_ckpt_path, train_best_ckpt_paths.format(k))

writer.close()