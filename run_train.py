"""
Project: 🍿POPCORN: High-resolution Population Maps Derived from Sentinel-1 and Sentinel-2 🌍🛰️
Nando Metzger, 2024
"""

import os
import argparse
from collections import defaultdict
import time

import numpy as np
import torch
from torch import optim
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, ConcatDataset
from torchvision import transforms
from utils.transform import OwnCompose, RandomRotationTransform, RandomHorizontalFlip, RandomVerticalFlip, RandomBrightness, RandomGamma
from tqdm import tqdm
 
import wandb
 
import gc

from arguments.train import parser as train_parser
from data.PopulationDataset import Population_Dataset, Population_Dataset_collate_fn
from utils.losses import get_loss, r2
from utils.metrics import get_test_metrics
from utils.utils import new_log, to_cuda_inplace, detach_tensors_in_dict, seed_all
from model.get_model import get_model_kwargs, model_dict
from utils.utils import load_json, apply_transformations_and_normalize
from utils.constants import config_path
 
from utils.constants import testlevels, overlap
from utils.constants import inference_patch_size as ips
from utils.utils import NumberList

torch.autograd.set_detect_anomaly(True)

import nvidia_smi
nvidia_smi.nvmlInit()


class Trainer:

    def __init__(self, args: argparse.Namespace) -> None:
        self.args = args

        # set up experiment folder
        self.experiment_folder, self.args.expN, self.args.randN = new_log(args.save_dir, args)
        self.args.experiment_folder = self.experiment_folder
        print("Experiment folder:", self.experiment_folder)
        
        # seed everything
        seed_all(args.seed)
        
        # set up dataloaders
        self.dataloaders = self.get_dataloaders(self, args)
        
        # define architecture
        model_kwargs = get_model_kwargs(args, args.model)
        self.model = model_dict[args.model](**model_kwargs).cuda()
        
        # set random seed after model initialization to ensure reproducibility of training pipline
        seed_all(args.seed+1)
        
        # number of params
        args.pytorch_total_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        args.num_effective_param = self.model.num_params
        print("Model", args.model, "; #Effective Params trainable:", args.num_effective_param)
        print("---------------------")

        # wandb config
        wandb.init(project=args.wandb_project, dir=self.experiment_folder)
        wandb.config.update(self.args)
        wandb.watch(self.model, log='all')  
        
        # seed after initialization of model to ensure reproducibility
        seed_all(args.seed+2)

        # set up optimizer and scheduler
        # Get all parameters except the head bias and the head bias parameter, only bias, if available
        head_name = ['head.6.weight','head.6.bias']
        params_with_decay = [param for name, param in self.model.named_parameters() if name not in head_name and 'unetmodel' not in name]
        params_unet_only = [param for name, param in self.model.named_parameters() if name not in head_name and name and 'unetmodel' in name]
        params_without_decay = [param for name, param in self.model.named_parameters() if name in head_name and 'unetmodel' not in name]
        self.optimizer = optim.Adam([
                {'params': params_with_decay, 'weight_decay': args.weightdecay}, # Apply weight decay here
                {'params': params_unet_only, 'weight_decay': args.weightdecay}, # Apply weight decay here
                {'params': params_without_decay, 'weight_decay': 0.0}, # No weight decay
            ] , lr=args.learning_rate)
            
        # set up scheduler    
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=args.lr_step, gamma=args.lr_gamma)
        
        # set up info
        self.info = { "epoch": 0,  "iter": 0,  "sampleitr": 0} 
        self.train_stats, self.val_stats = defaultdict(lambda: np.nan), defaultdict(lambda: np.nan)
        self.best_optimization_loss = np.inf

        # in case of checkpoint resume
        if args.resume is not None:
            self.resume(path=args.resume)

        compile = False
        if compile:
            self.model = torch.compile(self.model)

    def train(self):
        """
        Main training loop
        """
        self.pred_buffer = NumberList(300)
        self.target_buffer = NumberList(300)

        with tqdm(range(self.info["epoch"], self.args.num_epochs), leave=True) as tnr:
            tnr.set_postfix(training_loss=np.nan, validation_loss=np.nan, best_validation_loss=np.nan)
            for _ in tnr:

                self.train_epoch(tnr)
                torch.cuda.empty_cache()

                if self.args.save_model in ['last', 'both']:
                    self.save_model('last')

                # weak validation, e.g training validation
                if (self.info["epoch"] + 1) % self.args.val_every_n_epochs == 0:
                    if self.args.weak_validation:
                        self.validate_weak()
                        torch.cuda.empty_cache()

                if (self.info["epoch"] + 1) % (self.args.val_every_n_epochs) == 0:
                    self.test_target(save=True)
                    torch.cuda.empty_cache()

                    if self.args.save_model in ['last', 'both']:
                        self.save_model('last')
                
                # logging and scheduler step
                if self.args.lr_gamma != 1.0: 
                    self.scheduler.step()
                    wandb.log({**{'log_lr': np.log10(self.scheduler.get_last_lr())}, **self.info}, self.info["iter"])
                
                self.info["epoch"] += 1

    def train_epoch(self, tnr=None):
        """
        Train for one epoch
        """
        train_stats = defaultdict(float)

        # set model to train mode
        self.model.train()

        # get GPU memory usage
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(0)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        train_stats["gpu_used"] = info.used / 1e9 # in GB 

        # check if we are in unsupervised or supervised mode and adjust dataloader accordingly
        dataloader = self.dataloaders['train'] 
        self.optimizer.zero_grad()

        num_buildings, num_people = 0, 0

        with tqdm(dataloader, leave=False, total=len(dataloader)) as inner_tnr:
            inner_tnr.set_postfix(training_loss=np.nan)

            # iterate over samples of one epoch
            for i, sample in enumerate(inner_tnr):
                # self.optimizer.zero_grad()
                optim_loss = 0.0
                loss_dict_weak = {}
                loss_dict_raw = {}
                                
                # calculate global disaggregation factor, this is used to calculate the disaggregation factor which can be used to initialize the bias of the last layer
                calculate_disaggregation_factor = False
                if calculate_disaggregation_factor:
                    this_mask = sample_weak["admin_mask"]==sample_weak["census_idx"].view(-1,1,1)
                    num_buildings += (sample_weak["building_counts"] * this_mask).sum()
                    num_people += sample_weak["y"].sum()
                    print("Disaggregation factor",  (num_people/num_buildings).item())
                    continue

                # forward pass and loss computation
                sample_weak = to_cuda_inplace(sample) 
                sample_weak = apply_transformations_and_normalize(sample_weak, self.data_transform, self.dataset_stats, buildinginput=self.args.buildinginput,
                                                                    segmentationinput=self.args.segmentationinput)
                
                # check if the input is to large & freeze encoder and decoder if input is to large to fit on GPU
                num_pix = sample_weak["input"].shape[0]*sample_weak["input"].shape[2]*sample_weak["input"].shape[3]
                encoder_no_grad, unet_no_grad = False, False 
                if num_pix > self.args.limit1:
                    encoder_no_grad, unet_no_grad = True, False 
                    if num_pix > self.args.limit2:
                        encoder_no_grad, unet_no_grad = True, True  
                        if num_pix > self.args.limit3: 
                            continue
                
                # perform forward pass
                output_weak = self.model(sample_weak, train=True, return_features=False, padding=False,
                                            encoder_no_grad=encoder_no_grad, unet_no_grad=unet_no_grad, sparse=True )

                # compute loss
                loss_weak, loss_dict_weak = get_loss(
                    output_weak, sample_weak, scale=output_weak["scale"], loss=args.loss, lam=args.lam,
                    scale_regularization=args.scale_regularization, tag="weak")
                
                # Detach tensors
                loss_dict_weak = detach_tensors_in_dict(loss_dict_weak)
                
                # update loss
                optim_loss += loss_weak * self.args.lam_weak 

                for key in loss_dict_weak:
                    train_stats[key] += loss_dict_weak[key].cpu().item() if torch.is_tensor(loss_dict_weak[key]) else loss_dict_weak[key] 
                train_stats["log_count"] += 1

                # collect buffer for training stats (r2 score)
                self.pred_buffer.add(output_weak["popcount"].cpu().detach())
                self.target_buffer.add(sample_weak["y"].cpu().detach())
                                
                # detect NaN loss 
                if torch.isnan(optim_loss):
                    raise Exception("detected NaN loss..")
                if torch.isinf(optim_loss):
                    raise Exception("detected Inf loss..")
                
                # backprop
                optim_loss.backward()

                # gradient clipping
                if self.args.gradient_clip > 0.:
                    clip_grad_norm_(self.model.parameters(), self.args.gradient_clip)
                
                # if (i + 1) % self.accumulation_steps == 0:
                self.optimizer.step()
                self.optimizer.zero_grad()

                # clear memory and detach tensors
                optim_loss = optim_loss.detach()
                if output_weak is not None:
                    output_weak = detach_tensors_in_dict(output_weak)
                    del output_weak
                del sample 
                gc.collect()
                
                # clear GPU cache
                torch.cuda.empty_cache()

                # update info
                self.info["iter"] += 1 
                self.info["sampleitr"] += self.args.weak_batch_size
                # logging and stuff
                if (i+1) % self.args.val_every_i_steps == 0:
                    if self.args.weak_validation:
                        self.log_train(train_stats)
                        self.validate_weak()
                        self.model.train()

                # logging and stuff
                if (i+1) % self.args.test_every_i_steps == 0:
                    self.log_train(train_stats)
                    self.test_target(save=True)
                    self.model.train()

                if (i + 1) % min(self.args.logstep_train, len(self.dataloaders['train'])) == 0:
                    self.log_train(train_stats,(inner_tnr, tnr))
                    train_stats = defaultdict(float)
    

    def log_train(self, train_stats, tqdmstuff=None):
        train_stats = {k: v / train_stats["log_count"] for k, v in train_stats.items()}
        train_stats["Population_weak/r2"] = r2(torch.tensor(self.pred_buffer.get()),torch.tensor(self.target_buffer.get()))

        # print logs to console via tqdm
        if tqdmstuff is not None:
            inner_tnr, tnr = tqdmstuff
            inner_tnr.set_postfix(training_loss=train_stats['optimization_loss'])
            if tnr is not None:
                tnr.set_postfix(training_loss=train_stats['optimization_loss'],
                                validation_loss=self.val_stats['optimization_loss'],
                                best_validation_loss=self.best_optimization_loss)

        # upload logs to wandb
        wandb.log({**{k + '/train': v for k, v in train_stats.items()}, **self.info}, self.info["iter"])
        

    def validate_weak(self):
        self.valweak_stats = defaultdict(float)

        self.model.eval()

        with torch.no_grad():
            for valdataloader in self.dataloaders["weak_target_val"]:
                pred, gt = [], []
                for i,sample in enumerate(tqdm(valdataloader, leave=False)):
                    sample = to_cuda_inplace(sample)
                    sample = apply_transformations_and_normalize(sample, transform=None, dataset_stats=self.dataset_stats, buildinginput=self.args.buildinginput,
                                                                 segmentationinput=self.args.segmentationinput, empty_eps=self.args.empty_eps)

                    output = self.model(sample, padding=False)

                    # Colellect predictions and samples
                    pred.append(output["popcount"]); gt.append(sample["y"])

                # compute metrics
                pred = torch.cat(pred); gt = torch.cat(gt)
                self.valweak_stats = { **self.valweak_stats,
                                       **get_test_metrics(pred, gt.float().cuda(), tag="MainCensus_{}_{}".format(valdataloader.dataset.region, self.args.train_level))  }

            wandb.log({**{k + '/val': v for k, v in self.valweak_stats.items()}, **self.info}, self.info["iter"])

    def test_target(self, save=False, full=True):

        # Test on target domain
        self.model.eval()
        self.test_stats = defaultdict(float)

        with torch.no_grad(): 
            self.target_test_stats = defaultdict(float)
            for testdataloader in self.dataloaders["test_target"]:

                # inputialize the output map
                h, w = testdataloader.dataset.shape()
                output_map = torch.zeros((h, w), dtype=torch.float16)
                output_scale_map = torch.zeros((h, w), dtype=torch.float16)
                output_map_count = torch.zeros((h, w), dtype=torch.int8)

                for sample in tqdm(testdataloader, leave=False):
                    sample = to_cuda_inplace(sample)
                    sample = apply_transformations_and_normalize(sample, transform=None, dataset_stats=self.dataset_stats, buildinginput=self.args.buildinginput,
                                                                 segmentationinput=self.args.segmentationinput)

                    # get the valid coordinates
                    xl,yl = [val.item() for val in sample["img_coords"]]
                    mask = sample["mask"][0].bool()

                    # get the output with a forward pass
                    output = self.model(sample, padding=False)
                    output_map[xl:xl+ips, yl:yl+ips][mask.cpu()] += output["popdensemap"][0][mask].cpu().to(torch.float16)
                    if "scale" in output.keys() and output["scale"] is not None:
                        output_scale_map[xl:xl+ips, yl:yl+ips][mask.cpu()] += output["scale"][0][mask].cpu().to(torch.float16)

                    output_map_count[xl:xl+ips, yl:yl+ips][mask.cpu()] += 1

                # average over the number of times each pixel was visited, mask out values that are not visited of visited exactly once
                div_mask = output_map_count > 1
                output_map[div_mask] = output_map[div_mask] / output_map_count[div_mask]
                
                # average over the number of times each pixel was visited, mask out values that are not visited of visited exactly once
                if "scale" in output.keys():
                    output_scale_map[div_mask] = output_scale_map[div_mask] / output_map_count[div_mask]
                
                # save maps
                if save:
                    # save the output map
                    testdataloader.dataset.save(output_map, self.experiment_folder)
                    if "scale" in output.keys():
                        testdataloader.dataset.save(output_scale_map, self.experiment_folder, tag="SCALE_{}".format(testdataloader.dataset.region))
                
                # convert populationmap to census
                for level in testlevels[testdataloader.dataset.region]:
                    census_pred, census_gt = testdataloader.dataset.convert_popmap_to_census(output_map, gpu_mode=True, level=level)
                    self.target_test_stats = {**self.target_test_stats,
                                              **get_test_metrics(census_pred, census_gt.float().cuda(), tag="MainCensus_{}_{}".format(testdataloader.dataset.region, level))}

            wandb.log({**{k + '/targettest': v for k, v in self.target_test_stats.items()}, **self.info}, self.info["iter"])

            del output_map, output_map_count, output_scale_map


    @staticmethod
    def get_dataloaders(self, args: argparse.Namespace) -> dict: 
        """
        Get dataloaders for the source and target domains
        Inputs:
            args: command line arguments 
        Outputs:
            dataloaders: dictionary of dataloaders
        """

        # define input definitions (standards)
        input_defs = {'S1': args.Sentinel1, 'S2': args.Sentinel2, 'NIR': args.NIR}

        self.data_transform = {}
        general_transforms = [
            RandomVerticalFlip(p=0.5, allsame=True),
            RandomHorizontalFlip(p=0.5, allsame=True),
            RandomRotationTransform(angles=[90, 180, 270], p=0.75),
        ]

        self.data_transform["general"] = transforms.Compose(general_transforms)

        S2augs = [
            RandomBrightness(p=0.9, beta_limit=(0.666, 1.5)),
            RandomGamma(p=0.9, gamma_limit=(0.6666, 1.5)),
        ]

        # collect all transformations
        self.data_transform["S2"] = OwnCompose(S2augs)
        self.data_transform["S1"] = transforms.Compose([ ])
        
        # load normalization stats
        self.dataset_stats = load_json(os.path.join(config_path, 'dataset_stats.json'))
        for mkey in self.dataset_stats.keys():
            if isinstance(self.dataset_stats[mkey], dict):
                for key,val in self.dataset_stats[mkey].items():
                    self.dataset_stats[mkey][key] = torch.tensor(val)
            else:
                self.dataset_stats[mkey] = torch.tensor(val)

        # get the target regions for testing
        need_asc = ["uga"] # some regions do not have full S1 descending data, so we need to fill it with ascending data
        datasets = {
            "test_target": [ Population_Dataset( reg, patchsize=ips, overlap=overlap, sentinelbuildings=args.sentinelbuildings, ascfill=reg in need_asc, **input_defs) \
                                for reg in args.target_regions ] }
        dataloaders =  {
            "test_target":  [DataLoader(datasets["test_target"], batch_size=1, num_workers=self.args.num_workers, shuffle=False, drop_last=False) \
                                for datasets["test_target"] in datasets["test_target"] ]  }
        
            
        weak_datasets = []
        # for reg in args.target_regions_train:
        for reg, lvl in zip(args.target_regions_train, args.train_level):
            splitmode = 'train' if self.args.weak_validation else 'all'
            weak_datasets.append( Population_Dataset(reg, mode="weaksup", split=splitmode, patchsize=None, overlap=None, max_samples=args.max_weak_samples,
                                                            fourseasons=True, transform=None, sentinelbuildings=args.sentinelbuildings, 
                                                            ascfill=reg in need_asc, train_level=lvl, max_pix=self.args.max_weak_pix, max_pix_box=self.args.max_pix_box, ascAug=args.ascAug, **input_defs)  )
        dataloaders["weak_target_dataset"] = ConcatDataset(weak_datasets)
        dataloaders["train"] = DataLoader(dataloaders["weak_target_dataset"], batch_size=args.weak_batch_size, num_workers=self.args.num_workers, shuffle=True, collate_fn=Population_Dataset_collate_fn, drop_last=True)
        
        weak_datasets_val = []
        if self.args.weak_validation: 
            for reg, lvl in zip(args.target_regions_train, args.train_level):
                weak_datasets_val.append(Population_Dataset(reg, mode="weaksup", split="val", patchsize=None, overlap=None, max_samples=args.max_weak_samples,
                                                                fourseasons=True, transform=None, sentinelbuildings=args.sentinelbuildings, 
                                                                ascfill=reg in need_asc, train_level=lvl, max_pix=self.args.max_weak_pix, max_pix_box=self.args.max_pix_box, **input_defs) )
            dataloaders["weak_target_val"] = [ DataLoader(weak_datasets_val[i], batch_size=self.args.weak_val_batch_size, num_workers=self.args.num_workers, shuffle=False, collate_fn=Population_Dataset_collate_fn, drop_last=True)
                                                for i in range(len(args.target_regions_train)) ]

        return dataloaders
   

    def save_model(self, prefix=''):
        """
        Input:
            prefix: string to prepend to the filename
        """
        torch.save({
            'model': self.model.state_dict(),
            'epoch': self.info["epoch"] + 1,
            'iter': self.info["iter"],
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict(),
        }, os.path.join(self.experiment_folder, f'{prefix}_model.pth'))


    def resume(self, path, load_optimizer=True):
        """
        Input:
            path: path to the checkpoint
        """
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')

        # load checkpoint
        checkpoint = torch.load(path)
        self.model.load_state_dict(checkpoint['model'])
        if load_optimizer:
            self.optimizer.load_state_dict(checkpoint['optimizer'])
        self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.info["epoch"] = checkpoint['epoch']
        self.info["iter"] = checkpoint['iter']

        print(f'Checkpoint \'{path}\' loaded.')


if __name__ == '__main__':
    args = train_parser.parse_args()
    print(train_parser.format_values())

    trainer = Trainer(args)

    since = time.time()
    trainer.train()
    time_elapsed = time.time() - since
    print('Training completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
