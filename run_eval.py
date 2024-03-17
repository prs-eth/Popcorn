import os
import argparse
from collections import defaultdict
import time

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

import wandb

import rasterio
from rasterio.windows import Window
from shutil import copyfile

# from arguments import eval_parser
from arguments.eval import parser as eval_parser
from data.PopulationDataset_target import Population_Dataset_target
from utils.metrics import get_test_metrics
from utils.utils import to_cuda_inplace, seed_all
from model.get_model import get_model_kwargs, model_dict
from utils.utils import load_json, apply_transformations_and_normalize
from utils.constants import config_path

from utils.constants import  overlap, testlevels, testlevels_eval
from utils.constants import inference_patch_size as ips

import nvidia_smi
nvidia_smi.nvmlInit()


class Trainer:

    def __init__(self, args: argparse.Namespace):
        self.args = args

        # set up experiment folder
        self.args.experiment_folder = os.path.join("/",os.path.join(*args.resume[0].split("/")[:-1]), "eval_outputs_ensemble_{}_members_{}".format(time.strftime("%Y%m%d-%H%M%S"), len(args.resume)))
        self.experiment_folder = self.args.experiment_folder
        print("Experiment folder:", self.experiment_folder)

        if not os.path.exists(self.experiment_folder):
            os.makedirs(self.experiment_folder)

        # seed before dataloader initialization
        seed_all(args.seed)

        # set up dataloaders
        self.dataloaders = self.get_dataloaders(self, args)
        
        # define architecture
        self.model = []
        for _ in args.resume:
            if args.model in model_dict:
                model_kwargs = get_model_kwargs(args, args.model)
                model = model_dict[args.model](**model_kwargs).cuda()
                self.model.append(model)
            else:
                raise ValueError(f"Unknown model: {args.model}")
        
        # wandb config
        wandb.init(project=args.wandb_project, dir=self.args.experiment_folder)
        wandb.config.update(self.args)

        # seed after initialization
        seed_all(args.seed+2)

        # initialize log dict
        self.info = { "epoch": 0,  "iter": 0,  "sampleitr": 0}

        # checkpoint resume
        for j, checkpoint in enumerate(args.resume):
            if args.resume is not None:
                self.resume(checkpoint, j)


    def test_target(self, save=False, full=False, save_scatter=False):
        
        # Test on target domain
        save_scatter = save_scatter
        for j in range(len(self.model)):
            self.model[j].eval()
        self.test_stats = defaultdict(float)

        with torch.no_grad(): 
            self.target_test_stats = defaultdict(float)
            for testdataloader in self.dataloaders["test_target"]: 

                # inputialize the output map
                h, w = testdataloader.dataset.shape()
                output_map = torch.zeros((h, w), dtype=torch.float32)
                output_scale_map = torch.zeros((h, w), dtype=torch.float32)
                output_map_count = torch.zeros((h, w), dtype=torch.int16)

                # if len(self.model) > 1:
                output_map_squared = torch.zeros((h, w), dtype=torch.float32)
                # output_STD_map = torch.zeros((h, w), dtype=torch.float32)
                output_scale_map_squared = torch.zeros((h, w), dtype=torch.float32)
                # output_scale_STD_map = torch.zeros((h, w), dtype=torch.float32)

                for sample in tqdm(testdataloader, leave=True):
                    sample = to_cuda_inplace(sample)
                    sample = apply_transformations_and_normalize(sample,  transform=None, dataset_stats=self.dataset_stats, buildinginput=self.args.buildinginput,
                                                                      segmentationinput=self.args.segmentationinput)

                    # get the valid coordinates
                    xl,yl = [val.item() for val in sample["img_coords"]]
                    mask = sample["mask"][0].bool()

                    # get the output with a forward pass
                    popdense = torch.zeros((len(self.model), ips, ips), dtype=torch.float32, device="cuda")
                    scale = torch.zeros((len(self.model), ips, ips), dtype=torch.float32, device="cuda")
                    popdense_squared = torch.zeros((len(self.model), ips, ips), dtype=torch.float32, device="cuda")
                    scale_squared = torch.zeros((len(self.model), ips, ips), dtype=torch.float32, device="cuda")

                    # Evaluate each model in the ensemble
                    for i, model in enumerate(self.model):
                        this_output = model(sample, padding=False)
                        popdense[i] = this_output["popdensemap"][0].cuda()
                        popdense_squared[i] = this_output["popdensemap"][0].to(torch.float32).cuda()**2
                        if "scale" in this_output.keys():
                            if this_output["scale"] is not None:
                                scale[i] = this_output["scale"][0].cuda()
                                scale_squared[i] = this_output["scale"][0].to(torch.float32).cuda()**2
                    
                    output = {
                        "popdensemap": popdense.sum(dim=0, keepdim=True),
                        "popdensemap_squared": popdense_squared.sum(dim=0, keepdim=True)
                    }
                    if "scale" in this_output.keys():
                        if this_output["scale"] is not None:
                            output["scale"] = scale.cuda().sum(dim=0, keepdim=True)
                            output["scale_squared"] = scale_squared.cuda().sum(dim=0, keepdim=True)
                    
                    # save predictions to large map
                    output_map[xl:xl+ips, yl:yl+ips][mask.cpu()] += output["popdensemap"][0][mask].cpu().to(torch.float32)
                    output_map_squared[xl:xl+ips, yl:yl+ips][mask.cpu()] += output["popdensemap_squared"][0][mask].cpu().to(torch.float32)

                    if "scale" in output.keys():
                        if output["scale"] is not None:
                            output_scale_map[xl:xl+ips, yl:yl+ips][mask.cpu()] += output["scale"][0][mask].cpu().to(torch.float32)
                            output_scale_map_squared[xl:xl+ips, yl:yl+ips][mask.cpu()] += output["scale_squared"][0][mask].cpu().to(torch.float32)

                    output_map_count[xl:xl+ips, yl:yl+ips][mask.cpu()] += len(self.model)

                ###### average over the number of times each pixel was visited ######
                print("averaging over the number of times each pixel was visited")
                # mask out values that are not visited of visited exactly once
                div_mask = output_map_count > 1
                # a = output_map.clone()

                a = output_map[div_mask] / output_map_count[div_mask].to(torch.float32)
                output_map[div_mask] = output_map[div_mask] / output_map_count[div_mask].to(torch.float32)

                # calculate the standard deviation from the sum of squares and the mean as "std_dev = math.sqrt((sum_of_squares - n * mean ** 2) / (n - 1))"
                output_map_squared[div_mask] = torch.sqrt((output_map_squared[div_mask] - (output_map[div_mask] ** 2) * output_map_count[div_mask]) / (output_map_count[div_mask] - 1))

                # mask out values that are not visited of visited exactly once
                if "scale" in output.keys():
                    if output["scale"] is not None:
                        output_scale_map[div_mask] = output_scale_map[div_mask] / output_map_count[div_mask]

                        # calculate the standard deviation from the sum of squares and the mean as "std_dev = math.sqrt((sum_of_squares - n * mean ** 2) / (n - 1))"
                        output_scale_map_squared[div_mask] = torch.sqrt((output_scale_map_squared[div_mask] - (output_scale_map[div_mask] ** 2) * output_map_count[div_mask]) / (output_map_count[div_mask] - 1))

                # save maps
                print("saving maps")
                if save:
                    # save the output map
                    testdataloader.dataset.save(output_map, self.experiment_folder)
                    testdataloader.dataset.save(output_map_squared, self.experiment_folder, tag="STD")

                    if "scale" in output.keys():
                        if output["scale"] is not None:
                            testdataloader.dataset.save(output_scale_map, self.experiment_folder, tag="SCALE_{}".format(testdataloader.dataset.region))
                            testdataloader.dataset.save(output_scale_map_squared, self.experiment_folder, tag="SCALE_STD") 
                
                # convert populationmap to census
                gpu_mode = True
                for level in testlevels_eval[testdataloader.dataset.region]:
                    print("-"*50)
                    print("Evaluating level: ", level)
                    # convert map to census
                    details_path = os.path.join(self.experiment_folder, "{}_{}".format(testdataloader.dataset.region, level)) if full else None
                    census_pred, census_gt = testdataloader.dataset.convert_popmap_to_census(output_map, gpu_mode=gpu_mode, level=level, details_to=details_path)
                    this_metrics = get_test_metrics(census_pred, census_gt.float().cuda(), tag="MainCensus_{}_{}".format(testdataloader.dataset.region, level))
                    print(this_metrics)
                    self.target_test_stats = {**self.target_test_stats, **this_metrics}

                    # get the metrics for the clearly built up areas
                    built_up = census_gt>10
                    self.target_test_stats = {**self.target_test_stats,
                                              **get_test_metrics(census_pred[built_up], census_gt[built_up].float().cuda(), tag="MainCensusPos_{}_{}".format(testdataloader.dataset.region, level))}
                    
                    # create scatterplot and upload to wandb
                    if save_scatter:
                        scatterplot = scatter_plot3(census_pred.tolist(), census_gt.tolist(), log_scale=True)
                        if scatterplot is not None:
                            self.target_test_stats["Scatter/Scatter_{}_{}".format(testdataloader.dataset.region, level)] = wandb.Image(scatterplot)
                    
                # adjust map (disaggregate) and recalculate everything
                print("-"*50)
                print("Adjusting map")
                output_map_adj = testdataloader.dataset.adjust_map_to_census(output_map)

                # save adjusted map
                if save:
                    testdataloader.dataset.save(output_map_adj, self.experiment_folder, tag="ADJ_{}".format(testdataloader.dataset.region))

                for level in testlevels_eval[testdataloader.dataset.region]:
                    # convert map to census
                    print("-"*50)
                    print("Evaluating level: ", level)
                    details_path = os.path.join(self.experiment_folder, "{}_{}_adj".format(testdataloader.dataset.region, level)) if full else None
                    census_pred, census_gt = testdataloader.dataset.convert_popmap_to_census(output_map_adj, gpu_mode=gpu_mode, level=level, details_to=details_path)
                    test_stats_adj = get_test_metrics(census_pred, census_gt.float().cuda(), tag="AdjCensus_{}_{}".format(testdataloader.dataset.region, level))
                    print(test_stats_adj)
                    
                    built_up = census_gt>10
                    test_stats_adj = {**test_stats_adj,
                                      **get_test_metrics(census_pred[built_up], census_gt[built_up].float().cuda(), tag="AdjCensusPos_{}_{}".format(testdataloader.dataset.region, level))}
                    
                    # print(test_stats_adj)
                    self.target_test_stats = {**self.target_test_stats,
                                              **test_stats_adj}

                    if save_scatter:
                        # create scatterplot and upload to wandb
                        scatterplot = scatter_plot3(census_pred.tolist(), census_gt.tolist(), log_scale=True)
                        if scatterplot is not None:
                            self.target_test_stats["Scatter/Scatter_{}_{}_adj".format(testdataloader.dataset.region, level)] = wandb.Image(scatterplot)
                    
            
            # save the target test stats
            wandb.log({**{k + '/targettest': v for k, v in self.target_test_stats.items()}, **self.info}, self.info["iter"])

    @staticmethod
    def get_dataloaders(self, args): 
        """
        Get dataloaders for the source and target domains
        Inputs:
            args: command line arguments
            force_recompute: if True, recompute the dataloader's and look out for new files even if the file list already exist
        Outputs:
            dataloaders: dictionary of dataloaders
        """

        input_defs = {'S1': args.Sentinel1, 'S2': args.Sentinel2, 'NIR': args.NIR}

        self.dataset_stats = load_json(os.path.join(config_path, 'dataset_stats', 'my_dataset_stats_unified_2A.json'))
        for mkey in self.dataset_stats.keys():
            if isinstance(self.dataset_stats[mkey], dict):
                for key,val in self.dataset_stats[mkey].items():
                    self.dataset_stats[mkey][key] = torch.tensor(val)
            else:
                self.dataset_stats[mkey] = torch.tensor(val)

        # create the raw source dataset
        need_asc = ["uga"]
        datasets = {
            "test_target": [ Population_Dataset_target(reg, patchsize=ips, overlap=overlap, sentinelbuildings=args.sentinelbuildings, ascfill=reg in need_asc,
                                                       fourseasons=self.args.fourseasons, train_level=lvl, **input_defs)
                                for reg,lvl in zip(args.target_regions, args.train_level) ]
        }
        
        # create the dataloaders
        dataloaders =  {
            "test_target":  [DataLoader(datasets["test_target"], batch_size=1, num_workers=args.num_workers, shuffle=False, drop_last=False)
                                for datasets["test_target"] in datasets["test_target"] ]
        }
        
        return dataloaders


    def resume(self, path, j):
        """
        Input:
            path: path to the checkpoint
        """
        if not os.path.isfile(path):
            raise RuntimeError(f'No checkpoint found at \'{path}\'')

        # load checkpoint
        checkpoint = torch.load(path)
        self.model[j].load_state_dict(checkpoint['model'])
        # self.optimizer.load_state_dict(checkpoint['optimizer'])
        # self.scheduler.load_state_dict(checkpoint['scheduler'])
        self.info["epoch"] = checkpoint['epoch']
        self.info["iter"] = checkpoint['iter']

        print(f'Checkpoint \'{path}\' loaded.')


if __name__ == '__main__':
    args = eval_parser.parse_args()
    print(eval_parser.format_values())

    trainer = Trainer(args)

    since = time.time() 
    trainer.test_target(save=True)
    time_elapsed = time.time() - since
    print('Evaluating completed in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
