"""
Code is adapted from https://github.com/SebastianHafner/DDA_UrbanExtraction
Modified: Arno Rüegg, Nando Metzger
"""
import torch
from torch.utils import data as torch_data
import numpy as np
import wandb
from tqdm import tqdm
from ..utils import datasets, metrics

def disc_evaluation(net, cfg, device, run_type: str, epoch: float, step: int, max_samples: int = None):
    net.to(device)
    net.eval()

    dataset = datasets.UrbanExtractionDataset(cfg=cfg, dataset=run_type, no_augmentations=True, include_unlabeled=False)

    # reset the generators
    num_workers = 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER
    dataloader = torch_data.DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=True, drop_last=True)

    stop_step = len(dataloader) if max_samples is None else max_samples

    correct_sar = 0
    total_sar = 0
    correct_opt = 0
    total_opt = 0
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step == stop_step:
                break

            imgs = batch['x'].to(device)
            y_true = batch['y'].to(device)

            logits_SAR, logits_OPT, logits_fusion, disc_logits_sar, disc_logits_optical = net(imgs)

            dis_GT_sar = torch.ones(([disc_logits_sar.shape[0], disc_logits_sar.shape[2], disc_logits_sar.shape[3]])).cuda()
            #dis_GT_sar = torch.nn.functional.one_hot(dis_GT_sar.to(torch.int64)).view(-1,2)

            dis_GT_optical = torch.ones(([disc_logits_optical.shape[0], disc_logits_optical.shape[2], disc_logits_optical.shape[3]])).cuda()
            #dis_GT_optical = torch.nn.functional.one_hot(dis_GT_optical.to(torch.int64),2).view(-1,2)

            y_pred_sar = torch.argmax(disc_logits_sar,1)# > 0.5
            y_pred_optical = torch.argmax(disc_logits_optical,1) #> 0.5

            correct_sar += (y_pred_sar == dis_GT_sar).float().sum()
            total_sar += y_pred_sar.shape[0]*y_pred_sar.shape[1]*y_pred_sar.shape[2]
            correct_opt += (y_pred_optical == dis_GT_optical).float().sum()
            total_opt += y_pred_optical.shape[0] * y_pred_optical.shape[1] * y_pred_optical.shape[2]

            if cfg.DEBUG:
                break

    print("Acc of Discriminator for sar: ", (correct_sar/total_sar).item())
    print("Acc of Discriminator for optical: ", (correct_opt/total_opt).item())


    wandb.log({f'{run_type} Discriminator Accuracy sar': correct_sar/total_opt,
                f'{run_type} Discriminator Accuracy optical': correct_opt/total_opt,
               'step': step, 'epoch': epoch,
               })

def model_evaluation(net, cfg, device, run_type: str, epoch: float, step: int, max_samples: int = None):
    net.to(device)
    net.eval()

    thresholds = torch.linspace(0.5, 1, 1).to(device)
    measurer_fusion = metrics.MultiThresholdMetric(thresholds)
    measurer_SAR = metrics.MultiThresholdMetric(thresholds)
    measurer_OPT = metrics.MultiThresholdMetric(thresholds)

    dataset = datasets.UrbanExtractionDataset(cfg=cfg, dataset=run_type, no_augmentations=True, include_unlabeled=False)

    # reset the generators
    num_workers = 0 if cfg.DEBUG else cfg.DATALOADER.NUM_WORKER
    dataloader = torch_data.DataLoader(dataset, batch_size=1, num_workers=num_workers, shuffle=True, drop_last=True)

    stop_step = len(dataloader) if max_samples is None else max_samples
    
    #lists of metric values
    boundary_IoU_fusion, hausdorff_fusion, closed_IoU_fusion, opened_IoU_fusion, gradient_IoU_fusion, ssim_fusion = [],[],[],[],[],[]
    boundary_IoU_sar, hausdorff_sar, closed_IoU_sar, opened_IoU_sar, gradient_IoU_sar, ssim_sar = [],[],[],[],[],[]
    boundary_IoU_optical, hausdorff_optical, closed_IoU_optical, opened_IoU_optical, gradient_IoU_optical, ssim_optical = [],[],[],[],[],[]
    
    with torch.no_grad():
        for step, batch in enumerate(dataloader):
            if step == stop_step:
                break

            imgs = batch['x'].to(device)
            y_true = batch['y'].to(device)

            logits_SAR, logits_OPT, logits_fusion, logits_disc_sar, logits_disc_optical = net(imgs)

            y_pred_fusion = torch.sigmoid(logits_fusion) #> 0.5
            y_pred_SAR = torch.sigmoid(logits_SAR) #> 0.5
            y_pred_OPT = torch.sigmoid(logits_OPT) #> 0.5

            y_true = y_true.detach() #> 0.5
            y_pred_fusion = y_pred_fusion.detach()
            y_pred_SAR = y_pred_SAR.detach()
            y_pred_OPT = y_pred_OPT.detach()

            boundary_IoU_fusion.append(metrics.boundary_IoU(y_true, y_pred_fusion))
            #hausdorff_fusion.append(metrics.hausdorff(y_true, y_pred_fusion))
            closed_IoU_fusion.append(metrics.closed_IoU(y_true, y_pred_fusion))
            opened_IoU_fusion.append(metrics.opened_IoU(y_true, y_pred_fusion))
            gradient_IoU_fusion.append(metrics.gradient_IoU(y_true, y_pred_fusion))
            ssim_fusion.append(metrics.ssim(y_true, y_pred_fusion))

            boundary_IoU_sar.append(metrics.boundary_IoU(y_true, y_pred_SAR))
            #hausdorff_sar.append(metrics.hausdorff(y_true, y_pred_SAR))
            closed_IoU_sar.append(metrics.closed_IoU(y_true, y_pred_SAR))
            opened_IoU_sar.append(metrics.opened_IoU(y_true, y_pred_SAR))
            gradient_IoU_sar.append(metrics.gradient_IoU(y_true, y_pred_SAR))
            ssim_sar.append(metrics.ssim(y_true, y_pred_SAR))

            boundary_IoU_optical.append(metrics.boundary_IoU(y_true, y_pred_OPT))
            #hausdorff_optical.append(metrics.hausdorff(y_true, y_pred_OPT))
            closed_IoU_optical.append(metrics.closed_IoU(y_true, y_pred_OPT))
            opened_IoU_optical.append(metrics.opened_IoU(y_true, y_pred_OPT))
            gradient_IoU_optical.append(metrics.gradient_IoU(y_true, y_pred_OPT))
            ssim_optical.append(metrics.ssim(y_true, y_pred_OPT))

            measurer_fusion.add_sample(y_true, y_pred_fusion)
            measurer_SAR.add_sample(y_true, y_pred_SAR)
            measurer_OPT.add_sample(y_true, y_pred_OPT)

            if cfg.DEBUG:
                break

    #print(f'Computing {run_type} F1 score for both images: ', end=' ', flush=True)
    f1s_fusion = measurer_fusion.compute_f1()
    #print(f'Computing {run_type} F1 score for SAR only: ', end=' ', flush=True)
    f1s_SAR = measurer_SAR.compute_f1()
    #print(f'Computing {run_type} F1 score for optical only: ', end=' ', flush=True)
    f1s_OPT = measurer_OPT.compute_f1()

    precisions_fusion, recalls_fusion, IoU_fusion = measurer_fusion.precision, measurer_fusion.recall, measurer_fusion.IoU
    precisions_SAR, recalls_SAR, IoU_SAR = measurer_SAR.precision, measurer_SAR.recall, measurer_SAR.IoU
    precisions_OPT, recalls_OPT, IoU_OPT = measurer_OPT.precision, measurer_OPT.recall, measurer_OPT.IoU

    # best f1 score for passed thresholds
    f1_fusion = f1s_fusion.max()
    f1_SAR = f1s_SAR.max()
    f1_OPT = f1s_OPT.max()
    argmax_f1_fusion = f1s_fusion.argmax()
    argmax_f1_SAR = f1s_SAR.argmax()
    argmax_f1_OPT = f1s_OPT.argmax()

    best_thresh_fusion = thresholds[argmax_f1_fusion]
    best_thresh_SAR = thresholds[argmax_f1_SAR]
    best_thresh_OPT = thresholds[argmax_f1_OPT]

    precision_fusion = precisions_fusion[argmax_f1_fusion]
    precision_SAR = precisions_SAR[argmax_f1_SAR]
    precision_OPT = precisions_OPT[argmax_f1_OPT]

    recall_fusion = recalls_fusion[argmax_f1_fusion]
    recall_SAR = recalls_SAR[argmax_f1_SAR]
    recall_OPT = recalls_OPT[argmax_f1_OPT]

    IoU_fusion = IoU_fusion[argmax_f1_fusion]
    IoU_SAR = IoU_SAR[argmax_f1_SAR]
    IoU_OPT = IoU_OPT[argmax_f1_OPT]

    boundary_IoU_fusion = torch.mean(torch.stack(boundary_IoU_fusion))
    #hausdorff_fusion = torch.mean(torch.stack(hausdorff_fusion))
    closed_IoU_fusion = torch.mean(torch.stack(closed_IoU_fusion))
    opened_IoU_fusion = torch.mean(torch.stack(opened_IoU_fusion))
    gradient_IoU_fusion = torch.mean(torch.stack(gradient_IoU_fusion))
    ssim_fusion = torch.mean(torch.stack(ssim_fusion))

    boundary_IoU_sar = torch.mean(torch.stack(boundary_IoU_sar))
    #hausdorff_sar = torch.mean(torch.stack(hausdorff_sar))
    closed_IoU_sar = torch.mean(torch.stack(closed_IoU_sar))
    opened_IoU_sar = torch.mean(torch.stack(opened_IoU_sar))
    gradient_IoU_sar = torch.mean(torch.stack(gradient_IoU_sar))
    ssim_sar = torch.mean(torch.stack(ssim_sar))

    boundary_IoU_optical = torch.mean(torch.stack(boundary_IoU_optical))
    #hausdorff_optical = torch.mean(torch.stack(hausdorff_optical))
    closed_IoU_optical = torch.mean(torch.stack(closed_IoU_optical))
    opened_IoU_optical = torch.mean(torch.stack(opened_IoU_optical))
    gradient_IoU_optical = torch.mean(torch.stack(gradient_IoU_optical))
    ssim_optical = torch.mean(torch.stack(ssim_optical))

    print("F1 Score (fusion)",f'{f1_fusion.item():.3f}', flush=True)
    print("F1 Score (SAR)",f'{f1_SAR.item():.3f}', flush=True)
    print("F1 Score (optical)",f'{f1_OPT.item():.3f}', flush=True)

    wandb.log({f'{run_type} F1 fusion': f1_fusion,
               f'{run_type} threshold fusion': best_thresh_fusion,
               f'{run_type} precision fusion': precision_fusion,
               f'{run_type} recall fusion': recall_fusion,
               f'{run_type} IoU fusion': IoU_fusion,
               f'{run_type} boundary IoU fusion': boundary_IoU_fusion,
               f'{run_type} closed IoU fusion': closed_IoU_fusion,
               f'{run_type} opened IoU fusion': opened_IoU_fusion,
               f'{run_type} gradient IoU fusion': gradient_IoU_fusion,
               f'{run_type} SSIM fusion': ssim_fusion,
               f'{run_type} F1 SAR': f1_SAR,
               f'{run_type} threshold SAR': best_thresh_SAR,
               f'{run_type} precision SAR': precision_SAR,
               f'{run_type} recall SAR': recall_SAR,
               f'{run_type} IoU SAR': IoU_SAR,
               f'{run_type} boundary IoU SAR': boundary_IoU_sar,
               f'{run_type} closed IoU SAR': closed_IoU_sar,
               f'{run_type} opened IoU SAR': opened_IoU_sar,
               f'{run_type} gradient IoU SAR': gradient_IoU_sar,
               f'{run_type} SSIM SAR': ssim_sar,
               f'{run_type} F1 optical': f1_OPT,
               f'{run_type} threshold optical': best_thresh_OPT,
               f'{run_type} precision optical': precision_OPT,
               f'{run_type} recall optical': recall_OPT,
               f'{run_type} IoU optical': IoU_OPT,
               f'{run_type} boundary IoU optical': boundary_IoU_optical,
               f'{run_type} closed IoU optical': closed_IoU_optical,
               f'{run_type} opened IoU optical': opened_IoU_optical,
               f'{run_type} gradient IoU optical': gradient_IoU_optical,
               f'{run_type} SSIM optical': ssim_optical,
               'step': step, 'epoch': epoch,
               })

def get_rgb(x):

    quantile = 95

    rgb = np.flip(x[2:5].permute(1,2,0).cpu().numpy(),axis=2)

    maxi = np.percentile(rgb[:,:,0].flatten(),quantile)
    mini = np.percentile(rgb[:,:,0].flatten(),100-quantile)
    rgb[:,:,0] = np.where(rgb[:,:,0] > maxi, maxi, rgb[:,:,0])
    rgb[:,:,0] = np.where(rgb[:,:,0] < mini, mini, rgb[:,:,0])
    rgb[:,:,0] = (rgb[:,:,0]-mini)/(maxi-mini)

    maxi = np.percentile(rgb[:,:,1].flatten(),quantile)
    mini = np.percentile(rgb[:,:,1].flatten(),100-quantile)
    rgb[:,:,1] = np.where(rgb[:,:,1] > maxi, maxi, rgb[:,:,1])
    rgb[:,:,1] = np.where(rgb[:,:,1] < mini, mini, rgb[:,:,1])
    rgb[:,:,1] = (rgb[:,:,1]-mini)/(maxi-mini)

    maxi = np.percentile(rgb[:,:,2].flatten(),quantile)
    mini = np.percentile(rgb[:,:,2].flatten(),100-quantile)
    rgb[:,:,2] = np.where(rgb[:,:,2] > maxi, maxi, rgb[:,:,2])
    rgb[:,:,2] = np.where(rgb[:,:,2] < mini, mini, rgb[:,:,2])
    rgb[:,:,2] = (rgb[:,:,2]-mini)/(maxi-mini)

    return rgb

def model_testing(net, cfg, device, step, epoch):
    net.to(device)
    net.eval()

    dataset = datasets.SpaceNet7Dataset(cfg)

    y_true_dict = {'test': []}
    y_pred_fusion_dict = {'test': []}
    y_pred_SAR_dict = {'test': []}
    y_pred_OPT_dict = {'test': []}
    boundary_IoU_fusion, hausdorff_fusion, closed_IoU_fusion, opened_IoU_fusion, gradient_IoU_fusion, ssim_fusion = [],[],[],[],[],[]
    boundary_IoU_sar, hausdorff_sar, closed_IoU_sar, opened_IoU_sar, gradient_IoU_sar, ssim_sar = [],[],[],[],[],[]
    boundary_IoU_optical, hausdorff_optical, closed_IoU_optical, opened_IoU_optical, gradient_IoU_optical, ssim_optical = [],[],[],[],[],[]

    for index in range(len(dataset)):
        sample = dataset.__getitem__(index)

        with torch.no_grad():
            x = sample['x'].to(device)
            y_true = sample['y'].to(device)
            y_true = y_true[None, :]

            logits_SAR, logits_OPT, logits_fusion, logits_disc_sar, logits_disc_optical = net(x.unsqueeze(0))

            y_pred_fusion = torch.sigmoid(logits_fusion) #> 0.5
            y_pred_SAR = torch.sigmoid(logits_SAR) #> 0.5
            y_pred_OPT = torch.sigmoid(logits_OPT) #> 0.5

            boundary_IoU_fusion.append(metrics.boundary_IoU(y_true, y_pred_fusion).item())
            #hausdorff_fusion.append(metrics.hausdorff(y_true, y_pred_fusion))
            closed_IoU_fusion.append(metrics.closed_IoU(y_true, y_pred_fusion).item())
            opened_IoU_fusion.append(metrics.opened_IoU(y_true, y_pred_fusion).item())
            gradient_IoU_fusion.append(metrics.gradient_IoU(y_true, y_pred_fusion).item())
            ssim_fusion.append(metrics.ssim(y_true, y_pred_fusion).item())

            boundary_IoU_sar.append(metrics.boundary_IoU(y_true, y_pred_SAR).item())
            #hausdorff_sar.append(metrics.hausdorff(y_true, y_pred_SAR))
            closed_IoU_sar.append(metrics.closed_IoU(y_true, y_pred_SAR).item())
            opened_IoU_sar.append(metrics.opened_IoU(y_true, y_pred_SAR).item())
            gradient_IoU_sar.append(metrics.gradient_IoU(y_true, y_pred_SAR).item())
            ssim_sar.append(metrics.ssim(y_true, y_pred_SAR).item())

            boundary_IoU_optical.append(metrics.boundary_IoU(y_true, y_pred_OPT).item())
            #hausdorff_optical.append(metrics.hausdorff(y_true, y_pred_OPT))
            closed_IoU_optical.append(metrics.closed_IoU(y_true, y_pred_OPT).item())
            opened_IoU_optical.append(metrics.opened_IoU(y_true, y_pred_OPT).item())
            gradient_IoU_optical.append(metrics.gradient_IoU(y_true, y_pred_OPT).item())
            ssim_optical.append(metrics.ssim(y_true, y_pred_OPT).item())

            y_true = y_true.detach().cpu().flatten().numpy()
            y_pred_fusion = y_pred_fusion.detach().cpu().flatten().numpy()
            y_pred_SAR = y_pred_SAR.detach().cpu().flatten().numpy()
            y_pred_OPT = y_pred_OPT.detach().cpu().flatten().numpy()
            
            """region = sample['region']
            if region not in y_true_dict.keys():
                y_true_dict[region] = [y_true]
                y_pred_fusion_dict[region] = [y_pred_fusion]
                y_pred_SAR_dict[region] = [y_pred_SAR]
                y_pred_OPT_dict[region] = [y_pred_OPT]

            else:
                y_true_dict[region].append(y_true)
                y_pred_fusion_dict[region].append(y_pred_fusion)
                y_pred_SAR_dict[region].append(y_pred_SAR)
                y_pred_OPT_dict[region].append(y_pred_OPT)"""

            y_true_dict['test'].append(y_true)
            y_pred_fusion_dict['test'].append(y_pred_fusion)
            y_pred_SAR_dict['test'].append(y_pred_SAR)
            y_pred_OPT_dict['test'].append(y_pred_OPT)

    boundary_IoU_fusion = np.mean(boundary_IoU_fusion)
    #hausdorff_fusion = torch.mean(torch.stack(hausdorff_fusion))
    closed_IoU_fusion = np.mean(closed_IoU_fusion)
    opened_IoU_fusion = np.mean(opened_IoU_fusion)
    gradient_IoU_fusion = np.mean(gradient_IoU_fusion)
    ssim_fusion = np.mean(ssim_fusion)

    boundary_IoU_sar = np.mean(boundary_IoU_sar)
    #hausdorff_sar = torch.mean(torch.stack(hausdorff_sar))
    closed_IoU_sar = np.mean(closed_IoU_sar)
    opened_IoU_sar = np.mean(opened_IoU_sar)
    gradient_IoU_sar = np.mean(gradient_IoU_sar)
    ssim_sar = np.mean(ssim_sar)

    boundary_IoU_optical = np.mean(boundary_IoU_optical)
    #hausdorff_optical = torch.mean(torch.stack(hausdorff_optical))
    closed_IoU_optical = np.mean(closed_IoU_optical)
    opened_IoU_optical = np.mean(opened_IoU_optical)
    gradient_IoU_optical = np.mean(gradient_IoU_optical)
    ssim_optical = np.mean(ssim_optical)

    # add always the same image at testtime and add it to wandb
    dataset = datasets.UrbanExtractionDataset(cfg=cfg, dataset='training')
    sample = dataset.__getitem__(0, aug=False) # random index

    with torch.no_grad():
        x = sample['x'].to(device)
        train_shape = x.shape[1:]
        y_true = sample['y'].to(device)
        logits_SAR, logits_OPT, logits_fusion, logits_disc_sar, logits_disc_optical = net(x.unsqueeze(0))

        y_pred_fusion = torch.sigmoid(logits_fusion) > 0.5
        y_pred_SAR = torch.sigmoid(logits_SAR) > 0.5
        y_pred_OPT = torch.sigmoid(logits_OPT) > 0.5

        rgb_train = get_rgb(x)
        
        y_true_train = y_true.detach().cpu().flatten().numpy().reshape(train_shape)
        y_pred_fusion_train = y_pred_fusion.detach().cpu().flatten().numpy().reshape(train_shape)
        y_pred_SAR_train = y_pred_SAR.detach().cpu().flatten().numpy().reshape(train_shape)
        y_pred_OPT_train = y_pred_OPT.detach().cpu().flatten().numpy().reshape(train_shape)

    dataset = datasets.SpaceNet7Dataset(cfg)
    sample = dataset.__getitem__(0) # random index

    with torch.no_grad():
        x = sample['x'].to(device)[:,:391,:391]
        test_shape = x.shape[1:]
        y_true = sample['y'].to(device)[:,:391,:391]
        logits_SAR, logits_OPT, logits_fusion, logits_disc_sar, logits_disc_optical = net(x.unsqueeze(0))

        y_pred_fusion = torch.sigmoid(logits_fusion) > 0.5
        y_pred_SAR = torch.sigmoid(logits_SAR) > 0.5
        y_pred_OPT = torch.sigmoid(logits_OPT) > 0.5

        rgb_test = get_rgb(x)

        y_true_test = y_true.detach().cpu().flatten().numpy().reshape(test_shape)
        y_pred_fusion_test = y_pred_fusion.detach().cpu().flatten().numpy().reshape(test_shape)
        y_pred_SAR_test = y_pred_SAR.detach().cpu().flatten().numpy().reshape(test_shape)
        y_pred_OPT_test = y_pred_OPT.detach().cpu().flatten().numpy().reshape(test_shape)

    y_true_train = wandb.Image(y_true_train, caption= "GT")
    y_pred_fusion_train = wandb.Image(y_pred_fusion_train, caption= "Pred Fusion")
    y_pred_SAR_train = wandb.Image(y_pred_SAR_train, caption= "Pred SAR")
    y_pred_OPT_train = wandb.Image(y_pred_OPT_train, caption= "Pred OPT")
    Train_rgb = wandb.Image(rgb_train, caption="Train RGB", mode="RGB") 

    y_true_test = wandb.Image(y_true_test, caption= "GT")
    y_pred_fusion_test = wandb.Image(y_pred_fusion_test, caption= "Pred Fusion")
    y_pred_SAR_test = wandb.Image(y_pred_SAR_test, caption= "Pred SAR")
    y_pred_OPT_test = wandb.Image(y_pred_OPT_test, caption= "Pred OPT")
    Test_rgb = wandb.Image(rgb_test, caption="Test RGB", mode="RGB") 

    wandb.log({"Output Test": [Test_rgb, y_true_test, y_pred_fusion_test, y_pred_SAR_test, y_pred_OPT_test],
                "Output Train": [Train_rgb, y_true_train, y_pred_fusion_train, y_pred_SAR_train, y_pred_OPT_train]})


    def evaluate_region(region_name: str):
        y_true_region = torch.Tensor(np.concatenate(y_true_dict[region_name]))
        y_pred_fusion_region = torch.Tensor(np.concatenate(y_pred_fusion_dict[region_name]))
        y_pred_SAR_region = torch.Tensor(np.concatenate(y_pred_SAR_dict[region_name]))
        y_pred_OPT_region = torch.Tensor(np.concatenate(y_pred_OPT_dict[region_name]))

        prec_fusion = metrics.precision(y_true_region, y_pred_fusion_region, dim=0).item()
        rec_fusion = metrics.recall(y_true_region, y_pred_fusion_region, dim=0).item()
        f1_fusion = metrics.f1_score(y_true_region, y_pred_fusion_region, dim=0).item()
        IoU_fusion = metrics.IoU(y_true_region, y_pred_fusion_region, dim=0).item()
        prec_SAR = metrics.precision(y_true_region, y_pred_SAR_region, dim=0).item()
        rec_SAR = metrics.recall(y_true_region, y_pred_SAR_region, dim=0).item()
        f1_SAR = metrics.f1_score(y_true_region, y_pred_SAR_region, dim=0).item()
        IoU_SAR = metrics.IoU(y_true_region, y_pred_SAR_region, dim=0).item()
        prec_OPT = metrics.precision(y_true_region, y_pred_OPT_region, dim=0).item()
        rec_OPT = metrics.recall(y_true_region, y_pred_OPT_region, dim=0).item()
        f1_OPT = metrics.f1_score(y_true_region, y_pred_OPT_region, dim=0).item()
        IoU_OPT = metrics.IoU(y_true_region, y_pred_OPT_region, dim=0).item()

        wandb.log({f'{region_name} F1 fusion': f1_fusion,
                   f'{region_name} precision fusion': prec_fusion,
                   f'{region_name} recall fusion': rec_fusion,
                   f'{region_name} IoU fusion': IoU_fusion,
                   f'{region_name} boundary IoU fusion': boundary_IoU_fusion,
                   f'{region_name} closed IoU fusion': closed_IoU_fusion,
                   f'{region_name} opened IoU fusion': opened_IoU_fusion,
                   f'{region_name} gradient IoU fusion': gradient_IoU_fusion,
                   f'{region_name} SSIM fusion': ssim_fusion,
                   f'{region_name} F1 SAR': f1_SAR,
                   f'{region_name} precision SAR': prec_SAR,
                   f'{region_name} recall SAR': rec_SAR,
                   f'{region_name} IoU SAR': IoU_SAR,
                   f'{region_name} boundary IoU SAR': boundary_IoU_sar,
                   f'{region_name} closed IoU SAR': closed_IoU_sar,
                   f'{region_name} opened IoU SAR': opened_IoU_sar,
                   f'{region_name} gradient IoU SAR': gradient_IoU_sar,
                   f'{region_name} SSIM SAR': ssim_sar,
                   f'{region_name} F1 optical': f1_OPT,
                   f'{region_name} precision optical': prec_OPT,
                   f'{region_name} recall optical': rec_OPT,
                   f'{region_name} IoU optical': IoU_OPT,
                   f'{region_name} boundary IoU optical': boundary_IoU_optical,
                   f'{region_name} closed IoU optical': closed_IoU_optical,
                   f'{region_name} opened IoU optical': opened_IoU_optical,
                   f'{region_name} gradient IoU optical': gradient_IoU_optical,
                   f'{region_name} SSIM optical': ssim_optical,
                   'step': step, 'epoch': epoch,
                   })

    """for region in dataset.regions['regions'].values():
        evaluate_region(region)"""
    evaluate_region('test')


