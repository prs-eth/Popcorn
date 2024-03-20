 
import torch.nn as nn
import torch
import torch.nn.functional as F

from model.DDA_model.utils.networks import load_checkpoint
from utils.constants import dda_cfg, stage1feats, stage2feats

class POPCORN(nn.Module):
    '''
    POPCORN model
    Description:
        - POPCORN model for population estimation
        - The model is uses a building extractor and an occupancy model
    '''
    def __init__(self, input_channels, feature_extractor="DDA",
                occupancymodel=False, pretrained=False, biasinit=0.75,
                sentinelbuildings=False):
        super(POPCORN, self).__init__()
        """
        Args:
            - input_channels (int): number of input channels
            - feature_dim (int): number of output channels of the feature extractor
            - feature_extractor (str): name of the feature extractor
            - classifier (str): name of the classifier 
            - occupancymodel (bool): whether to use the occupancy model
            - pretrained (bool): whether to use the pretrained feature extractor
            - biasinit (float): initial value of the bias
            - sentinelbuildings (bool): whether to use the sentinel buildings or the load the Google Buildings
        """
 
        self.occupancymodel = occupancymodel 
        self.sentinelbuildings = sentinelbuildings
        self.feature_extractor = feature_extractor  
 
        head_input_dim = 0
        
        # Padding Params
        self.p = 14
        self.p2d = (self.p, self.p, self.p, self.p)

        self.parent = None
        
        self.S1, self.S2 = True, True
        if input_channels==0:
            self.S1, self.S2 = False, False
        elif input_channels==2:
            self.S1, self.S2 = True, False
        elif input_channels==4:
            self.S1, self.S2 = False, True
        
        ## load weights from checkpoint
        self.unetmodel, _, _ = load_checkpoint(epoch=30, cfg=dda_cfg, device="cuda")

        if not pretrained:
            # initialize weights randomly
            for m in self.unetmodel.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                elif isinstance(m, nn.BatchNorm2d):
                    nn.init.constant_(m.weight, 1)
                    nn.init.constant_(m.bias, 0)
                    
        # unet_out = 8*2
        unet_out = self.S1*stage1feats + self.S2*stage1feats
        num_params_sar = sum(p.numel() for p in self.unetmodel.sar_stream.parameters() if p.requires_grad)
        print("trainable DDA SAR: ", num_params_sar)

        num_params_opt = sum(p.numel() for p in self.unetmodel.optical_stream.parameters() if p.requires_grad)
        print("trainable DDA OPT: ", num_params_opt)

        # remove the discriminator from checkpoint, as it is not needed in this version of the code
        self.unetmodel.num_params = sum(p.numel() for p in self.unetmodel.parameters() if p.requires_grad)

        # Build the head
        h = 64
        head_input_dim += unet_out
        self.head = nn.Sequential(
            nn.Conv2d(head_input_dim, h, kernel_size=1, padding=0), nn.ReLU(inplace=True),
            nn.Conv2d(h, h, kernel_size=1, padding=0), nn.ReLU(inplace=True),
            nn.Conv2d(h, h, kernel_size=1, padding=0), nn.ReLU(inplace=True),
            nn.Conv2d(h, 2, kernel_size=1, padding=0)
        )

        # lift the bias of the head to avoid the risk of dying ReLU
        self.head[-1].bias.data = biasinit * torch.ones(2)

        # print size of the embedder and head network
        self.num_params = 0
        if hasattr(self, "embedder"):
            print("Embedder: ",sum(p.numel() for p in self.embedder.parameters() if p.requires_grad)) 
            self.num_params += sum(p.numel() for p in self.embedder.parameters() if p.requires_grad)
        print("Head: ",sum(p.numel() for p in self.head.parameters() if p.requires_grad))
        self.num_params += sum(p.numel() for p in self.head.parameters() if p.requires_grad)
        self.num_params += self.unetmodel.num_params if self.unetmodel is not None else 0

        # define urban extractor, which is again a dual stream unet
        print("Loading urban extractor")
        self.building_extractor, _, _ = load_checkpoint(epoch=30, cfg=dda_cfg, device="cuda")
        self.building_extractor = self.building_extractor.cuda()


    def forward(self, inputs, train=False, padding=True, return_features=True,
                encoder_no_grad=False, unet_no_grad=False, sparse=False):
        """
        Forward pass of the model
        Assumptions:
            - inputs["input"] is the input image (Concatenation of Sentinel-1 and/or Sentinel-2)
            - inputs["input"].shape = [batch_size, input_channels, height, width]
        """

        X = inputs["input"]

        # create building score, if not available in the dataset, or overwrite it if sentinelbuildings is True
        if "building_counts" not in inputs.keys() or self.sentinelbuildings:
            with torch.no_grad():
                inputs["building_counts"]  = self.create_building_score(inputs)
            torch.cuda.empty_cache()
        
        aux = {}
        middlefeatures = []

        # Forward the main model
        if self.unetmodel is not None: 
            X, (px1,px2,py1,py2) = self.add_padding(X, padding) 
            self.unetmodel.freeze_bn_layers()
            if self.S1 and self.S2:
                X = torch.cat([
                    X[:, 4:6], # S1
                    torch.flip(X[:, :3],dims=(1,)), # S2_RGB
                    X[:, 3:4]], # S2_NIR
                dim=1)
            elif self.S1 and not self.S2:
                X = torch.cat([
                    X, # S1
                    torch.zeros(X.shape[0], 4, X.shape[2], X.shape[3], device=X.device)], # S2
                dim=1)
            elif not self.S1 and self.S2:
                X = torch.cat([
                    torch.zeros(X.shape[0], 2, X.shape[2], X.shape[3], device=X.device), # S1
                    torch.flip(X[:, :3],dims=(1,)), # S2_RGB
                    X[:, 3:4]], # S2_NIR
                dim=1)
            
            X = self.unetmodel(X, alpha=0, encoder_no_grad=encoder_no_grad, return_features=True, S1=self.S1, S2=self.S2)

            # revert padding
            X = self.revert_padding(X, (px1,px2,py1,py2))
            middlefeatures.append(X)

        headin = torch.cat(middlefeatures, dim=1)

        # forward the head
        out = self.head(headin)[:,0]

        # Population map and total count
        if self.occupancymodel:

            # activation function for the population map is a ReLU to avoid negative values
            scale = nn.functional.relu(out)

            if "building_counts" in inputs.keys(): 
                
                # save the scale
                aux["scale"] = scale

                # Get the population density map
                popdensemap = scale * inputs["building_counts"][:,0]
            else: 
                raise ValueError("building_counts not in inputs.keys(), but occupancy model is True")
        else:
            popdensemap = nn.functional.relu(out)
            aux["scale"] = None
        
        # aggregate the population counts over the administrative region
        if "admin_mask" in inputs.keys():
            # make the following line work for both 2D and 3D 
            this_mask = inputs["admin_mask"]==inputs["census_idx"].view(-1,1,1)
            popcount = (popdensemap * this_mask).sum((1,2))
            
        else:
            popcount = popdensemap.sum((1,2))

        return {"popcount": popcount, "popdensemap": popdensemap,
                **aux,
                }


    def add_padding(self, data: torch.Tensor, force=True) -> torch.Tensor:
        """
        Description:
            - Add padding to the input data
        Input:
            - data (torch.Tensor): input data
            - force (bool): whether to force the padding
        Output:
            - data (torch.Tensor): padded data
        """
        # Add padding
        px1,px2,py1,py2 = None, None, None, None
        if force:
            data  = nn.functional.pad(data, self.p2d, mode='reflect')
            px1,px2,py1,py2 = self.p, self.p, self.p, self.p
        else:
            # pad to make sure it is divisible by 32
            if (data.shape[2] % 32) != 0:
                px1 = (64 - data.shape[2] % 64) //2
                px2 = (64 - data.shape[2] % 64) - px1
                # data = nn.functional.pad(data, (px1,0,px2,0), mode='reflect') 
                data = nn.functional.pad(data, (0,0,px1,px2,), mode='reflect') 
            if (data.shape[3] % 32) != 0:
                py1 = (64 - data.shape[3] % 64) //2
                py2 = (64 - data.shape[3] % 64) - py1
                data = nn.functional.pad(data, (py1,py2,0,0), mode='reflect')

        return data, (px1,px2,py1,py2)
    

    def revert_padding(self, data: torch.tensor, padding: tuple) -> torch.Tensor:
        """
        Description:
            - Revert the padding of the input data
        Input:
            - data (torch.Tensor): input data
            - padding (tuple): padding parameters
        Output:
            - data (torch.Tensor): padded data
        """
        px1,px2,py1,py2 = padding
        if px1 is not None or px2 is not None:
            data = data[:,:,px1:-px2,:]
        if py1 is not None or py2 is not None:
            data = data[:,:,:,py1:-py2]
        return data


    def create_building_score(self, inputs: dict) -> torch.Tensor:
        """
        input:
            - inputs: dictionary with the input data
        output:
            - score: building score
        """

        # initialize the neural network, load from checkpoint
        self.building_extractor.eval()
        self.unetmodel.freeze_bn_layers()
 
        # add padding
        X, (px1,px2,py1,py2) = self.add_padding(inputs["input"], True)

        # forward the neural network
        if self.S1 and self.S2:
            X = torch.cat([
                X[:, 4:6], # S1
                torch.flip(X[:, :3],dims=(1,)), # S2_RGB
                X[:, 3:4]], # S2_NIR
            dim=1)
            _, _, logits, _, _ = self.building_extractor(X, alpha=0, return_features=False, S1=self.S1, S2=self.S2)
        elif self.S1 and not self.S2:
            X = torch.cat([
                X, # S1
                torch.zeros(X.shape[0], 4, X.shape[2], X.shape[3], device=X.device)], # S2
            dim=1)
            logits = self.building_extractor(X, alpha=0, return_features=False, S1=self.S1, S2=self.S2)
        elif not self.S1 and self.S2:
            X = torch.cat([
                torch.zeros(X.shape[0], 2, X.shape[2], X.shape[3], device=X.device), # S1
                torch.flip(X[:, :3],dims=(1,)), # S2_RGB
                X[:, 3:4]], # S2_NIR
            dim=1)
            logits = self.building_extractor(X, alpha=0, return_features=False, S1=self.S1, S2=self.S2)
            
        # forward the model
        score = torch.sigmoid(logits)

        # revert padding
        score = self.revert_padding(score, (px1,px2,py1,py2))

        return score
    