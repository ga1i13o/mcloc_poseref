import sys
from pathlib import Path
from collections import OrderedDict
import torch
from torch import nn
from torchvision.models import resnet18, resnet50

from gloc.models.layers import L2Norm


class BaseFeaturesClass(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Identity()

    def forward(self, x):
        """
        To be used by subclasses, each specifying their own `self.model`
        Args:
            x (torch.tensor): batch of images shape Bx3xHxW
        Returns:
            torch.tensor: Features maps of shape BxDxHxW
        """
        return self.model(x)


class CosplaceFeatures(BaseFeaturesClass):
    def __init__(self, model_name):
        super().__init__()
        if 'r18' in model_name:
            arch = 'ResNet18'
        else: # 'r50' in model_name
            arch = 'ResNet50'
        # FC dim set to 512 as a placeholder, it will be truncated anyway before the last FC
        model = torch.hub.load("gmberton/cosplace", "get_trained_model", 
                    backbone=arch, fc_output_dim=512)
                
        backbone = model.backbone    
        if '_l1' in model_name:
            backbone = backbone[:-3]
        elif '_l2' in model_name:
            backbone = backbone[:-2]
        elif '_l3' in model_name:
            backbone = backbone[:-1]
        
        self.model = backbone.eval()
        

class ResnetFeatures(BaseFeaturesClass):
    def __init__(self, model_name):
        super().__init__()        
        
        if model_name.startswith('resnet18'):
            model = resnet18(weights='DEFAULT')
        elif model_name.startswith('resnet50'):
            model = resnet50(weights='DEFAULT')
        else:
            raise NotImplementedError
        
        layers = list(model.children())[:-2]  # Remove avg pooling and FC layer
        backbone = torch.nn.Sequential(*layers)

        if '_l1' in model_name:
            backbone = backbone[:-3]
        elif '_l2' in model_name:
            backbone = backbone[:-2]
        elif '_l3' in model_name:
            backbone = backbone[:-1]
        
        self.model = backbone.eval()
        

class AlexnetFeatures(BaseFeaturesClass):
    def __init__(self, model_name):
        super().__init__()        
        
        model = torch.hub.load('pytorch/vision:v0.10.0', 'alexnet', pretrained=True)
        backbone = model.features
        
        if '_l1' in model_name:
            backbone = backbone[:4]
        elif '_l2' in model_name:
            backbone = backbone[:7]
        elif '_l3' in model_name:
            backbone = backbone[:9]
        
        self.model = backbone.eval()


class DinoFeatures(BaseFeaturesClass):
    def __init__(self, conf):
        super().__init__() 
        self.conf = conf
        self.clamp  = conf.clamp
        self.norm = L2Norm()
        self.conf.bs = 32
        self.feat_level = conf.level[0]

        dinov2_vits14 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14')
        self.ref_model = dinov2_vits14

    # override
    def forward(self, x):
        desc = self.ref_model.get_intermediate_layers(x, n=self.feat_level, reshape=True)[-1]
        desc = self.norm(desc)

        return desc


class RomaFeatures(BaseFeaturesClass):
    weight_urls = {
        "roma": {
            "outdoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_outdoor.pth",
            "indoor": "https://github.com/Parskatt/storage/releases/download/roma/roma_indoor.pth",
        },
        "dinov2": "https://dl.fbaipublicfiles.com/dinov2/dinov2_vitl14/dinov2_vitl14_pretrain.pth", #hopefully this doesnt change :D
    }

    def __init__(self, conf):
        super().__init__() 
        sys.path.append(str(Path(__file__).parent.joinpath('third_party/RoMa')))
        from roma.models.encoders import CNNandDinov2

        self.conf = conf
        weights = torch.hub.load_state_dict_from_url(self.weight_urls["roma"]["outdoor"])
        dinov2_weights = torch.hub.load_state_dict_from_url(self.weight_urls["dinov2"])

        ww = OrderedDict({k.replace('encoder.', ''): v for (k, v) in weights.items() if k.startswith('encoder')  })
        encoder = CNNandDinov2(
            cnn_kwargs = dict(
                pretrained=False,
                amp = True),
            amp = True,
            use_vgg = True,
            dinov2_weights = dinov2_weights
        )
        encoder.load_state_dict(ww)
        
        self.ref_model = encoder.cnn
        self.clamp  = conf.clamp
        self.scale_n = conf.scale_n
        self.norm = L2Norm()
        self.conf.bs = 16
        self.feat_level = conf.level[0]

    # override
    def forward(self, x):
        f_pyramid = self.ref_model(x)
        fmaps = f_pyramid[self.feat_level]

        if self.scale_n != -1:
            # optionally scale down fmaps
            nh, nw = tuple(fmaps.shape[-2:])
            half = nn.AdaptiveAvgPool2d((nh // self.scale_n, nw // self.scale_n))
            fmaps = half(fmaps)

        desc = self.norm(fmaps)
        return desc
