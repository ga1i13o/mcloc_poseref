import logging
import sys
from dataclasses import dataclass
import torch
from torch import nn
import os
from torchvision.models import resnet18, resnet50

from gloc.models.layers import L2Norm, FlattenFeatureMaps
from gloc.models import features


def get_retrieval_model(model_name, cuda=True, eval=True):
    if model_name.startswith('cosplace'):
        model = torch.hub.load("gmberton/cosplace", "get_trained_model", 
                    backbone="ResNet18", fc_output_dim=512)    
    else:
        raise NotImplementedError()
    
    if cuda:
        model = model.cuda()
    if eval:
        model = model.eval()

    return model


def get_feature_model(args, model_name, cuda=True):
    if model_name.startswith('cosplace'):
        feat_model = features.CosplaceFeatures(model_name)
    
    elif model_name.startswith('resnet'):
        feat_model = features.ResnetFeatures(model_name)

    elif model_name.startswith('alexnet'):
        feat_model = features.AlexnetFeatures(model_name)

    elif model_name == 'Dinov2':
        conf = DinoConf(clamp=args.clamp_score, level=args.feat_level)
        feat_model = features.DinoFeatures(conf)
    
    elif model_name == 'Roma':
        conf = RomaConf(clamp=args.clamp_score, level=args.feat_level, scale_n=args.scale_fmaps)
        feat_model = features.RomaFeatures(conf)

    else:
        raise NotImplementedError()

    if cuda:
        feat_model = feat_model.cuda()

    return feat_model


def get_ref_model(args, cuda=True):
    model_name = args.ref_model
    feat_model = get_feature_model(args, args.feat_model, cuda)

    if model_name == 'DenseFeatures':
        from gloc.models.refinement_model import DenseFeaturesRefiner
        model_class = DenseFeaturesRefiner
        conf = DenseFeaturesConf(clamp=args.clamp_score)
        
    else:
        raise NotImplementedError()

    model = model_class(conf, feat_model)
    if cuda:
        model = model.cuda()

    return model


@dataclass
class DenseFeaturesConf:
    clamp: float = -1
    def get_str__conf(self):
        repr = f"_cl{self.clamp}"
        return repr


@dataclass
class DinoConf:
    clamp: float = -1
    level: int = 8
    def get_str__conf(self):
        repr = f"_l{self.level}_cl{self.clamp}"
        return repr

@dataclass
class RomaConf:
    clamp: float = -1
    level: int = 4 # 1 2 4 8
    # pool feature maps to 1/n
    scale_n: int = -1
    def get_str__conf(self):
        repr = f"_l{self.level}_sn{self.scale_n}_cl{self.clamp}"
        return repr
