from dataclasses import dataclass
import torch

from gloc.resamplers.samplers import (RandomGaussianSampler, RandomSamplerByAxis, 
                                      RandomDoubleAxisSampler, RandomAndDoubleAxisSampler)
from gloc.resamplers.scalers import ConstantScaler, UniformScaler


def get_sampler(args, sampler_name, scaler_name):    
    max_angle_delta = torch.tensor(args.teta)
    max_center_std = torch.tensor(args.center_std)

    sampler_conf = SamplerConf()
    if sampler_name == 'rand':
        sampler_class = RandomGaussianSampler
    elif sampler_name == 'rand_yaw_or_pitch':
        sampler_class = RandomSamplerByAxis
    elif sampler_name == 'rand_yaw_and_pitch':
        sampler_class = RandomDoubleAxisSampler
    elif sampler_name == 'rand_and_yaw_and_pitch':
        sampler_class = RandomAndDoubleAxisSampler
    else:
        raise NotImplementedError()

    if scaler_name == '0':
        scaler_conf = ConstantScalerConf(max_center_std=max_center_std, max_angle=max_angle_delta)
        scaler_class = ConstantScaler
    elif scaler_name == '1':
        scaler_conf = UniformScalerConf(max_center_std=max_center_std, max_angle=max_angle_delta,
                                         N_steps=args.steps, gamma=args.gamma)
        scaler_class = UniformScaler
    else:
        raise NotImplementedError()
    
    sampler = sampler_class(sampler_conf)    
    scaler = scaler_class(scaler_conf)

    return sampler, scaler


@dataclass
class ConstantScalerConf:
    max_center_std: torch.tensor = torch.tensor([1, 1, 1])
    max_angle: torch.tensor = torch.tensor([8])


@dataclass
class UniformScalerConf(ConstantScalerConf):
    N_steps: int = 20
    gamma: float = 0.1    


@dataclass
class SamplerConf:
    pass