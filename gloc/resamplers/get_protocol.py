from dataclasses import dataclass
from os.path import join
import torch

from gloc.resamplers.scalers_conf import get_sampler
from gloc.resamplers.strategies import Protocol1, Protocol2


def get_protocol(args, n_views, protocol):
    scaler_name = protocol.split('_')[-1]
    sampler, scaler = get_sampler(args, args.sampler, scaler_name)

    if protocol.startswith('1'):
        protocol_conf = Protocol1Conf(N_steps=args.steps, n_views=n_views)
        protocol_class = Protocol1
        
    elif protocol.startswith('2'):
        protocol_conf = Protocol2Conf(N_steps=args.steps, n_views=n_views, M_candidates=args.M)
        protocol_class = Protocol2
    
    else:
        raise NotImplementedError

    protocol_obj = protocol_class(protocol_conf, sampler, scaler, protocol)
    return protocol_obj

    
@dataclass
class ProtocolConf:
    N_steps: int = 20
    n_views: int = 20 
    
       
@dataclass
class Protocol1Conf(ProtocolConf):
    pass


@dataclass
class Protocol2Conf(ProtocolConf):
    M_candidates: int = 4
