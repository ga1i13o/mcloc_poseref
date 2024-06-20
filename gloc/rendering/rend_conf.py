from dataclasses import dataclass

from gloc.rendering.mesh_renderer import MeshRenderer


@dataclass
class O3DConf():
    mesh_path:str
    
@dataclass
class NeRFConf():
    ns_config:str
    ns_transform:str

@dataclass
class GSplattingConf():
    gaussians:str


def get_renderer(args, paths_conf):
    
    if args.renderer == 'o3d':
        rend_class = MeshRenderer
        conf = O3DConf(paths_conf[args.name]['mesh_path'])

    elif args.renderer == 'nerf':
        from gloc.rendering.nerf_renderer import NerfRenderer
        conf = NeRFConf(ns_config=paths_conf[args.name]['ns_config'], 
                        ns_transform=paths_conf[args.name]['ns_transform'])
        rend_class = NerfRenderer
            
    elif args.renderer == 'g_splatting':
        from gloc.rendering.splatting_renderer import GaussianSplattingRenderer 
        conf = GSplattingConf(paths_conf[args.name]['gaussians'])
        rend_class = GaussianSplattingRenderer

    else:
        raise NotImplementedError()
    
    renderer = rend_class(conf)
    
    return renderer
