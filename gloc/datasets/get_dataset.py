from os.path import join
import torchvision.transforms as T

from gloc.datasets import PoseDataset
from gloc.datasets.dataset_nolabels import IntrinsicsDataset


def get_dataset(name, paths_conf, transform=None):
    # if 'Aachen' in name:
    if name in ['Aachen_night', 'Aachen_day', 'Aachen_real', 'Aachen_real_und']:
        dataset = IntrinsicsDataset(name, paths_conf, transform)
    else:
        dataset = PoseDataset(name, paths_conf, transform)
    
    return dataset


def get_transform(args, colmap_dir=''):
    res = args.res
    if args.feat_model == 'Dinov2':
        cam_file = join(colmap_dir, 'cameras.txt')
        random_line = open(cam_file, 'r').readlines()[10].split(' ')
        w, h = int(random_line[2]), int(random_line[3])
        patch_size = 14
        new_h = patch_size * (h // patch_size)
        new_w = patch_size * (w // patch_size)
        transform = T.Compose([
            T.ToTensor(),
            T.Resize((new_h, new_w), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])        

    elif ('Aachen' not in args.name) and (colmap_dir != ''):            
        cam_file = join(colmap_dir, 'cameras.txt')                                                                                                
        random_line = open(cam_file, 'r').readlines()[10].split(' ')
        w, h = int(random_line[2]), int(random_line[3])
        ratio = min(h, w) / res
        new_h = int(h/ratio)
        new_w = int(w/ratio)
        transform = T.Compose([
            T.ToTensor(),
            T.Resize((new_h, new_w), antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    else:
        transform = T.Compose([
            T.ToTensor(),
            T.Resize(res, antialias=True),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    return transform
