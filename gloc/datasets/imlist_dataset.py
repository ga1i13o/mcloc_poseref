from os.path import join
from PIL import Image
import os
import torch.utils.data as data
from tqdm import tqdm

from gloc.datasets import RenderedImagesDataset


class ImListDataset(data.Dataset):
    def __init__(self, path_list, transform=None):
        self.path_list = path_list
        self.transform = transform
        
    def __getitem__(self, idx):
        im = Image.open(self.path_list[idx])

        if self.transform:
            im = self.transform(im)
        return im
    
    def __len__(self):
        return len(self.path_list)


def find_candidates_paths(pose_dataset, n_beams, render_dir):
    candidates_pathlist = []
    for q_idx in tqdm(range(len(pose_dataset.q_frames_idxs)), ncols=100):
        q_name = pose_dataset.get_basename(pose_dataset.q_frames_idxs[q_idx])        
        query_dir = os.path.join(render_dir, q_name)

        for beam_i in range(n_beams):
            beam_dir = join(query_dir, f'beam_{beam_i}')

            rd = RenderedImagesDataset(beam_dir, verbose=False)        
            paths = rd.get_full_paths()
            candidates_pathlist += paths

    # return last query res; assumes they are all the same
    query_tensor = pose_dataset[pose_dataset.q_frames_idxs[q_idx]]['im']
    query_res = tuple(query_tensor.shape[-2:])
    
    return candidates_pathlist, query_res
