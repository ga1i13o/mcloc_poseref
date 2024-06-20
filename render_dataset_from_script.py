import argparse
import os
from os.path import join
import numpy as np

from configs import get_path_conf
from gloc import rendering
from gloc.rendering import get_renderer
from gloc.datasets import PoseDataset
from gloc.utils import rotmat2qvec

""" 
Example:
python render_dataset_from_script.py stairs --out_dir nerf_renders --colmap_res 320 --renderer g_splatting
"""

parser = argparse.ArgumentParser(description='Argument parser')
# general args
parser.add_argument('name', type=str, help='')
parser.add_argument('--out_dir', type=str, help='', default='')
parser.add_argument('--renderer', type=str, help='', default='nerf')
parser.add_argument('--colmap_res', type=int, default=320, help='')
args = parser.parse_args()

print(args)

DS = args.name
out_dir = join(args.out_dir, args.name, args.renderer, str(args.colmap_res))
os.makedirs(out_dir, exist_ok=True)
print(f'Renders will in in {out_dir}')

##### parse path info and instantiate dataset
paths_conf = get_path_conf(args.colmap_res, None)
pd = PoseDataset(DS, paths_conf[DS])
####################

#### parse pose and intrinsics metadata
all_tvecs, all_Rs = pd.get_q_poses()
names = [pd.images[pd.q_frames_idxs[q_idx]].name.replace('/', '_') for q_idx in range(len(pd.q_frames_idxs)) ]
key = os.path.splitext(pd.images[0].name)[0]
chosen_camera = pd.intrinsics[key]
height = chosen_camera['h']
width = chosen_camera['w']
K = chosen_camera['K']

calibr_pose = []
all_qvecs = []
for tvec, R in zip(all_tvecs, all_Rs):
    all_qvecs.append(rotmat2qvec(R))

    T = np.eye(4)
    T[0:3, 0:3] = R
    T[0:3, 3] = tvec
    calibr_pose.append((T, K))
####################
renderer = get_renderer(args, paths_conf)
mod = renderer.load_model()
renderer.render_poses(out_dir, mod, names, all_tvecs, all_qvecs, calibr_pose, (width, height), deferred=False)

renderer.clean_file_names(out_dir, names, verbose=True)
# specify 'mesh' as renderer because we converted filenames
rendering.log_poses(out_dir, names, all_tvecs, all_qvecs, 'mesh')
