import os
from os.path import join, dirname
import argparse

import commons
from parse_args import parse_args
from configs import get_path_conf
from gloc import extraction
from gloc import rendering
from gloc.rendering import get_renderer
from gloc.datasets import get_dataset, RenderedImagesDataset, get_transform
from gloc.utils import utils, visualization
from gloc.build_fine_model import get_fine_model
from gloc.resamplers import get_protocol


def main(args):
    DS = args.name
    print(f"Arguments: {args}")
    
    paths_conf = get_path_conf(320, None)
    pd = get_dataset(DS, paths_conf[DS], None)

    print(f'Loading pose prior from {args.pose_prior}')
    all_pred_t, all_pred_R = utils.load_pose_prior(args.pose_prior, pd)
    
    all_true_t, all_true_R = pd.get_q_poses()
    errors_t, errors_R = utils.get_all_errors_first_estimate(all_true_t, all_true_R, all_pred_t, all_pred_R)
    out_str, _ = utils.eval_poses(errors_t, errors_R, descr='Retrieval first estimate')
    print(out_str)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('name', type=str, help='colmap')
    parser.add_argument('pose_prior', type=str, help='colmap')

    args = parser.parse_args()
    main(args)


"""
Example:
python eval_pose_file.py chess_dslam logs/reb_chess_2_bms2_pt2_0_N36_M2_V004_T2_gsplat_dinol4_320/2024-01-27_16-21-05/renderings/pt2_0_s19_sz320_theta2,0_t0,0_0,0_0,0/est_poses.txt
"""
