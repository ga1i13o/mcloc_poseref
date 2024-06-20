import argparse
import os
import shutil
from os.path import join
from glob import glob
from tqdm import tqdm


def main(folder, only_step=-1):
    if not os.path.isdir(folder):
        raise ValueError(f'{folder} is not a directory')
    if not 'renderings' in os.listdir(folder):
        raise ValueError(f'{folder} should be a log directory')
        
    steps_folders = glob(join(folder, 'renderings', '*'))
    print(f'found {len(steps_folders)} folders')
    dir_to_step = lambda x: int(x.split('/')[-1].split('_')[2].split('s')[-1])
    
    for step_f in tqdm(steps_folders, ncols=100):
        s_num = dir_to_step(step_f)
        if (only_step != -1) and (s_num != only_step):
            continue
             
        query_renders = glob(join(step_f, '**', '*.png'), recursive=True)
        print(f'\nStep {s_num}: found {len(query_renders)} renderings, deleting them...')    
        for q_r in tqdm(query_renders, ncols=100):
            os.remove(q_r)

        vreph_dir = join(step_f, 'vreph_conf')
        if os.path.isdir(vreph_dir):
            n_vreph = len(os.listdir(vreph_dir))
            print(f'Found {n_vreph} files in vreph conf, deleting them...')
            shutil.rmtree(vreph_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Argument parser')
    parser.add_argument('folder', type=str, help='log folder')
    parser.add_argument('-s', '--step', type=int, default=-1, help='delete only step n.')

    args = parser.parse_args()
    main(args.folder, args.step)
    
    
"""  
Example
python clean_logs.py logs/pt2_1_kings_cpl3_320_colmaporig/2023-08-24_23-40-16
python clean_logs.py logs/pt2_1_V2_T6_N40_M2_kings_nerf_cpl3_nozstd_320_1/2023-09-19_14-05-18
"""