import argparse
from os.path import join
from datetime import datetime


def parse_args():
    parser = argparse.ArgumentParser(description='Argument parser')
    # exp args
    parser.add_argument('name', type=str, help='DS name', 
                        choices=['Aachen', 'Aachen_real', 'Aachen_day', 'Aachen_night', 'Aachen_real_und', 'Aachen_small', 
                                'KingsCollege', 'KingsCollege_und', 'StMarysChurch_und',
                                'ShopFacade', 'ShopFacade_und', 
                                'OldHospital', 'OldHospital_und',
                                'chess', 'office', 'fire', 'stairs', 'redkitchen', 'pumpkin', 'heads'])
    parser.add_argument('--exp_name', type=str, help='log folder', default='default')
    parser.add_argument('--res', type=int, help='resolution', default=320)
    parser.add_argument('--seed', type=int, help='seed', default=0)
    parser.add_argument('--first_step', type=int, help='start from', default=None)
    parser.add_argument('--hard_stop', type=int, help='interrupt at step N, but dont consider it for scaling noise', default=-1)
    parser.add_argument('--resume_step', type=str, help='resume folder', default=None)
    parser.add_argument('--save_feats', action='store_true', help='seed', default=False)
    parser.add_argument('--pose_prior', type=str, help='start from a pose prior in this file', default=None)
    parser.add_argument('--clean_logs', action='store_true', help='remove renderings in the end', default=False)
    parser.add_argument('--chunk_size', type=int, help='n feats at a time', default=1100)

    # model args
    parser.add_argument('--retr_model', type=str, help='retrieval model', default='cosplace', choices=['cosplace'])
    parser.add_argument('--ref_model', type=str, help='fine model', default='DenseFeatures', 
                        choices=['DenseFeatures'])
    parser.add_argument('--feat_model', type=str, help='refinement model arch', default='cosplace',
        choices=['',
                'cosplace_r18_l1', 'cosplace_r18_l2', 'cosplace_r18_l3', 
                 'cosplace_r50_l1', 'cosplace_r50_l2', 'cosplace_r50_l3', 
                 'resnet18_l1', 'resnet18_l2', 'resnet18_l3',
                 'resnet50_l1', 'resnet50_l2', 'resnet50_l3',
                 'alexnet_l1', 'alexnet_l2', 'alexnet_l3',
                 'Dinov2', 'Roma',
        ])
    
    # fine models args
    parser.add_argument('--clamp_score', type=float, help='thresholded scoring function', default=-1)
    parser.add_argument('--feat_level', nargs='+', type=int, help='Level of features for ALIKED', default=[-1])
    parser.add_argument('--scale_fmaps', type=int, help='Scale F.maps to 1/n', default=6)

    # path args
    parser.add_argument("--storage_dir", type=str, default='/storage/gtrivigno/vloc/renderings', help='model path')
    parser.add_argument("--fix_storage", action='store_true', default=False, help='model path')

    # render args
    parser.add_argument('--colmap_res', type=int, help='res', default=320)    
    parser.add_argument('--mesh', type=str, help='mesh type', choices=['colored', 'colored_14', 'colored_15', 'textured'], default='colored')    
    parser.add_argument('--renderer', type=str, help='renderer type', choices=['o3d', 'nerf', 'g_splatting'], default='o3d')    

    # perturb args
    parser.add_argument('-pt', '--protocol', type=str, help='protocol', 
                        choices=['1_0', '1_1', '2_0', '2_1'], default='2_1')
    parser.add_argument('--sampler', type=str, help='sampler', default='rand',
                        choices=['rand', 'rand_yaw_or_pitch', 'rand_yaw_and_pitch', 'rand_and_yaw_and_pitch'])    
    parser.add_argument('--beams', type=int, help='N. beams to optimize independetly', default=1)
    parser.add_argument('--steps', type=int, help='iterations', default=20)
    parser.add_argument('--teta', nargs='+', type=float, help='max angle', default=[8])
    parser.add_argument('--center_std', nargs='+', type=float, default=[1., 1., 1.])
    parser.add_argument('--N', type=int, help='N views to render in total, per query', default=20)
    parser.add_argument('--M', type=int, help='In each beam, perturb the first M rather than only first cand.', default=4)
    parser.add_argument('--gamma', type=float, help='min scale', default=0.1)

    # eval scripts
    parser.add_argument("--eval_renders_dir", type=str, default='', help='eval render dir')

    # random args
    parser.add_argument('--only_step', type=int, help='iterations', default=-1)

    args = parser.parse_args()
    args.save_dir = join("logs", args.exp_name, datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))

    ## some consistency checks
    assert args.N % args.beams == 0, 'N (total views to rend) has to be a multiple of N. beams'
    if args.mesh == 'textured' and not args.name.startswith('Aachen'):
        raise ValueError('Textured mesh is only available for Aachen')
    if args.protocol[0] not in ['0', '1']:
        assert args.N % args.M == 0, f'In protocol 2, N ({args.N}) has to be a multiple of M ({args.M})'
        assert (args.N // args.beams) % args.M == 0, f'In protocols with M!=1, N/beams ({args.N//args.beams}) has to be a multiple of M ({args.M})'
    if 'yaw_and_pitch' in args.sampler:
        assert len(args.teta) == 2, f'Sampler {args.sampler} requires 2 angles, 1 for yaw, 1 for pitch'
    else:
        assert len(args.teta) == 1, f'Sampler {args.sampler} requires only 1 angle'        

    return args