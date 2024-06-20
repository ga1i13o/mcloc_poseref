import logging
import os
import sys
import shutil
import torch
from os.path import join
from tqdm import tqdm
import numpy as np
import einops
from torch.utils.data.dataset import Subset
import torchvision.transforms as T

import commons
from parse_args import parse_args
from path_configs import get_path_conf
from gloc.models import get_ref_model
from gloc import extraction
from gloc import rendering
from gloc.rendering import get_renderer
from gloc.utils import utils, rotmat2qvec 
from gloc.resamplers import get_protocol
from gloc.datasets import RenderedImagesDataset, get_dataset, get_transform


def main(args):
    DS = args.name
    res = args.res

    commons.make_deterministic(args.seed)
    commons.setup_logging(args.save_dir, console="info")
    logging.info(" ".join(sys.argv))
    logging.info(f"Arguments: {args}")
    logging.info(f"The outputs are being saved in {args.save_dir}")
    
    paths_conf = get_path_conf(args.colmap_res, args.mesh)
    temp_dir = join(paths_conf['temp'], args.exp_name)            
    os.makedirs(temp_dir)

    transform = get_transform(args)
    pd = get_dataset(DS, paths_conf[DS], transform)

    if args.pose_prior == '':
        all_pred_t, all_pred_R = extraction.get_retrieval_predictions(args.model, args.outdim, args.res, pd, topk=args.beams*args.M)
    else:
        logging.info(f'Loading pose prior from {args.pose_prior}')
        all_pred_t, all_pred_R = utils.load_pose_prior(args.pose_prior, pd, args.beams*args.M)

    ######### START REFINEMENT LOOP
    N_steps = args.steps
    n_beams = args.beams
    N_per_beam = args.N // args.beams
    M = args.M
    fine_model = get_ref_model(args)

    logging.info('Recomputing query features with refinement model...')
    queries_subset = Subset(pd, pd.q_frames_idxs)
    q_descriptors = extraction.get_features_from_dataset(fine_model, queries_subset, use_tqdm=True)
    
    resampler = get_protocol(args, N_per_beam, args.protocol)
    renderer = get_renderer(args, paths_conf)

    first_step = 0
    if args.first_step is not None:
        first_step = args.first_step
            
    max_step = utils.get_n_steps(pd.num_queries(), args.N, N_steps, args.renderer, args.hard_stop)
    # go from (NQ, M*beams, 3/3,3) to (NQ, beams, M, 3/3, 3)
    all_pred_t = utils.reshape_preds_per_beam(n_beams, M, all_pred_t)
    all_pred_R = utils.reshape_preds_per_beam(n_beams, M, all_pred_R)
    for step in range(first_step, N_steps):
        if (step - first_step) == max_step:
            logging.info('Stopping due to Open3D bug')
            break

        resampler.init_step(step)
        center_std, angle_delta = resampler.scaler.get_noise()
        
        logging.info(f'[||] Starting iteration n.{step+1}/{N_steps} [||]')
        logging.info(f'Perturbing poses with Theta {angle_delta} and center STD {center_std}. Resolution {res}')

        if (first_step == step) and (args.resume_step is not None):
            render_dir = args.resume_step
        else:
            perturb_str = resampler.get_pertubr_str(step, res)
            render_dir = perturb_step(perturb_str, pd, renderer, resampler, all_pred_t, all_pred_R, temp_dir, n_beams)

        all_pred_t, all_pred_R = rank_candidates(fine_model, pd, render_dir, 
                                                pd.transform, q_descriptors, N_per_beam, n_beams)
        
        logging.info(f'[!] Concluded iteration n.{step+1}/{N_steps} [!]')

    ### cleaning up...    
    if args.clean_logs:
        from clean_logs import main as cl_logs
        logging.info('Removing rendering files...')
        cl_logs(temp_dir)

    logging.info(f'Moving rendering from temp dir {temp_dir} to {args.save_dir}')
    shutil.move(join(temp_dir, 'renderings'), args.save_dir, copy_function=shutil.move)
    shutil.rmtree(temp_dir)
    logging.info('Terminating without errors!')    


def perturb_step(perturb_str, pd, renderer, resampler, pred_t, pred_R, basepath, n_beams=1):    
    out_dir = os.path.join(basepath, 'renderings', perturb_str)
    os.makedirs(out_dir)
    logging.info(f'Generating renders in {out_dir}')

    rend_model = renderer.load_model()
    r_names_per_beam = {}        
    for q_idx in tqdm(range(len(pd.q_frames_idxs)), ncols=100):
        idx = pd.q_frames_idxs[q_idx]
        q_name = pd.get_basename(idx)
        q_key_name = pd.images[idx].name.split('.')[0]
        r_names_per_beam[q_idx] = {}

        w = pd.q_intrinsics[q_key_name]['w']    
        h = pd.q_intrinsics[q_key_name]['h']    
        K = pd.q_intrinsics[q_key_name]['K']
        r_dir = os.path.join(out_dir, q_name)
        os.makedirs(r_dir)
        for beam_i in range(n_beams):
            beam_dir = join(r_dir, f'beam_{beam_i}')
            os.makedirs(beam_dir)
            pred_t_beam = pred_t[q_idx, beam_i]
            pred_R_beam = pred_R[q_idx, beam_i]

            r_names, render_ts, render_qvecs, calibr_pose = resampler.resample(K, q_name, pred_t_beam, pred_R_beam, 
                                                    q_idx=q_idx, beam_i=beam_i)

            r_names_per_beam[q_idx][beam_i] = r_names
            # poses have to be logged in 'beam_dir', but rendered in 'r_dir', so that
            # they can be rendered all together by deferred renderers such as ibmr
            rendering.log_poses(beam_dir, r_names, render_ts, render_qvecs, args.renderer)
            renderer.render_poses(r_dir, rend_model, r_names, render_ts, render_qvecs, calibr_pose, (w, h))
    del rend_model
    
    renderer.end_epoch(out_dir)
    logging.info('Moving each renders into their beams folder')
    for q_idx in range(len(pd.q_frames_idxs)):
        idx = pd.q_frames_idxs[q_idx]
        q_name = pd.get_basename(idx)
        
        r_dir = join(out_dir, q_name)
        for beam_i in range(n_beams):
            beam_dir = join(r_dir, f'beam_{beam_i}')
            beam_names = r_names_per_beam[q_idx][beam_i]
            for b_name in beam_names:
                src = join(r_dir, b_name+'.png')
                dst = join(beam_dir, b_name+'.png')
                shutil.move(src, dst)
    
    return out_dir


def rank_candidates(fine_model, pd, render_dir, transform, q_descriptors, N_per_beam, n_beams):
    all_pred_t   = np.empty((len(pd.q_frames_idxs), n_beams, N_per_beam, 3))
    all_pred_R   = np.empty((len(pd.q_frames_idxs), n_beams, N_per_beam, 3, 3))
    all_scores   = np.empty((len(pd.q_frames_idxs), n_beams, N_per_beam))

    logging.info('Extracting features from rendered images...')
    for q_idx in tqdm(range(len(pd.q_frames_idxs)), ncols=100):
        q_name = pd.get_basename(pd.q_frames_idxs[q_idx])        
        query_dir = os.path.join(render_dir, q_name)
        query_tensor = pd[pd.q_frames_idxs[q_idx]]['im']
        query_res = tuple(query_tensor.shape[-2:])
        q_feats = extraction.get_query_descriptor_by_idx(q_descriptors, q_idx)

        for beam_i in range(n_beams):
            beam_dir = join(query_dir, f'beam_{beam_i}')
            rd = RenderedImagesDataset(beam_dir, transform, query_res, verbose=False)        
            r_db_descriptors = extraction.get_features_from_dataset(fine_model, rd, bs=fine_model.conf.bs, is_render=True)
            predictions, scores = fine_model.rank_candidates(q_feats, r_db_descriptors, get_scores=True)
            pred_t, pred_R = utils.get_pose_from_preds(q_idx, pd, rd, predictions, N_per_beam)

            all_pred_t[q_idx][beam_i] = pred_t
            all_pred_R[q_idx][beam_i] = pred_R
            all_scores[q_idx][beam_i] = scores[:N_per_beam]

            scores_file = join(beam_dir, 'scores.pth')
            torch.save((predictions, scores), scores_file)

    del q_feats, r_db_descriptors
    # flatten stuff for eval
    flat_score = lambda x: einops.rearrange(x, 'q nb N     -> q (nb N)')
    flat_R = lambda x:   einops.rearrange(x, 'q nb N d1 d2 -> q (nb N) d1 d2', d1=3, d2=3)
    flat_t = lambda x:   einops.rearrange(x, 'q nb N d     -> q (nb N) d', d=3)
    flat_preds = np.argsort(flat_score(all_scores))
    flat_pred_t = flat_t(all_pred_t)
    flat_pred_R = flat_R(all_pred_R)

    # log pose estimate
    logging.info(f'Generating pose file...')
    f_results = utils.log_pose_estimate(render_dir, pd, flat_pred_R, flat_pred_t, flat_preds=flat_preds)
    
    method_name = os.path.dirname(render_dir).split('/')[-2]
    method_name = method_name.replace('_aachenreal_ibmr', '')
    dir_to_step = lambda x: int(x.split('/')[-1].split('_')[2].split('s')[-1])
    step_n = dir_to_step(render_dir)
    if (step_n+1) % 5 == 0:
        method_name += f'_s{step_n}'
        try:
            commons.submit_poses(method=method_name, path=f_results)
        except:
            logging.info('Submit script failed')

    return all_pred_t, all_pred_R


if __name__ == '__main__':
    args = parse_args()
    main(args)
