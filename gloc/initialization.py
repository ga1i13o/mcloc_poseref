import os
from os.path import join, dirname
import torch
import logging

from gloc import extraction
from gloc.utils import utils


def init_refinement(args, pose_dataset):
    first_step = 0
    scores = {}

    if (args.pose_prior is None) and (args.resume_step is None):
        # if no pose prior, init with retrieval
        all_pred_t, all_pred_R = extraction.get_retrieval_predictions(args.retr_model, args.res, pose_dataset, args.beams*args.M)
    elif args.pose_prior is not None:
        assert os.path.isfile(args.pose_prior), f'{args.pose_prior} does not exist as a file'

        logging.info(f'Loading pose prior from {args.pose_prior}')
        all_pred_t, all_pred_R = utils.load_pose_prior(args.pose_prior, pose_dataset, args.beams*args.M)

    if args.resume_step is None:
        all_true_t, all_true_R = pose_dataset.get_q_poses()
        errors_t, errors_R = utils.get_all_errors_first_estimate(all_true_t, all_true_R, all_pred_t, all_pred_R)

        out_str, out_vals = utils.eval_poses(errors_t, errors_R, descr='Retrieval first estimate')
        scores['baseline'] = out_vals
        logging.info(out_str)

        # go from (NQ, M*beams, 3/3,3) to (NQ, beams, M, 3/3, 3)
        all_pred_t = utils.reshape_preds_per_beam(args.beams, args.M, all_pred_t)
        all_pred_R = utils.reshape_preds_per_beam(args.beams, args.M, all_pred_R)
    else:
        all_pred_t, all_pred_R = None, None

    scores['steps'] = []
    if args.first_step is not None:
        first_step = args.first_step
        if args.resume_step is not None:
            score_path = join(dirname(dirname(args.resume_step)), 'scores.pth')
            if os.path.isfile(score_path):
                scores = torch.load(score_path)
                if len(scores['steps']) > first_step:
                    logging.info(f"Cutting score file to {first_step},  from {len(scores['steps'])}")
                    scores['steps'] = scores['steps'][:first_step] 

    return first_step, all_pred_t, all_pred_R, scores
