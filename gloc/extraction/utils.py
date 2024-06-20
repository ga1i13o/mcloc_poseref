import logging
import os
import numpy as np
from tqdm import tqdm
import torch
import einops
from os.path import join
from torch.utils.data.dataset import Subset

from gloc.utils import utils, qvec2rotmat
from gloc import extraction
from gloc.datasets import RenderedImagesDataset
from gloc import models


def get_retrieval_predictions(model_name, res, pose_dataset, topk=5):
    model = models.get_retrieval_model(model_name)

    db_descriptors, q_descriptors = extraction.extract_features(model, model_name, pose_dataset, res, bs=1)
    logging.info(f'N. db descriptors: {db_descriptors.shape[0]}, N. Q descriptors: {q_descriptors.shape[0]}')    
    logging.info('Computing first retrieval prediction...')
    all_pred_t, all_pred_R = utils.get_predictions(db_descriptors, q_descriptors, pose_dataset, top_k=topk)
    # all_true_t, all_true_R, all_pred_t, all_pred_R = utils.get_predictions_w_truths(db_descriptors, q_descriptors, pd, top_k=topk)

    return all_pred_t, all_pred_R
    

def get_predictions_from_step_dir(fine_model, pd, render_dir, transform, N_per_beam, n_beams):
    queries_subset = Subset(pd, pd.q_frames_idxs)
    q_descriptors = extraction.get_features_from_dataset(fine_model, queries_subset)

    all_pred_t   = np.empty((len(pd.q_frames_idxs), n_beams, N_per_beam, 3))
    all_pred_R   = np.empty((len(pd.q_frames_idxs), n_beams, N_per_beam, 3, 3))
    names =     np.empty((len(pd.q_frames_idxs), n_beams, N_per_beam), dtype=object)
    all_scores   = np.empty((len(pd.q_frames_idxs), n_beams, N_per_beam))

    for q_idx in tqdm(range(len(pd.q_frames_idxs)), ncols=100):
        q_name = pd.get_basename(pd.q_frames_idxs[q_idx])        
        query_dir = os.path.join(render_dir, q_name)
        query_tensor = pd[pd.q_frames_idxs[q_idx]]['im']
        query_res = tuple(query_tensor.shape[-2:])

        q_feats = extraction.get_query_descriptor_by_idx(q_descriptors, q_idx)
        for beam_i in range(n_beams):
            if n_beams == 1:
                beam_dir = query_dir
            else:
                beam_dir = join(query_dir, f'beam_{beam_i}')
            rd = RenderedImagesDataset(beam_dir, transform, query_res, verbose=False)        

            scores_file = join(beam_dir, 'scores.pth')
            if not os.path.isfile(scores_file):
                r_db_descriptors = extraction.get_features_from_dataset(fine_model, rd, bs=fine_model.conf.bs, is_render=True)
                predictions, scores = fine_model.rank_candidates(q_feats, r_db_descriptors, get_scores=True)
                torch.save((predictions, scores), scores_file)
            else:
                predictions, scores = torch.load(scores_file)
            pred_t, pred_R = utils.get_pose_from_preds(q_idx, pd, rd, predictions, N_per_beam)

            all_pred_t[q_idx, beam_i] = pred_t
            all_pred_R[q_idx, beam_i] = pred_R
            all_scores[q_idx, beam_i] = scores[:N_per_beam]

            nn = []
            for pr in predictions[:N_per_beam]:
                nn.append(rd.images[pr].name)
            names[q_idx, beam_i] = np.array(nn)
    
    #del q_feats, r_db_descriptors
    if n_beams > 1:
        # flatten stuff to sort predictions based on similarity
        flatten_beams = lambda x: einops.rearrange(x, 'q nb N      -> q (nb N)')
        flatten_R = lambda x:   einops.rearrange(x, 'q nb N d1 d2 -> q (nb N) d1 d2', d1=3, d2=3)
        flatten_t = lambda x:   einops.rearrange(x, 'q nb N d     -> q (nb N) d', d=3)

        flat_preds = np.argsort(flatten_beams(all_scores))
        names = np.take_along_axis(flatten_beams(names), flat_preds, axis=1)

        flat_t = flatten_t(all_pred_t)
        flat_R = flatten_R(all_pred_R)

        stacked_t, stacked_R = [], []
        for i in range(len(all_pred_t)):
            sorted_t = flat_t[i, flat_preds[i]]
            sorted_R = flat_R[i, flat_preds[i]]

            stacked_t.append(sorted_t)
            stacked_R.append(sorted_R)
        all_pred_t = np.stack(stacked_t)
        all_pred_R = np.stack(stacked_R)
    else:
        all_pred_t = all_pred_t.squeeze()
        all_pred_R = all_pred_R.squeeze()
        names = names.squeeze()         
        
    return all_pred_t, all_pred_R, names


def split_renders_into_chunks(num_queries, num_candidates, n_beams, N_per_beam, im_per_chunk):
    chunk_limit = im_per_chunk
    if num_queries > chunk_limit:
        q_range = np.arange(num_queries)
        # this will be [0, 1100, 2200..]
        start_chunk_q_idx = q_range[::chunk_limit] 
        # start from [1], so the first chunk is from 0 to 1100*beams*n_beam
        chunk_idx_ims = [start_q*n_beams*N_per_beam for start_q in start_chunk_q_idx[1:]] 
        chunks = [np.arange(chunk_idx_ims[0])]
        for ic in range(1, len(chunk_idx_ims)):
            this_chunk = np.arange(chunk_idx_ims[ic-1], chunk_idx_ims[ic])
            chunks.append(this_chunk)
        last_chunk = np.arange(chunk_idx_ims[-1], num_candidates)
        chunks.append(last_chunk)
        
        chunk_start_q_idx = start_chunk_q_idx
        chunk_end_q_idx = list(start_chunk_q_idx[1:]) + [num_queries]
    else:
        chunks = [np.arange(num_candidates)]
        chunk_start_q_idx = [0]
        chunk_end_q_idx = [num_queries]

    return chunk_start_q_idx, chunk_end_q_idx, chunks


def get_feat_dim(fine_model, query_res):
    x = torch.rand(1, 3, *query_res)
    dim = tuple(fine_model(x.cuda()).shape)[1:]

    return dim
