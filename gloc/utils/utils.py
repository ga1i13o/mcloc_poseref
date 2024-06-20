import os
from math import ceil
import numpy as np
import einops
import faiss
from os.path import join

from gloc.utils import qvec2rotmat, rotmat2qvec

threshs_t = [ 0.25, 0.5, 5.0, 10.0]
threshs_R = [2.0, 5.0, 10.0, 15.0]


def log_pose_estimate(render_dir, pd, pred_R, pred_t, flat_preds=None, top_ns=[3, 6]):
    f_results = join(render_dir, 'est_poses.txt')
    is_aachen = 'Aachen' in pd.name
    print(f'WRITING TO {f_results}')
    with open(f_results, 'w') as f:
        for q_idx in range(len(pd.q_frames_idxs)):
            idx = pd.q_frames_idxs[q_idx]
            name = pd.images[idx].name
            if flat_preds is not None:
                qvec = rotmat2qvec(pred_R[q_idx][flat_preds[q_idx,0]])
                tvec = pred_t[q_idx][flat_preds[q_idx,0]]
            else:
                qvec = rotmat2qvec(pred_R[q_idx][0])
                tvec = pred_t[q_idx][0]

            if is_aachen:
                name = os.path.basename(name)
            qvec = ' '.join(map(str, qvec))
            tvec = ' '.join(map(str, tvec))
            f.write(f'{name} {qvec} {tvec}\n')

    for topn in top_ns:
        tn_results = join(render_dir, f'top{topn}_est_poses.txt')
        print(f'WRITING TO {tn_results}')

        with open(tn_results, 'w') as f:
            for q_idx in range(len(pd.q_frames_idxs)):
                idx = pd.q_frames_idxs[q_idx]
                name = pd.images[idx].name
                for i in range(topn):
                    if flat_preds is not None:
                        qvec = rotmat2qvec(pred_R[q_idx][flat_preds[q_idx, i]])
                        tvec = pred_t[q_idx][flat_preds[q_idx, i]]
                    else:
                        qvec = rotmat2qvec(pred_R[q_idx, i])
                        tvec = pred_t[q_idx, i]
                    
                    name = os.path.basename(name)
                    qvec = ' '.join(map(str, qvec))
                    tvec = ' '.join(map(str, tvec))
                    f.write(f'{name} {qvec} {tvec}\n')

    return f_results


def load_pose_prior(pose_file, pd, M=1):
    with open(pose_file, 'r') as fp:
        est_poses = fp.readlines()
    # the format is: 'basename_img.ext qw qx qy qz tx ty tz\n'   
    est_poses = list(map(lambda x: x.strip().split(' '), est_poses))
    poses_dict = {}
    for pose in est_poses:
        qvec_float = list(map(float, pose[1:5]))
        tvec_float = list(map(float, pose[5:8]))
        if pose[0] not in poses_dict:
            poses_dict[pose[0]] = []
            
        poses_dict[pose[0]].append( (np.array(qvec_float), np.array(tvec_float)) )
        
    all_pred_t, all_pred_R = np.empty((pd.n_q, M, 3)), np.empty((pd.n_q, M, 3, 3))
    
    if pd.name in ['Aachen_night', 'Aachen_day', 'Aachen_real', 'Aachen_real_und']:
        get_q_key = lambda x: os.path.basename(x)
    else:
        get_q_key = lambda x: x        
    for q_idx in range(len(pd.q_frames_idxs)):
        idx = pd.q_frames_idxs[q_idx]
        #q_key = os.path.basename(pd.images[idx].name)
        q_key = get_q_key(pd.images[idx].name)
        poses_q = poses_dict[q_key]

        if len(poses_q) == 1:
            qvec, tvec = poses_q[0]
            R = qvec2rotmat(qvec).reshape(-1, 3, 3)
            R_rp = np.repeat(R, M, axis=0)
            tvec_rp = np.repeat(tvec.reshape(-1, 3), M, axis=0)

            all_pred_t[q_idx] = tvec_rp
            all_pred_R[q_idx] = R_rp
    
        else:
            assert len(poses_q) >= M, f'This query has {len(poses_q)} poses and you asked for {M}'
            for i in range(M):
                qvec, tvec = poses_q[i]
                R = qvec2rotmat(qvec)
                tvec = tvec
                all_pred_t[q_idx, i] = tvec
                all_pred_R[q_idx, i] = R

    return all_pred_t, all_pred_R


def reshape_preds_per_beam(n_beams, M, preds):
    n_dims = len(preds.shape)
    to_stack = []
    for i in range(n_beams):
        # with n-beams, beam i gets the i-th cand., and the i+n, i+2, so on
        # so take one every n
        ts  = preds[:,i::n_beams][:,:M, :]
        to_stack.append(ts)

    if n_dims == 3:
        stacked = np.hstack(to_stack).reshape(-1, n_beams, M, 3)
    elif n_dims == 4:
        stacked = np.hstack(to_stack).reshape(-1, n_beams, M, 3, 3)

    return stacked


def repeat_first_preds_per_beam(n_beams, M, preds):
    first_preds = preds[:, :n_beams, :]
    # go from (Q, n_beams, 3/3,3), to (Q, n_beams, 1, 3/3,3)
    # so that then the first pred. in each beam can be repeated N times
    first_preds = np.expand_dims(first_preds, axis=2)
    result = np.repeat(first_preds, M, axis=2)

    return result


def get_n_steps(num_queries, render_per_step, max_steps, renderer, hard_stop):
    if hard_stop > 0:
        return hard_stop
    if renderer != 'o3d':
        return max_steps
    # due to open3d bug, black images after 1e5 renders
    max_renders = 1e5
    n_steps = ceil(max_renders / (num_queries*render_per_step))
    return n_steps
        
    
def eval_poses(errors_t, errors_R, descr=''):
    med_t = np.median(errors_t)
    med_R = np.median(errors_R)
    out = f'Results {descr}:'
    out += f'\nMedian errors: {med_t:.3f}m, {med_R:.3f}deg'
    out_vals = []
    
    out += '\nPercentage of test images localized within:'
    for th_t, th_R in zip(threshs_t, threshs_R):
        ratio = np.mean((errors_t < th_t) & (errors_R < th_R))
        out += f'\n\t{th_t:.2f}m, {th_R:.0f}deg : {ratio*100:.2f}%'
        out_vals.append(ratio)
        
    return out, np.array(out_vals)


def eval_poses_top_n(all_errors_t, all_errors_R, descr=''):
    best_candidates = [1, 5, max(20, all_errors_R.shape[1])]

    med_t = np.median(all_errors_t[:, 0])
    med_best_t = np.median(all_errors_t.min(axis=1))

    med_R = np.median(all_errors_R[:, 0])
    med_best_R = np.median(all_errors_R.min(axis=1))

    out = f'Results {descr}:'
    out += f'\nMedian errors on first/best: {med_t:.2f}m, {med_R:.2f}deg // {med_best_t:.2f}m, {med_best_R:.2f}deg'
    out_vals = np.zeros((len(best_candidates), len(threshs_t)))
    
    out += f"\nPercentage of test images localized within (TOP {'|TOP '.join(map(str, best_candidates))}):"
    for i, (th_t, th_R) in enumerate(zip(threshs_t, threshs_R)):
        out += f'\n\t{th_t:.2f}m, {int(th_R):2d}deg :'
        for j, best in enumerate(best_candidates):
            ratio = np.mean( ((all_errors_t[:, :best] < th_t) & (all_errors_R[:, :best] < th_R)).any(axis=1) )
            out += f' {ratio*100:4.1f}% |'
            out_vals[j, i] = ratio
            
    return out, out_vals


def get_predictions(db_descriptors, q_descriptors, pose_dataset, fc_output_dim=512, top_k=20):
    pd = pose_dataset

    # Use a kNN to find predictions
    faiss_index = faiss.IndexFlatL2(fc_output_dim)
    faiss_index.add(db_descriptors)

    _, predictions = faiss_index.search(q_descriptors, top_k)    
    pred_Rs = []
    pred_ts = []
    for q_idx in range(len(pd.q_frames_idxs)):
        pred_Rs_per_query = []
        pred_ts_per_query = []
        for k in range(top_k):
            pred_idx = pd.db_frames_idxs[predictions[q_idx, k]]
            R = qvec2rotmat(pd.images[pred_idx].qvec)
            t = pd.images[pred_idx].tvec
            pred_Rs_per_query.append(R)
            pred_ts_per_query.append(t)
        
        pred_Rs.append(np.array(pred_Rs_per_query))
        pred_ts.append(np.array(pred_ts_per_query))

    return np.array(pred_ts), np.array(pred_Rs)


def get_error(R, t, R_gt, t_gt):
    e_t = np.linalg.norm(-R_gt.T @ t_gt + R.T @ t, axis=0)
    cos = np.clip((np.trace(np.dot(R_gt.T, R)) - 1) / 2, -1., 1.)
    e_R = np.rad2deg(np.abs(np.arccos(cos)))
    
    return e_t, e_R


def get_errors_from_preds(true_t, true_R, pred_t, pred_R, top_k=20):
    errors_t = []
    errors_R = []
    
    top_k = min(top_k, pred_t.shape[1])
    for k in range(top_k):
        e_t, e_R = get_error(pred_R[0, k], pred_t[0, k], true_R[0], true_t[0])

        errors_t.append(e_t)
        errors_R.append(e_R)
    errors_t, errors_R = np.array(errors_t), np.array(errors_R)
    
    return errors_t, errors_R


def get_all_errors_first_estimate(true_t, true_R, pred_t, pred_R):
    errors_t = []
    errors_R = []
    n_queries = len(pred_R)
    for q_idx in range(n_queries):
        e_t, e_R = get_error(pred_R[q_idx, 0], pred_t[q_idx, 0], true_R[q_idx], true_t[q_idx])

        errors_t.append(e_t)
        errors_R.append(e_R)
    errors_t = np.array(errors_t)
    errors_R = np.array(errors_R)

    return errors_t, errors_R


def get_pose_from_preds_w_truth(q_idx, pd, rd, predictions, top_k=20):
    true_Rs = []
    true_ts = []
    pred_Rs = []
    pred_ts = []

    idx = pd.q_frames_idxs[q_idx]
    R_gt = qvec2rotmat(pd.images[idx].qvec)
    t_gt = pd.images[idx].tvec
    true_Rs.append(R_gt)
    true_ts.append(t_gt)

    pred_Rs_per_query = []
    pred_ts_per_query = []

    top_k = min(top_k, len(predictions))
    for k in range(top_k):
        pred_idx = predictions[k]

        R = qvec2rotmat(rd.images[pred_idx].qvec)
        t = rd.images[pred_idx].tvec
        pred_Rs_per_query.append(R)
        pred_ts_per_query.append(t)

    pred_Rs.append(np.array(pred_Rs_per_query))
    pred_ts.append(np.array(pred_ts_per_query))

    true_t, true_R, pred_t, pred_R= np.array(true_ts), np.array(true_Rs), np.array(pred_ts), np.array(pred_Rs)
    return true_t, true_R, pred_t, pred_R


def get_pose_from_preds(q_idx, pd, rd, predictions, top_k=20):
    pred_Rs = []
    pred_ts = []

    pred_Rs_per_query = []
    pred_ts_per_query = []

    top_k = min(top_k, len(predictions))
    for k in range(top_k):
        pred_idx = predictions[k]

        R = qvec2rotmat(rd.images[pred_idx].qvec)
        t = rd.images[pred_idx].tvec
        pred_Rs_per_query.append(R)
        pred_ts_per_query.append(t)

    pred_Rs.append(np.array(pred_Rs_per_query))
    pred_ts.append(np.array(pred_ts_per_query))

    pred_t, pred_R= np.array(pred_ts), np.array(pred_Rs)
    return pred_t, pred_R


def sort_preds_across_beams(all_scores, all_pred_t, all_pred_R, all_errors_t, all_errors_R):
    # flatten stuff to sort predictions based on similarity
    flat_err = lambda x: einops.rearrange(x, 'q nb N      -> q (nb N)')
    flat_R = lambda x:   einops.rearrange(x, 'q nb N d1 d2 -> q (nb N) d1 d2', d1=3, d2=3)
    flat_t = lambda x:   einops.rearrange(x, 'q nb N d     -> q (nb N) d', d=3)
    flat_preds = np.argsort(flat_err(all_scores))
    all_errors_t = np.take_along_axis(flat_err(all_errors_t), flat_preds, axis=1)
    all_errors_R = np.take_along_axis(flat_err(all_errors_R), flat_preds, axis=1)
    flat_pred_t = flat_t(all_pred_t)
    flat_pred_R = flat_R(all_pred_R)
    
    return flat_pred_R, flat_pred_t, flat_preds, all_errors_t, all_errors_R


def update_scores(scores, scores_temp):
    if scores is None:
        scores = scores_temp
    else:
        scores['steps'] += scores_temp['steps']

    return scores
