import os
import numpy as np
from os.path import join
import matplotlib as mpl
import matplotlib.pyplot as plt
import cv2

from gloc.utils.utils import threshs_R, threshs_t


def plot_error_distr(err_R, err_t, step_num, save_dir, f_name):
    fig, axi = plt.subplots(1, 2, figsize=(12,6), dpi=80)

    meds = [np.median(err_R[:,0]), np.median(err_t[:,0])]
    for i, errors in enumerate([err_R, err_t]):
        ax = axi[i]
        for label in (ax.get_xticklabels() + ax.get_yticklabels()):
                label.set_fontsize(16)
        med = meds[i]
        ax.hist(errors[:, 0], bins=50, range=(0, 20), alpha=0.8)
        ax.vlines(med, 0, 50, colors='red', label='50th perc.')
        ax.legend(prop={"size":15})
    
    fig.suptitle(f'Step {step_num}\nMedian errors: ( {meds[1]:.1f} m, {meds[0]:.1f}° )', fontsize=18)
    plt.tight_layout()
    plt.savefig(join(save_dir, f_name), bbox_inches='tight', pad_inches=0.0)
    plt.close()    


def plot_scores(scores, out_dir):
    threshs = list(zip(threshs_t, threshs_R))
    
    steps = np.array(scores['steps'])    
    fig, axis = plt.subplots(1, 2, figsize=(14, 7), dpi=100)
    x=list(range(len(steps)))

    for i, idx in enumerate([1, 2]):
        ax = axis[i]

        ax.plot(x, steps[:, 0, idx], label=f'Recall')
        ax.plot(x, steps[:, 2, idx], label=f'Upper bound')
        ax.hlines(scores['baseline'][idx], x[0], x[-1], colors='red', linestyles='dashed', label='baseline')
        ax.set_title(f'Threshold= {threshs[idx]}')
        ax.legend()
    plt.tight_layout()
    plt.savefig(f'{out_dir}/scores_plot.png')
