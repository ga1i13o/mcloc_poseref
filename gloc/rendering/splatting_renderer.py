import logging
from os.path import join
import os
import sys
from pathlib import Path
from argparse import Namespace
import torch
import torchvision
import numpy as np
from tqdm import tqdm
import shutil
from glob import glob
from typing import List

from gloc.utils import camera_utils, Image as RImage
from gloc.rendering.base_renderer import BaseRenderer

sys.path.append(str(Path(__file__).parent.parent.parent.joinpath('third_party/gaussian-splatting')))

from scene import Scene
from gaussian_renderer import render
from gaussian_renderer import GaussianModel


class GaussianSplattingRenderer(BaseRenderer):
    def __init__(self, conf):
        super().__init__(conf)
        self.pipeline = Namespace(convert_SHs_python=False, compute_cov3D_python=False, debug=False)
        self.gaussians = conf.gaussians
        self.sh_degree = 3
        self.supports_deferred_rendering = True
        self.iteration = 7000
        self.buf_deferred = self.init_buffer()
        logging.info(f'Using Gaussians from from {self.gaussians}')
        self.dataset = Namespace(data_device= 'cuda', eval= False, images='dont', 
                        model_path=self.gaussians, resolution='dont', sh_degree= 3, 
                        source_path='', white_background=False)

    # override
    def load_model(self):
        gaussians = GaussianModel(self.sh_degree)
        return gaussians

    # override
    def render_poses(self, out_dir, model, r_names, render_ts, render_qvecs, pose_list, wh, deferred=True):
        # for refinement experiments, 
        # the K matrix is the same throughout the pose list, so take the 1st
        K = pose_list[0][1]
        if deferred:
            self.update_buffer(wh, K, r_names, render_ts, render_qvecs)
        else:
            if not isinstance(wh, list):
                # wh, r_names, render_ts, render_qvecs = [wh], [r_names], [render_ts], [render_qvecs]
                wh = [wh]
                 
            # BEFORE it was only THIS
            colmap_dir = self._gen_colmap(out_dir, wh, [K], [r_names], [render_ts], [render_qvecs])
            self.dataset.source_path = colmap_dir
            scene = Scene(self.dataset, model, load_iteration=self.iteration, shuffle=False)
            bg_color = [0, 0, 0]
            
            with torch.no_grad():
                background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
                views = scene.getTrainCameras()
                for view in tqdm(views, ncols=100):
                    rendering = render(view, model, self.pipeline, background)["render"]
                    torchvision.utils.save_image(rendering, join(out_dir, view.image_name+'.png'))

    # override
    def end_epoch(self, step_dir):
        if self.is_buffer_empty():
            return
        
        model = GaussianModel(self.sh_degree)

        wh_l, K_l, r_names_l, render_ts_l, render_qvecs_l = self.read_buffer()
        colmap_dir = self._gen_colmap(step_dir, wh_l, K_l, r_names_l, render_ts_l, render_qvecs_l)
        self.dataset.source_path = colmap_dir
        scene = Scene(self.dataset, model, load_iteration=self.iteration, shuffle=False)
        bg_color = [0, 0, 0]
        
        with torch.no_grad():
            background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")
            views = scene.getTrainCameras()
            for view in tqdm(views, ncols=100):
                rendering = render(view, model, self.pipeline, background)["render"]
                torchvision.utils.save_image(rendering, join(step_dir, view.image_name+'.png'))

        # clear buffer
        self.buf_deferred = self.init_buffer()
        print(f'Sorting out the step dir....')
        self.sort_step_dir(step_dir)
    
    def sort_step_dir(self, step_dir):
        # put all the renders inside their query directory
        all_images = glob(join(step_dir, '*.png'))
        im_per_query = {}
        for im in all_images:
            r_name = im.split('.png')[0]
            end_name = r_name.rfind('_')
            q_name = r_name[:end_name]
            q_name = os.path.basename(q_name)
            if q_name not in im_per_query:
                im_per_query[q_name] = [r_name]
            else:
                im_per_query[q_name].append(r_name)
        
        for q_name in tqdm(im_per_query, ncols=100):
            q_renders = im_per_query[q_name]
            os.makedirs(join(step_dir, q_name), exist_ok=True)
            for q_r in q_renders:
                rbg_r = q_r + '.png'
                shutil.move(rbg_r, join(step_dir, q_name))
        
    def is_buffer_empty(self):
        return (len(self.buf_deferred['K']) == 0)
    
    def read_buffer(self):
        bf = (self.buf_deferred['wh'], self.buf_deferred['K'], self.buf_deferred['r_names'], 
              self.buf_deferred['render_ts'], self.buf_deferred['render_qvecs'])
        return bf
        
    def update_buffer(self, wh, K, r_names, render_ts, render_qvecs):
        self.buf_deferred['wh'].append(wh)
        self.buf_deferred['K'].append(K)
        self.buf_deferred['r_names'].append(r_names)
        self.buf_deferred['render_ts'].append(render_ts)
        self.buf_deferred['render_qvecs'].append(render_qvecs)
        
    @staticmethod
    def init_buffer():
        buffer = {
            'wh': [], 
            'K': [], 
            'r_names': [], 
            'render_ts': [], 
            'render_qvecs': []
        }
        return buffer        
        
    @staticmethod
    def _gen_colmap(out_dir: str, wh_l: List[tuple], K_l: List[np.array], 
                    r_names_l: List[str], render_ts_l: List[np.array], render_qvecs_l: List[np.array]):
        out_cameras = {}
        out_images = {}
        # print(type(wh_l), type(wh_l[0]), len(wh_l))
        # print(len(r_names_l), len(render_ts_l), len(render_qvecs_l))
        n_cameras = len(K_l)
        im_per_camera = len(r_names_l[0])
        # print(n_cameras)
        # print(K_l)
        for c_id in range(n_cameras):
            wh, K, r_names, render_ts, render_qvecs = wh_l[c_id], K_l[c_id], r_names_l[c_id], render_ts_l[c_id], render_qvecs_l[c_id]
            width, height = wh
            # assumes PINHOLE model
            model = 'PINHOLE'
            fx, fy = K[0,0], K[1,1]
            cx, cy = K[0,2], K[1,2]
            params = np.array([fx, fy, cx, cy])
            c = camera_utils.Camera(id=c_id, model=model, width=width, height=height, params=params)
            out_cameras[c_id] = c

            for r_id in range(len(r_names)):
                name = r_names[r_id]
                im_id = c_id*im_per_camera + r_id
                im = RImage(id=im_id, qvec=render_qvecs[r_id], 
                            tvec=render_ts[r_id], camera_id=c_id, 
                            name=name, xys={}, point3D_ids={})
                out_images[im_id] = im

        colmap_subdir = join(out_dir, 'sparse', '0')
        os.makedirs(colmap_subdir)
        camera_utils.write_model_nopoints(out_cameras, out_images, colmap_subdir, ext='.txt')        

        return out_dir
