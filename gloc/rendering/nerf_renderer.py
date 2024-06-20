import os
import json
import shutil
import numpy as np
from os.path import join
from pathlib import Path
import mediapy as media
import torch
import logging
from typing import List

from nerfstudio.cameras.cameras import Cameras
from nerfstudio.pipelines.base_pipeline import Pipeline
from nerfstudio.utils import colormaps
from nerfstudio.utils.eval_utils import eval_setup
from nerfstudio.cameras.camera_paths import get_path_from_json
from nerfstudio.utils import colormaps

from gloc.rendering.base_renderer import BaseRenderer
from gloc.utils import get_c2w_nerfconv


class NerfRenderer(BaseRenderer):
    def __init__(self, conf):
        super().__init__(conf)
        self.ns_config = conf.ns_config
        self.ns_transform = conf.ns_transform
        self.supports_deferred_rendering = False
        logging.info(f'Using nerf from {self.ns_config}')

    # override
    def load_model(self):
        _, pipeline, _, _ = eval_setup(
            Path(self.ns_config),
            eval_num_rays_per_chunk=None,
            test_mode="inference",
        )
        
        return pipeline

    # override
    def render_poses(self, out_dir, model, r_names, render_ts, render_qvecs, pose_list, wh, deferred=False):
        # for refinement experiments, 
        # the K matrix is the same throughout the pose list, so take the 1st
        K = pose_list[0][1]
        traj_file = self._gen_trajectory_file(out_dir, wh, K, r_names, render_ts, render_qvecs)
        
        with open(traj_file, "r", encoding="utf-8") as f:
            camera_path = json.load(f)
        camera_path = get_path_from_json(camera_path)

        output_path = Path(out_dir)
        rendered_output_names = ['rgb']
        self._render_trajectory(
            model,
            camera_path,
            output_filename=output_path,
            rendered_output_names=rendered_output_names,
        )

    def _gen_trajectory_file(self, out_dir, wh, K, r_names, render_ts, render_qvecs):
        width, height = wh
        f_len = K[0,0]
        traj_file = join(out_dir, 'trajectory.json')
        
        out = {
            'camera_path': []
        }
        out["render_height"] = height
        out["render_width"] = width
        out["seconds"] = len(r_names)

        # load the transformation from dataparser_transforms.json
        with open(self.ns_transform) as f:
            json_data_txt = f.read()
        json_data = json.loads(json_data_txt)

        T_dp = np.eye(4)
        T_dp[0:3,:] = np.array(json_data["transform"])
        s_dp = json_data["scale"]

        for i in range(len(r_names)):
            cam_dict = {}
            name = r_names[i]
            tvec, qvec = render_ts[i], render_qvecs[i]
            
            # get camera to world matrix, in nerf convention
            c2w = get_c2w_nerfconv(qvec, tvec)
            # apply dataparser transform
            c2w = T_dp @ c2w
            c2w[0:3, 3] = s_dp * c2w[0:3, 3]

            cam_dict['camera_to_world'] = c2w.tolist()
            cam_dict['fov'] = np.degrees(2*np.arctan2(height, 2*f_len))
            cam_dict['file_path'] = name
            
            out['camera_path'].append(cam_dict)
            
        with open(traj_file, 'w') as f:
            json.dump(out, f, indent=4)
        
        return traj_file

    @staticmethod
    def _render_trajectory(
        pipeline: Pipeline,
        cameras: Cameras,
        output_filename: Path,
        rendered_output_names: List[str],
        colormap_options: colormaps.ColormapOptions = colormaps.ColormapOptions(),
    ) -> None:
        """Helper function to create a video of the spiral trajectory.

        Args:
            pipeline: Pipeline to evaluate with.
            cameras: Cameras to render.
            output_filename: Name of the output file.
            rendered_output_names: List of outputs to visualise.
            colormap_options: Options for colormap.
        """
        cameras = cameras.to(pipeline.device)
        output_image_dir = output_filename.parent / output_filename.stem
        for camera_idx in range(cameras.size):
            aabb_box = None
            camera_ray_bundle = cameras.generate_rays(camera_indices=camera_idx, aabb_box=aabb_box)

            with torch.no_grad():
                outputs = pipeline.model.get_outputs_for_camera_ray_bundle(camera_ray_bundle)

            render_image = []
            for rendered_output_name in rendered_output_names:
                output_image = outputs[rendered_output_name]
                
                output_image = colormaps.apply_colormap(
                    image=output_image,
                    colormap_options=colormap_options,
                ).cpu().numpy()
                
                render_image.append(output_image)
            render_image = np.concatenate(render_image, axis=1)

            media.write_image(output_image_dir / f"{camera_idx:05d}.png", render_image, fmt="png")
    
    @staticmethod
    def clean_file_names(r_dir, r_names, verbose=False):
        if verbose:
            print('Changing filenames format...')
            
        f_names = sorted(filter(lambda x: x.endswith('.png'), os.listdir(r_dir)))
        for i, f_name in enumerate(f_names):
            src = join(r_dir, f_name)
            dst = join(r_dir, r_names[i]+'.png')
            shutil.move(src, dst)
