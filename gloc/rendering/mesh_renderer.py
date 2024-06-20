import PIL
import logging
import numpy as np
import open3d as o3d
from os.path import join

from gloc.rendering.base_renderer import BaseRenderer


class MeshRenderer(BaseRenderer):
    def __init__(self, conf):
        super().__init__(conf)
        self.mesh_path = conf.mesh_path
        self.background = 'black'
        self.supports_deferred_rendering = False
        logging.info(f'Using mesh from {self.mesh_path}')

    # override
    def load_model(self):
        mesh = o3d.io.read_triangle_model(self.mesh_path, False)

        for iter in range(len(mesh.materials)):
            mesh.materials[iter].shader = "defaultLit"
            # mesh.materials[iter].shader = "defaultUnlit"
            
            # - the original colors make the textures too dark - set to white
            mesh.materials[iter].base_color = [1.0, 1.0, 1.0, 1.0]

        return mesh
    
    # override
    def render_poses(self, out_dir, model, r_names, render_ts, render_qvecs, pose_list, wh, **kwargs):
        w, h = wh
        
        renderer = o3d.visualization.rendering.OffscreenRenderer(w, h)
        renderer.scene.add_model("Scene mesh", model)
        # - setup lighting
        renderer.scene.scene.enable_sun_light(True)
        if self.background == 'black':
            renderer.scene.set_background([0., 0., 0.,0.])

        for i, fname in enumerate(r_names):
            output_path = join(out_dir, f"{fname}.png")
            T, K = pose_list[i]

            renderer.setup_camera(K, T, w, h)        
            color = np.array(renderer.render_to_image())

            img_rendering = PIL.Image.fromarray(color)
            img_rendering.save(output_path)
        
        del renderer
