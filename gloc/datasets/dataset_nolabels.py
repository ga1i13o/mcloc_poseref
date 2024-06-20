import os
import logging
import numpy as np
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as T

from gloc.utils import read_model_nopoints as read_model, parse_cam_model, qvec2rotmat
from gloc.utils.camera_utils import read_cameras_intrinsics, Image as RImage


class IntrinsicsDataset(data.Dataset):
    def __init__(self, name, paths_conf, transform=None):
        self.name = name
        self.root = paths_conf['root']
        self.colmap_model = paths_conf['colmap']
        self.q_files = paths_conf['q_intrinsics']
        self.transform = transform

        self.db_images, self.db_intrinsics = IntrinsicsDataset.load_colmap(self.colmap_model)
        self.q_list, self.q_intrinsics = IntrinsicsDataset.load_queries(self.q_files)
        self.intrinsics = self.db_intrinsics.copy()
        self.intrinsics.update(self.q_intrinsics)
        
        self.images = self.db_images.copy()
        for q in self.q_list:                
            im = RImage(id=-1, qvec=-1, tvec=-1, name=q.id, camera_id='', xys={}, point3D_ids={})
            self.images.append(im)

        self.db_frames_idxs = list(range(len(self.db_images)))
        self.q_frames_idxs = list(np.arange(len(self.q_list)) + len(self.db_images))
        self.db_qvecs = np.array(list(map(lambda x:x.qvec, self.db_images)))
        self.db_tvecs = np.array(list(map(lambda x:x.tvec, self.db_images)))

        self.n_db = len(self.db_images)
        self.n_q = len(self.q_list)
        logging.info(f'Loaded dataset with {self.n_db} db images and {self.n_q} queries w/ intrinsics')
        
    def get_basename(self, im_idx):
        q_name = self.images[im_idx].name.replace('/', '_').split('.')[0]
        return q_name
    
    def get_pose(self, idx):
        assert idx < self.n_db, f'Only db images have extrinsics, {idx} is a query'
        R = qvec2rotmat(self.images[idx].qvec)
        t = self.images[idx].tvec
        
        return t, R
    
    def num_queries(self):
        return self.n_q
    
    def get_db_poses(self):
        Rs = []
        ts = []

        for q_idx in range(len(self.db_frames_idxs)):
            idx = self.db_frames_idxs[q_idx]
            
            R = qvec2rotmat(self.images[idx].qvec)
            t = self.images[idx].tvec
            Rs.append(R)
            ts.append(t)
            
        return np.array(ts), np.array(Rs)
    
    def __getitem__(self, idx):
        """Return:
           dict:'im' is the image tensor
                'xyz' is the absolute position of the image
                'wpqr' is the  absolute rotation quaternion of the image
        """
        data_dict = {}
        im_data = self.images[idx]
        data_dict['im_ref'] = im_data
        im = Image.open(os.path.join(self.root, im_data.name))
        if self.transform:
            im = self.transform(im)
        data_dict['im'] = im        

        return data_dict
    
    def __len__(self):
        return len(self.images)

    def get_pose_by_name(self, name):
        names =np.array(list(map(lambda x: x.name, self.images)))
        idx = np.argwhere(name == names)[0,0]
        qvec, tvec = self.images[idx].qvec, self.images[idx].tvec

        return qvec, tvec
        
    @staticmethod
    def load_queries(q_file_list):
        q_intrinsincs = {}
        q_list = []
        for q_file in q_file_list:
            data, intrinsics_dict = IntrinsicsDataset.load_camera_intrinsics(q_file)
            q_list += data
            q_intrinsincs.update(intrinsics_dict)
        
        return q_list, q_intrinsincs
            
    @staticmethod
    def load_camera_intrinsics(intrinsic_file):
        # Load the images
        logging.info(f'Loading intrinsics from {intrinsic_file}')
        cam_list = {}
        cameras = read_cameras_intrinsics(intrinsic_file)
        for cam_data in cameras:
            cam_dict = parse_cam_model(cam_data)
            K = np.array([
                [cam_dict["fx"], 0.0, cam_dict["cx"]],
                [0.0, cam_dict["fy"], cam_dict["cy"]],
                [0.0, 0.0, 1.0]])
            w, h = cam_dict["width"], cam_dict["height"]

            basename = os.path.splitext(cam_data.id)[0]
            cam_list[basename] = {'K':K, 'T':T, 'w':w, 'h':h, 
                                  'model': cam_data.model, 'params':cam_data.params}

        return cameras, cam_list
        
    @staticmethod
    def load_colmap(colmap_model):
        # Load the images
        logging.info(f'Loading colmap from {colmap_model}')
        cam_list = {}
        cameras, images = read_model(colmap_model)
        for i in images:
            qvec = images[i].qvec
            tvec = images[i].tvec
            cam_data = cameras[images[i].camera_id]
            cam_dict = parse_cam_model(cam_data)

            K = np.array([
                [cam_dict["fx"], 0.0, cam_dict["cx"]],
                [0.0, cam_dict["fy"], cam_dict["cy"]],
                [0.0, 0.0, 1.0]])

            R = qvec2rotmat(qvec)
            T = np.eye(4)
            T[0:3, 0:3] = R
            T[0:3, 3] = tvec
            w, h = cam_dict["width"], cam_dict["height"]

            basename = os.path.splitext(images[i].name)[0]

            cam_list[basename] = {'K':K, 'T':T, 'w':w, 'h':h}
        return list(images.values()), cam_list
