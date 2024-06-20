import os
import logging
import numpy as np
from os.path import join
import torch.utils.data as data
from PIL import Image
import torchvision.transforms as T

from gloc.utils import read_model_nopoints as read_model, parse_cam_model, qvec2rotmat
from gloc.utils import Image as RImage


def get_query_id(name):
    if isinstance(name, tuple):
        name = name[0]
    name = name.split('/')[-1].split('.')[0]
    return name


def get_render_id(name):
    if isinstance(name, tuple):
        name = name[0]
    if ('night' in name) or ('day' in name):
        i = name.find('IMG')
        name = name[i:].split('_')[:-1]
        name = '_'.join(name)
        return name

    name = name.split('/')[-1].split('.')[0].split('_')[:-1]
    if len(name) > 2 and name[1] == 'nexus4':
        name = '_'.join(name[5:])
    elif len(name) > 2 and name[1] == 'gopro3':
        name = '_'.join(name[3:])
    else:
        name = name[1]
    return name


class PoseDataset(data.Dataset):
    def __init__(self, name, paths_conf, transform=None, rendered_db=None):
        self.name = name
        self.root = paths_conf['root']
        self.colmap_model = paths_conf['colmap']
        self.transform = transform
        self.rendered_db = rendered_db
        self.use_render = False

        self.images, self.intrinsics = PoseDataset.load_colmap(self.colmap_model)

        queries = paths_conf['q_file']
        db = paths_conf['db_file']
        if (queries != '') and (db != ''):
            all_frames = np.array(list(map(lambda x: x.name, self.images)))
            self.db_frames_idxs, self.db_tvecs, self.db_qvecs = self.load_txt(db, all_frames)
            self.q_frames_idxs, self.q_tvecs, self.q_qvecs = self.load_txt(queries, all_frames)
    
            all_frames = np.array(list(map(lambda x: x.name, self.images)))
            self.db_frames_idxs, self.db_tvecs, self.db_qvecs = self.load_txt(db, all_frames)
            self.q_frames_idxs, self.q_tvecs, self.q_qvecs = self.load_txt(queries, all_frames)
            self.n_q = len(self.q_frames_idxs)
        
    def load_txt(self, fpath, all_frames):
        with open(fpath, 'r') as f:
            lines = f.readlines()
        if lines[0].startswith('Visual Landmark'):
            lines = lines[3:]
        frames = np.array(list(map(lambda x: x.split(' ')[0].strip(), lines)))
        frames_idxs = list(np.where(np.in1d(all_frames, frames))[0])
        tvecs = np.array(list(map(lambda x: x.tvec, self.images)))[frames_idxs]
        qvecs = np.array(list(map(lambda x: x.qvec, self.images)))[frames_idxs]
        
        return frames_idxs, tvecs, qvecs
    
    def get_basename(self, im_idx):
        q_name = self.images[im_idx].name.replace('/', '_').split('.')[0]
        return q_name
    
    def get_pose(self, idx):
        R = qvec2rotmat(self.images[idx].qvec)
        t = self.images[idx].tvec
        
        return t, R

    def get_pose_by_name(self, name):
        names =np.array(list(map(lambda x: x.name, self.images)))
        idx = np.argwhere(name == names)[0,0]
        qvec, tvec = self.images[idx].qvec, self.images[idx].tvec

        return qvec, tvec

    def num_queries(self):
        return len(self.q_frames_idxs)

    def get_pose_by_name(self, name):
        names =np.array(list(map(lambda x: x.name, self.images)))
        idx = np.argwhere(name == names)[0,0]
        qvec, tvec = self.images[idx].qvec, self.images[idx].tvec

        return qvec, tvec

    def get_intrinsics(self, q_key_name):
        assert q_key_name in self.intrinsics, f'{q_key_name} is not a valid image name'

        w = self.intrinsics[q_key_name]['w']    
        h = self.intrinsics[q_key_name]['h']    
        K = self.intrinsics[q_key_name]['K']

        return K, w, h

    def get_q_poses(self):
        Rs = []
        ts = []

        for q_idx in range(len(self.q_frames_idxs)):
            idx = self.q_frames_idxs[q_idx]
            
            R = qvec2rotmat(self.images[idx].qvec)
            t = self.images[idx].tvec
            Rs.append(R)
            ts.append(t)
            
        return np.array(ts), np.array(Rs)
    
    def get_all_poses(self):
        Rs = []
        ts = []

        for image in self.images:
            R = qvec2rotmat(image.qvec)
            t = image.tvec
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
        if not self.use_render:
            im = Image.open(join(self.root, im_data.name))
        else:
            im = Image.open(join(self.rendered_db, im_data.name.replace('/', '_')).replace('.jpg', '.png'))
        if self.transform:
            im = self.transform(im)
        data_dict['im'] = im        

        return data_dict
    
    def __len__(self):
        return len(self.images)

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

            cam_list[basename] = {'K':K, 'T':T, 'w':w, 'h':h, 'model': cam_data.model}
        return list(images.values()), cam_list
    

class RenderedImagesDataset(data.Dataset):
    def __init__(self, im_root, transform=None, query_res=None, verbose=True):
        self.root = im_root
        self.descr_file = os.path.join(im_root, 'rendered_views.txt')

        self.images = RenderedImagesDataset.load_images(self.descr_file, verbose)
        self.final_resize = None
        if query_res is not None:
            # this to ensure that renderings end up having same resolution
            # as the query once resized
            self.final_resize = T.Resize(query_res, antialias=True)

        self.transform = transform
        
    def __getitem__(self, idx):
        """Return:
           dict:'im' is the image tensor
                'xyz' is the absolute position of the image
                'wpqr' is the  absolute rotation quaternion of the image
        """
        data_dict = {}
        im_data = self.images[idx]
        data_dict['im_ref'] = im_data
        im = Image.open(join(self.root, im_data.name))

        if self.transform:
            im = self.transform(im)
            if self.final_resize:
                im = self.final_resize(im)
        data_dict['im'] = im        

        return data_dict
    
    def __len__(self):
        return len(self.images)

    def get_full_paths(self):
        paths = list(map(lambda x: join(self.root, x.name), self.images))
        return paths

    def get_names(self):
        image_names = list(map(lambda x: x.name, self.images))
        image_names = np.array(image_names)

        return image_names
    
    def get_camera_centers(self):
        centers = []
        for im in self.images:
            R = qvec2rotmat(im.qvec)
            tvec = im.tvec
            im_center = - R.T @ tvec
            centers.append(im_center)
        centers = np.array(centers)

        return centers
        
    def get_poses(self):
        Rs = []
        ts = []

        for image in self.images:
            R = qvec2rotmat(image.qvec)
            t = image.tvec
            Rs.append(R)
            ts.append(t)
            
        return np.array(ts), np.array(Rs)

    @staticmethod
    def load_images(fpath, verbose):
        # Load the images
        if verbose:
            print('Loading the rendered and cameras')
        images = []
        with open(fpath, 'r') as rv:
            while True:
                line = rv.readline()
                if not line:
                    break
                line = line.strip()
                fields = line.split(' ')
                
                name = fields[0]+'.png'
                tvec = np.array(tuple(map(float, fields[1:4])))
                qvec = np.array(tuple(map(float, fields[4:8])))
                
                im = RImage(id=-1, qvec=qvec, tvec=tvec, name=name, 
                            camera_id='', xys={}, point3D_ids={})
                images.append(im)
        return images
            