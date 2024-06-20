from typing import List
import torch
import numpy as np
from pyquaternion import Quaternion

from gloc.utils import rotmat2qvec, qvec2rotmat


class RandomConstantSampler():
    def __init__(self, conf):
        pass
    
    def sample_batch(self, n_views, center_noise, angle_noise, old_t, old_R):
        qvecs = []
        tvecs = []
        poses = []

        for _ in range(n_views):
            new_tvec, new_qvec, new_T = self.sample(center_noise, angle_noise, old_t, old_R)

            qvecs.append(new_qvec)
            tvecs.append(new_tvec)
            poses.append(new_T)

        return tvecs, qvecs, poses

    @staticmethod
    def sample(center_noise, angle_noise, old_t, old_R, low_std_ax=1):
        old_qvec = rotmat2qvec(old_R) # transform to qvec

        r_axis = Quaternion.random().axis # sample random axis
        teta = angle_noise # sample random angle smaller than theta
        r_quat = Quaternion(axis=r_axis, degrees=teta)
        new_qvec = r_quat * old_qvec  # perturb the original pose

        # convert from Quaternion to np.array
        new_qvec = new_qvec.elements 
        new_R = qvec2rotmat(new_qvec)

        old_center = - old_R.T @ old_t # get image center using original pose
        perturb_c = np.random.rand(2)
        perturb_low_ax = np.random.rand(1)*0.1
        perturb_c = np.insert(perturb_c, low_std_ax, perturb_low_ax)
        perturb_c /= np.linalg.norm(perturb_c) # normalize noise vector

        # move along the noise direction for a fixed magnitude
        new_center = old_center + perturb_c*center_noise  
        new_t = - new_R @ new_center # use the new pose to convert to translation vector

        new_T = np.eye(4)
        new_T[0:3, 0:3] = new_R
        new_T[0:3, 3] = new_t

        return new_t, new_qvec, new_T
    
    
class RandomGaussianSampler():
    def __init__(self, conf):
        pass
    
    def sample_batch(self, n_views, center_std, max_angle, old_t, old_R):
        qvecs = []
        tvecs = []
        poses = []

        for _ in range(n_views):
            new_tvec, new_qvec, new_T = self.sample(center_std, max_angle, old_t, old_R)

            qvecs.append(new_qvec)
            tvecs.append(new_tvec)
            poses.append(new_T)

        return tvecs, qvecs, poses

    @staticmethod
    def sample(center_std, max_angle, old_t, old_R):
        old_qvec = rotmat2qvec(old_R) # transform to qvec

        r_axis = Quaternion.random().axis # sample random axis
        teta = np.random.rand()*max_angle # sample random angle smaller than theta
        r_quat = Quaternion(axis=r_axis, degrees=teta)
        new_qvec = r_quat * old_qvec  # perturb the original pose

        # convert from Quaternion to np.array
        new_qvec = new_qvec.elements 
        new_R = qvec2rotmat(new_qvec)

        old_center = - old_R.T @ old_t # get image center using original pose
        perturb_c = torch.normal(0., center_std)
        new_center = old_center + np.array(perturb_c) # perturb it 
        new_t = - new_R @ new_center # use the new pose to convert to translation vector

        new_T = np.eye(4)
        new_T[0:3, 0:3] = new_R
        new_T[0:3, 3] = new_t

        return new_t, new_qvec, new_T


class RandomDoubleAxisSampler():
    rotate_axis = {
        'pitch': [1, 0, 0], # x, pitch
        'yaw':   [0, 1, 0]  # y, yaw
    }
    
    def __init__(self, conf):
        pass
    
    def sample_batch(self, n_views: int, center_std: torch.tensor, max_angle: torch.tensor, 
                     old_t: np.array, old_R: np.array):
        qvecs = []
        tvecs = []
        poses = []

        for _ in range(n_views):
            # apply yaw first
            ax = self.rotate_axis['yaw']            
            new_tvec, _, new_R, _ = self.sample(ax, center_std, float(max_angle[0]), old_t, old_R)

            # apply pitch then
            ax = self.rotate_axis['pitch']            
            new_tvec, new_qvec, _, new_T = self.sample(ax, center_std, float(max_angle[1]), new_tvec, new_R)

            qvecs.append(new_qvec)
            tvecs.append(new_tvec)
            poses.append(new_T)

        return tvecs, qvecs, poses

    @staticmethod
    def sample(axis, center_std: torch.tensor, max_angle: float,  old_t: np.array, old_R: np.array, rot_only: bool =False):
        old_qvec = rotmat2qvec(old_R) # transform to qvec

        teta = np.random.rand()*max_angle # sample random angle smaller than theta
        r_quat = Quaternion(axis=axis, degrees=teta)
        new_qvec = r_quat * old_qvec  # perturb the original pose

        # convert from Quaternion to np.array
        new_qvec = new_qvec.elements 
        new_R = qvec2rotmat(new_qvec)

        if not rot_only:
            old_center = - old_R.T @ old_t # get image center using original pose
            perturb_c = torch.normal(0., center_std)
            new_center = old_center + np.array(perturb_c) # perturb it 
            new_t = - new_R @ new_center # use the new pose to convert to translation vector
        else:
            new_t = - new_R @ old_center # use the new pose to convert to translation vector            
        
        new_T = np.eye(4)
        new_T[0:3, 0:3] = new_R
        new_T[0:3, 3] = new_t

        return new_t, new_qvec, new_R, new_T


class RandomSamplerByAxis():
    rotate_axis = [
        [1, 0, 0], # x, pitch
        [0, 1, 0] # y, yaw
    ]
    
    def __init__(self, conf):
        pass
    
    def sample_batch(self, n_views, center_std, max_angle, old_t, old_R):
        qvecs = []
        tvecs = []
        poses = []

        for i in range(n_views):
            # use first axis half the time, the other for the rest
            ax_i = i // ((n_views+1) // 2)
            ax = self.rotate_axis[ax_i]
            
            new_tvec, new_qvec, new_T = self.sample(ax, center_std, max_angle, old_t, old_R)

            qvecs.append(new_qvec)
            tvecs.append(new_tvec)
            poses.append(new_T)

        return tvecs, qvecs, poses

    @staticmethod
    def sample(axis, center_std, max_angle, old_t, old_R):
        old_qvec = rotmat2qvec(old_R) # transform to qvec

        teta = np.random.rand()*max_angle # sample random angle smaller than theta
        r_quat = Quaternion(axis=axis, degrees=teta)
        new_qvec = r_quat * old_qvec  # perturb the original pose

        # convert from Quaternion to np.array
        new_qvec = new_qvec.elements 
        new_R = qvec2rotmat(new_qvec)

        old_center = - old_R.T @ old_t # get image center using original pose
        perturb_c = torch.normal(0., center_std)
        new_center = old_center + np.array(perturb_c) # perturb it 
        new_t = - new_R @ new_center # use the new pose to convert to translation vector

        new_T = np.eye(4)
        new_T[0:3, 0:3] = new_R
        new_T[0:3, 3] = new_t

        return new_t, new_qvec, new_T


class RandomAndDoubleAxisSampler():
    rotate_axis = {
        'pitch': [1, 0, 0], # x, pitch
        'yaw':   [0, 1, 0]  # y, yaw
    }
    
    def __init__(self, conf):
        pass
    
    def sample_batch(self, n_views: int, center_std: torch.tensor, max_angle: torch.tensor, 
                     old_t: np.array, old_R: np.array):
        qvecs = []
        tvecs = []
        poses = []

        for i in range(n_views):
            # use DoubleRotation half the time, Random the rest
            ax_i = i // ((n_views+1) // 2)
            if ax_i == 0:
                # double ax rotation
                
                # apply yaw first
                ax = self.rotate_axis['yaw']            
                new_tvec, _, new_R, _ = self.sample(ax, center_std, float(max_angle[0]), old_t, old_R)

                # apply pitch then
                ax = self.rotate_axis['pitch']            
                new_tvec, new_qvec, _, new_T = self.sample(ax, center_std, float(max_angle[1]), new_tvec, new_R)

                qvecs.append(new_qvec)
                tvecs.append(new_tvec)
                poses.append(new_T)
            else:
                # use random axis, with yaw magnitude
                new_tvec, new_qvec, _, new_T = self.sample(None, center_std, float(max_angle[0]), old_t, old_R)

                qvecs.append(new_qvec)
                tvecs.append(new_tvec)
                poses.append(new_T)                

        return tvecs, qvecs, poses

    @staticmethod
    def sample(axis, center_std: torch.tensor, max_angle: float,  old_t: np.array, old_R: np.array, rot_only: bool =False):
        old_qvec = rotmat2qvec(old_R) # transform to qvec

        if axis is None:
            # if no axis provided, use a random one
            axis = Quaternion.random().axis # sample random axis

        teta = np.random.rand()*max_angle # sample random angle smaller than theta
        r_quat = Quaternion(axis=axis, degrees=teta)
        new_qvec = r_quat * old_qvec  # perturb the original pose

        # convert from Quaternion to np.array
        new_qvec = new_qvec.elements 
        new_R = qvec2rotmat(new_qvec)

        if not rot_only:
            old_center = - old_R.T @ old_t # get image center using original pose
            perturb_c = torch.normal(0., center_std)
            new_center = old_center + np.array(perturb_c) # perturb it 
            new_t = - new_R @ new_center # use the new pose to convert to translation vector
        else:
            new_t = - new_R @ old_center # use the new pose to convert to translation vector            
        
        new_T = np.eye(4)
        new_T[0:3, 0:3] = new_R
        new_T[0:3, 3] = new_t

        return new_t, new_qvec, new_R, new_T
