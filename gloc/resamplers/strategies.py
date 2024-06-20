import numpy as np

from gloc.utils import rotmat2qvec


class BaseProtocol:
    """This base dummy class serves as template for subclasses. it always returns
    the same poses without perturbing them"""
    def __init__(self, conf, sampler, scaler, protocol_name):
        self.sampler = sampler
        self.scaler = scaler
        self.n_steps = conf.N_steps
        self.n_views = conf.n_views
        self.protocol = protocol_name
        # init for later
        self.center_std = None
        self.max_angle = None
        
    def init_step(self, i):
        self.scaler.step(i)
        self.center_std, self.max_angle = self.scaler.get_noise()
    
    def get_pertubr_str(self, step, res):
        c_str = "_".join(list(map(lambda x: f'{x:.1f}'.replace('.', ','), map(float, self.center_std))))
        angle_str = "_".join(list(map(lambda x: f'{x:.1f}'.replace('.', ','), map(float, self.max_angle))))

        perturb_str = f'pt{self.protocol}_s{step}_sz{res}_theta{angle_str}_t{c_str}'
        return perturb_str
    
    @staticmethod
    def get_r_name(q_name, r_i, beam_i):
        r_name = q_name+f'_{r_i}beam{beam_i}'
        return r_name
    
    def resample(self, K, q_name, pred_t, pred_R, beam_i=0, *args, **kwargs):
        # this base class returns the same pose all over again
        render_qvecs = []
        render_ts = []
        calibr_pose = []
        r_names = []
        
        for i in range(self.n_views):
            t = pred_t[i]  
            R = pred_R[i]  
            qvec = rotmat2qvec(R) 

            render_qvecs.append(qvec)
            render_ts.append(t)
            r_names.append(BaseProtocol.get_r_name(q_name, i, beam_i))
            T = np.eye(4)
            T[0:3, 0:3] = R
            T[0:3, 3] = t
            calibr_pose.append((T, K))
        
        return r_names, render_ts, render_qvecs, calibr_pose


class Protocol1(BaseProtocol):
    """
    This protocol keeps only the first prediction, to perturb N times
    """
    def __init__(self, conf, scaler, sampler, protocol_name):
        super().__init__(conf, scaler, sampler, protocol_name)
    
    # override
    def resample(self, K, q_name, pred_t, pred_R, beam_i=0, *args, **kwargs):
        render_qvecs = []
        render_ts = []
        calibr_pose = []
        r_names = []
        
        t = pred_t[0]  # take first prediction
        R = pred_R[0]  # take first prediction
        qvec = rotmat2qvec(R) # transform to qvec

        #### keep previous estimate #####
        render_qvecs.append(qvec)
        render_ts.append(t)
        r_names.append(BaseProtocol.get_r_name(q_name, 0, beam_i))
        T = np.eye(4)
        T[0:3, 0:3] = R
        T[0:3, 3] = t
        calibr_pose.append((T, K))
        ####################
        views_per_candidate = self.n_views - 1
        new_ts, new_qvecs, new_poses = self.sampler.sample_batch(views_per_candidate, 
                                            self.center_std, self.max_angle, 
                                            t, R)
        render_ts += new_ts
        render_qvecs += new_qvecs
        for j in range(views_per_candidate):
            r_name = BaseProtocol.get_r_name(q_name, j + 1, beam_i)            
            r_names.append(r_name)
            calibr_pose.append((new_poses[j], K))

        return r_names, render_ts, render_qvecs, calibr_pose

    
class Protocol2(BaseProtocol):
    """  
    This protocol keeps the first M predictions, perturbing them N // M times
    """
    def __init__(self, conf, scaler, sampler, protocol_name):
        super().__init__(conf, scaler, sampler, protocol_name)
        self.M = conf.M_candidates
        
    # override
    def resample(self, K, q_name, pred_t, pred_R, beam_i=0, *args, **kwargs):
        render_qvecs = []
        render_ts = []
        calibr_pose = []
        r_names = []
        
        #### keep previous first M estimates #####
        for i in range(self.M):
            t = pred_t[i]  
            R = pred_R[i]  
            qvec = rotmat2qvec(R) 

            render_qvecs.append(qvec)
            render_ts.append(t)
            r_names.append(BaseProtocol.get_r_name(q_name, i, beam_i))
            T = np.eye(4)
            T[0:3, 0:3] = R
            T[0:3, 3] = t
            calibr_pose.append((T, K))
        ####################

        views_per_candidate = self.n_views // self.M - 1
        for i in range(self.M):
            t = pred_t[i]  
            R = pred_R[i]  

            new_ts, new_qvecs, new_poses = self.sampler.sample_batch(views_per_candidate, self.center_std, self.max_angle, 
                                                                     t, R)
            render_ts += new_ts
            render_qvecs += new_qvecs
            for j in range(views_per_candidate):
                r_name = BaseProtocol.get_r_name(q_name, self.M+i*views_per_candidate+j, beam_i)
                r_names.append(r_name)
                calibr_pose.append((new_poses[j], K))
        return r_names, render_ts, render_qvecs, calibr_pose
