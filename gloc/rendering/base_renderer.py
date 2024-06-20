class BaseRenderer:
    def __init__(self, conf):
        pass
    
    def load_model(self):
        return None
    
    def render_poses(self, out_dir, model, r_names, render_ts, render_qvecs, pose_list, wh):
        pass

    def end_epoch(self, step_dir):
        pass
    
    @staticmethod
    def clean_file_names(r_dir, r_names, verbose):
        pass
