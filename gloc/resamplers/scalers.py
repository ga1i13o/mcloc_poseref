class ConstantScaler():
    def __init__(self, conf):
        self.max_angle = conf.max_angle
        self.center_std = conf.max_center_std

    def step(self, i):
        pass
    
    def get_noise(self):
        return self.center_std, self.max_angle
    
    def get_max_noise(self, multiplier=1):
        return self.center_std*multiplier, self.max_angle*multiplier

    
class UniformScaler():
    def __init__(self, conf):
        # gamma is the minimum multiplier that will be applied
        self.max_angle = conf.max_angle
        self.max_center_std = conf.max_center_std
        self.current_angle = conf.max_angle
        self.current_center_std = conf.max_center_std
        
        self.n_steps = conf.N_steps
        self.gamma = conf.gamma
        
    def step(self, i):
        scale_noise = max(self.gamma, (self.n_steps - i)/self.n_steps)
        
        self.current_center_std = self.max_center_std*scale_noise 
        self.current_angle = self.max_angle* scale_noise        

    def get_noise(self):
        return self.current_center_std, self.current_angle
    
    def get_max_noise(self, multiplier=1):
        return self.max_center_std*multiplier, self.max_angle*multiplier
