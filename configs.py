conf = {
    'cambridge': [
        {
            # Set N.1
            'beams': 2,
            'steps': 40,
            'N': 52,
            'M': 2,
            'feat_model': 'cosplace_r18_l3',
            'protocol': '2_1',
            'center_std': [1.2, 1.2, 0.1],
            'teta': [10],
            'gamma': 0.3,
            'res': 320,
            'colmap_res': 320,
        },
        {
            # Set N.2
            'beams': 2,
            'steps': 30,
            'N': 40,
            'M': 2,
            'feat_model': 'cosplace_r18_l2',
            'protocol': '2_1',
            'center_std': [.4, .4, .04],
            'teta': [4],
            'gamma': 0.3,
            'res': 320,
            'colmap_res': 320,
        },        
        {
            # Set N.3
            'beams': 2,
            'steps': 60,
            'N': 32,
            'M': 1,
            'feat_model': 'cosplace_r18_l2',
            'protocol': '2_0',
            'center_std': [.15, .15, 0.02],
            'teta': [1.5],
            'gamma': 0.3,
            'res': 480,
            'colmap_res': 480,
        },        
    ]
}


def get_config(ds_name):
    cambridge_scenes = [
        'StMarysChurch', 'OldHospital', 'KingsCollege', 'ShopFacade'
    ]
    
    if ds_name in cambridge_scenes:
        return conf['cambridge']
    else:
        return NotImplementedError
