from os.path import join


def get_paths(base_path, colmap_res, mesh_type):
    aachen_meshes = {
        'colored': 'AC13_colored.ply',
        'colored_14': 'AC14_colored.ply',
        'colored_15': 'AC15_colored.ply',
        'textured': 'AC13-C_textured/AC13-C_textured.obj',
    }
    
    paths_conf = {
        # Aachen 
        'Aachen': {
            'root': f'{base_path}/Aachen-Day-Night/images/images_upright',
            'colmap': f'{base_path}/all_colmaps/{colmap_res}_undist',
            'q_file': '',
            'db_file': '',
            'q_intrinsics': [f'{base_path}/Aachen-Day-Night/queries/undist_rs{colmap_res}_allquery_intrinsics.txt'],
            'mesh_path': f'{base_path}/meshes',
        },
    }

    # Cambridge scenes
    cambridge_scenes = ['StMarysChurch', 'OldHospital', 'KingsCollege', 'ShopFacade']
    gs_models = {
        'StMarysChurch': 'gauss_church', 
        'OldHospital': 'gauss_hosp', 
        'KingsCollege': 'gauss_kings', 
        'ShopFacade': 'gauss_shop',
    }
    for cs in cambridge_scenes:
        paths_conf[cs] = {
            'root': f'{base_path}/{cs}',
            'colmap': f'{base_path}/all_colmaps/{cs}/{colmap_res}_undist/sparse/0',
            'q_file': f'{base_path}/{cs}/dataset_test.txt',
            'db_file': f'{base_path}/{cs}/dataset_train.txt',
            'mesh_path': NotImplemented,
            'ns_config': NotImplemented,
            'ns_transform': NotImplemented,
            'gaussians': f'{base_path}/cambridge_splats/{gs_models[cs]}',
        }

    # 7scenes
    scenes7 = ['chess', 'office', 'fire', 'stairs', 'redkitchen', 'pumpkin', 'heads']
    for sc in scenes7:
        paths_conf[sc] = {
            'root': f'{base_path}/7scenes/{sc}',
            'colmap': f'{base_path}/all_colmaps/{sc}/colmap_{colmap_res}',
            'q_file': f'{base_path}/7scenes/{sc}/test.txt',
            'db_file': f'{base_path}/7scenes/{sc}/train.txt',
            'mesh_path': NotImplemented,
            'ns_config': NotImplemented,
            'ns_transform': NotImplemented,
            'gaussians': f'{base_path}/7scenes_splats/{sc}',
        }
    ######################################
                
    paths_conf['Aachen']['mesh_path'] = join(paths_conf['Aachen']['mesh_path'], aachen_meshes.get(mesh_type, ''))
        
    return paths_conf


def get_path_conf(colmap_res, mesh_type):
    base_path = 'data'
    temp_path = 'data/temp'
    
    conf = get_paths(base_path, colmap_res, mesh_type)
    conf['temp'] = temp_path
    return conf
