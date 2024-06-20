import numpy as np
from pyquaternion import Quaternion

from gloc.utils import qvec2rotmat


def gen_translations(border_points, radius, points_per_meter, q_center, axis_up):
    theta = np.linspace(0, 2 * np.pi, border_points, endpoint=False)

    x = np.cos(theta)*radius
    if axis_up == 'y':
        y = np.zeros(x.shape)
        z = np.sin(theta)*radius
    elif axis_up == 'z':
        y = np.sin(theta)*radius
        z = np.zeros(x.shape)
    else:
        raise NotImplementedError()
            
    points = np.array([x, y, z]).transpose()
    n_points = int(radius * points_per_meter+2)
    for point in points:
        x0 = np.linspace(0, point[0], n_points)[1:-1]
        if axis_up == 'y':
            y0 = np.zeros(x0.shape)
            z0 = np.linspace(0, point[2], n_points)[1:-1]
        elif axis_up == 'z':
            y0 = np.linspace(0, point[1], n_points)[1:-1]
            z0 = np.zeros(x0.shape)
        else:
            raise NotImplementedError()

        new_p=np.stack((x0, y0, z0)).transpose()
        points = np.vstack((points, new_p))

    points += q_center
    points = np.insert(points, 0, q_center, axis=0)
    camera_centers = list(points)
    return camera_centers


def gen_rotations(qvec, R, tvec, c_center, points_per_axis, max_angle, n_axis):
    axis_set = [
        [1, 0, 0],
        [0, 1, 0],
        [1, -1, 0],        
        [1, 1, 0],        
    ]

    gen_poses = [(qvec, R, tvec)]
    for axis in axis_set[:n_axis]:
        theta = np.linspace(-max_angle, max_angle, points_per_axis)
        theta = np.delete(theta, points_per_axis // 2)
        
        for th in theta:
            my_quaternion = Quaternion(axis=axis, angle=th)
            new_qvec = my_quaternion * qvec
            new_R = qvec2rotmat(new_qvec)
            new_t = - new_R @ c_center
            gen_poses.append((new_qvec, new_R, new_t))
    return gen_poses


def parse_pose_data(q_basename, gen_poses, K, sub_index):
    render_qvecs = []
    render_ts = []
    calibr_pose = []
    r_names = []
    for i, pose in enumerate(gen_poses):
        new_qvec, new_R, new_t = pose
        T = np.eye(4)
        T[0:3, 0:3] = new_R
        T[0:3, 3] = new_t
        
        render_qvecs.append(new_qvec)
        render_ts.append(new_t)
        r_names.append(q_basename + f'_{sub_index}' + f'_{i}')
        calibr_pose.append((T, K))    
    
    return render_qvecs, render_ts, r_names, calibr_pose
