import numpy as np
import torch
import os
import json
from fitting.fit_utils import get_code
    
class Mesh:
    def __init__(self, vertex:np.ndarray, template_str='flame'):
        self.template = template_str # flame or biwi
        self.v = vertex

    # return: N*3 asarray
    def read_obj(obj_str):
        vert_list = []
        with open(obj_str, mode='r') as f:
            break_flag = False
            while not break_flag:
                vert_str = ''
                while('v ' not in vert_str):
                    vert_str = f.readline()
                    if len(vert_str) == 0:
                        break_flag = True
                        break
                if not break_flag:
                    vert_list.append([float(v) for v in vert_str.split(' ')[1:]])
        return np.asarray(vert_list) # N*3
    
    def write_obj(template:str, vertex:np.ndarray, out_path:str):
        append_f = None
        temp_path = os.path.split(__file__)[0]
        if template.lower() == 'flame':
            # load flame template
            with open(os.path.join(temp_path, 'template/FLAME_template.obj'), mode='r') as f:
                append_f = f.read(None)
        elif template.lower() == 'biwi':
            with open(os.path.join(temp_path, 'template/BIWI_template.obj'), mode='r') as f:
                append_f = f.read(None)
        else:
            assert(False)
        f_vert = []
        for idx in range(vertex.shape[0]):
            vstr = 'v ' + str(vertex[idx][0]) + ' ' + str(vertex[idx][1]) + ' ' + str(vertex[idx][2]) + '\n'
            f_vert.append(vstr)

        with open(out_path, mode='w') as f:
            f.writelines(f_vert)
            f.write(append_f)
        print('Write obj:', out_path, ' template=', template)
        
    def create(obj_str, template):
        return Mesh(Mesh.read_obj(obj_str), template_str=template)


def get_landmark_idx(template:str):
    if template.lower() == 'biwi':
        landmark_idx = [9620, 12613, 16045, 12592, 2447, 19062, 6116, 17243, 19698, 2012, 4592, 10283, 12490, 18031, 11658, 11320, 17394, 4133, 91, 11997, 12054, 9186, 10531, 7867, 22851, 19206, 17433, 3476, 3349, 16056, 15976, 426, 377, 463, 13622, 847, 1028, 13825, 21167, 9568, 9516, 21487, 20999, 6142, 12794, 12651, 10205, 13820, 15529, 10143, 304] # 16808 ,8016
    elif template.lower() == 'flame':
        landmark_idx = [3763, 3157, 335, 3153, 3712, 3868, 2134, 16, 2138, 3892, 3553, 3561, 3508, 3668, 2788, 2790, 3507, 1674, 1672, 2428, 2381, 3690, 2487, 2293, 2332, 3827, 1343, 1294, 1146, 955, 827, 3797, 2814, 2803, 3537, 1687, 1691, 1702, 1796, 1866, 3511, 2949, 2878, 2830, 2865, 3546, 1750, 1713, 1850, 3506, 2939] # 2884, 367
    return landmark_idx

def get_mouth_landmark(type_str:str):
    '''
    left-right is opposite to face itself
    left corner: 0
    right corner: 12
    upper mouth, from left to right: 1-11
    lower mouth, from right to left: 13-23
    total 24 points
    '''
    if type_str.lower() == 'biwi':
        return [401, 21621, 405, 370, 12776, 21564, 21688, 10096, 21323, 21215, 10119, 13813, 13825, 13570, 14029, 21156, 21149, 9956, 13212, 10684, 10537, 21519, 13149, 505]
    elif type_str.lower() == 'flame':
        return [2827, 2833, 2850, 2813, 2811, 2774, 3543, 1657, 1694, 1696, 1735, 1716, 1710, 1773, 1774, 1795, 1802, 1865, 3503, 2948, 2905, 2898, 2881, 2880]
    else:
        assert(False)

def get_landmark(mesh:Mesh, mouth=False):
    '''
    input: Mesh object, template: flame or biwi
    output: landmark: ndarray, [[x1,y1,z1], [x2,y2,z2],...] in specific order
    '''
    template = mesh.template
    landmark_idx = get_landmark_idx(template) if not mouth else get_mouth_landmark(template)
    landmark = [[0,0,0] for _ in range(len(landmark_idx))]
    for k in range(len(landmark_idx)):
        landmark[k] = mesh.v[landmark_idx[k]]

    return np.asarray(landmark)

def get_front_face_idx():
    with open('quick_fit/front-face-idx.json', 'r') as f:
        idx_list = json.load(f)['front_face']
    return idx_list


def np_norm(x:np.ndarray):
    return x / np.linalg.norm(x)

def get_rotation_matrix(a:np.ndarray, b:np.ndarray):
    '''
    return a rotation matrix R, R*a=b
    '''
    # reference: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
    #if np.sum(a) < 1e-5 or np.sum(b) < 1e-5:
    #    return np.identity(3)
    a = np_norm(a)
    b = np_norm(b)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    c = np.sum(a * b)
    if np.abs(c + 1) < 1e-6:
        print('rotation ill pose')
        return np.asarray([[-1,0,0],[0,-1,0],[0,0,-1]], dtype=np.float32)
    coeff = 1 / (1+c)
    vx = np.asarray([[0,-v[2],v[1]],
                    [v[2],0,-v[0]],
                    [-v[1],v[0],0]], dtype=np.float32)
    v_2 = coeff * np.dot(vx,vx)
    R = np.identity(3) + vx + v_2
    return R

def set_point_as_origin(vertex, lmk, index):
    #print(vertex.shape, ' ', lmk.shape)
    v = vertex - lmk[index]
    lmk = lmk - lmk[index]
    return v, lmk

def approx_transform_mouth(source:Mesh, target:Mesh):
    # step 0: extract landmarks
    lmk_s = get_landmark(source, mouth=True)
    lmk_t = get_landmark(target, mouth=True)
    lmk_hs = get_landmark(source, mouth=False)
    lmk_ht = get_landmark(target, mouth=False)
    lmk_s = np.concatenate((lmk_s, lmk_hs), axis=0) # 24+0
    lmk_t = np.concatenate((lmk_t, lmk_ht), axis=0)
    # step 1: estimate scale
    scale = np.linalg.norm(lmk_hs[9] - lmk_hs[0])
    ori_scale = scale
    tg_scale = np.linalg.norm(lmk_ht[9] - lmk_ht[0])
    source.v *= (tg_scale / ori_scale)
    lmk_s *= (tg_scale / ori_scale)
    # print('estimated scale coeff=', scale / ori_scale)
    for _ in range(5): # avoid percision problem when angle is near 180
        # step 2: rotate to zero position
        source.v, lmk_s = set_point_as_origin(source.v, lmk_s, 6)
        R1 = get_rotation_matrix(lmk_s[12] - lmk_s[0], lmk_t[12] - lmk_t[0]) # .obj file format: x,z,y
        lmk_s = np.matmul(lmk_s, R1.T) # Ra^T=b^T->(aR^T)=b
        source.v = np.matmul(source.v, R1.T)
        source.v, lmk_s = set_point_as_origin(source.v, lmk_s, 24+16)
        vec_s = np.cross(lmk_s[24+4]-lmk_s[24+16], lmk_s[24+5]-lmk_s[24+16])
        vec_t = np.cross(lmk_t[24+4]-lmk_t[24+16], lmk_t[24+5]-lmk_t[24+16])
        R2 = get_rotation_matrix(vec_s, vec_t)
        lmk_s = np.matmul(lmk_s, R2.T)
        source.v = np.matmul(source.v, R2.T)
    source.v += ((lmk_t[0] + lmk_t[12]) - (lmk_s[0]+lmk_s[12])) * 0.5
    lmk_s += ((lmk_t[0] + lmk_t[12]) - (lmk_s[0]+lmk_s[12])) * 0.5
    # step 3: fix average lmk error
    # avg_vec = np.average(lmk_t - lmk_s, axis=0)
    # source.v, lmk_s = source.v + avg_vec, lmk_s + avg_vec
    return source
            

def cal_point_loss(conf, tracker, face, face_np, vertex_list, min_list, max_list, n_grid, cri, queue):
    loss = 0
    out_of_bound = 0
    n_active = 0
    for n in vertex_list:
        # find nearest point
        code = get_code(face_np[n,:], min_list, max_list, n_grid)
        code_err = sum([abs(min(max(v,0), n_grid) - v) for v in code])
        if code_err > conf['max_depth']: # too far away, skip
            out_of_bound += 1
            continue
        else:
            point = tracker.find_nearest(face_np[n,:], code)
            if point is not None:
                p_tensor = torch.Tensor(data=point).to(face.device)
                loss += 1e6 * cri(face[n,:], p_tensor)
                n_active += 1
    return {'loss': loss, 'out_of_bound':out_of_bound, 'n_active':n_active}

    
    
    
    
    
