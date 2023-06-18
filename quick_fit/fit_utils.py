import json
import os
import struct
from math import sqrt
from queue import Queue, SimpleQueue
from threading import Thread
from plyfile import PlyData, PlyElement
import numpy as np
import torch


def get_code(point, min_list, max_list, n_grid):

    code = []
    for k in [0,1,2]:
        code.append(
            round((point[k] - min_list[k]) * n_grid / (max_list[k] - min_list[k]))
        )
    return code

def get_hash(code:list):
    return (code[0] + 1000*code[1] + 1000000*code[2])

def dist(a,b):
    return np.linalg.norm(a-b)


# tested
def read_vl(vl_str):
    vertex = []
    with open(vl_str, 'rb') as f:
        length = struct.unpack('I', f.read1(4))[0]
        #print('length=', length)
        for k in range(length):
            vertex.append(struct.unpack('fff', f.read1(12)))
            #print(vertex[k])
    return np.asarray(vertex) # N*3

def read_ply(ply_str):
    plydata = PlyData.read(ply_str)
    return np.asarray([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T # N, 3

def get_used_vertex(json_path, model_type):
    print('load no use json from:', json_path)
    with open(json_path, 'r') as f:
        no_use_set = set(json.load(f)['no_use'])
    used_list = []
    n_vert = 5023 if model_type == 'flame' else 23370
    for n in range(n_vert):
        if n not in no_use_set:
            used_list.append(n)
    print(f'{model_type} use ', len(used_list), ' vertex')
    return used_list


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
        # print('Write obj:', out_path, ' template=', template)
        
    def create(obj_str, template):
        return Mesh(Mesh.read_obj(obj_str), template_str=template)

def get_landmark_idx(template:str):
    if template.lower() == 'biwi':
        landmark_idx = [9620, 12613, 16045, 12592, 2447, 19062, 6116, 17243, 19698, 2012, 4592, 10283, 12490, 18031, 11658, 11320, 17394, 4133, 91, 11997, 
    12054, 9186, 10531, 7867, 22851, 19206, 17433, 3476, 3349, 16056, 15976, 426, 377, 463, 13622, 847, 1028, 13825, 21167, 9568, 9516, 21487, 20999, 
    6142, 12794, 12651, 10205, 13820, 15529, 10143, 304] # 16808 ,8016
    elif template.lower() == 'flame':
        landmark_idx = [3763, 3157, 335, 3153, 3712, 3868, 2134, 16, 2138, 3892, 3553, 3561, 3508, 3668, 2788, 2790, 3507, 1674, 1672, 2428, 2381, 3690, 2487, 
    2293, 2332, 3827, 1343, 1294, 1146, 955, 827, 3797, 2814, 2803, 3537, 1687, 1691, 1702, 1796, 1866, 3511, 2949, 2878, 2830, 2865, 3546, 1750, 1713, 1850, 
3506, 2939] # 2884, 367
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
        return [401, 21621, 405, 370, 12776, 21564, 21688, 10096, 21323, 21215, 10119, 13813, 13825, 13570, 14029, 21156, 21149, 9956, 13212, 10684, 10537, 
21519, 13149, 505]
    elif type_str.lower() == 'flame':
        return [2827, 2833, 2850, 2813, 2811, 2774, 3543, 1657, 1694, 1696, 1735, 1716, 1710, 1773, 1774, 1795, 1802, 1865, 3503, 2948, 2905, 2898, 2881, 2880]
    else:
        assert(False)


'''
    input: Mesh object, template: flame or biwi
    output: landmark: ndarray, [[x1,y1,z1], [x2,y2,z2],...] in specific order
'''
def get_landmark(mesh:Mesh, mouth=False):
    template = mesh.template
    landmark_idx = get_landmark_idx(template) if not mouth else get_mouth_landmark(template)
    landmark = [[0,0,0] for _ in range(len(landmark_idx))]
    for k in range(len(landmark_idx)):
        landmark[k] = mesh.v[landmark_idx[k]]
    
    return np.asarray(landmark)

def np_norm(x:np.ndarray):
    return x / np.linalg.norm(x)

'''
return a rotation matrix R, R*a=b
'''
def get_rotation_matrix(a:np.ndarray, b:np.ndarray):
    # reference: https://math.stackexchange.com/questions/180418/calculate-rotation-matrix-to-align-vector-a-to-vector-b-in-3d
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
