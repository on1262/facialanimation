import os
import struct
from plyfile import PlyData
import numpy as np


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
