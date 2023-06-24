import sys
sys.path.append('/home/chenyutong/facialanimation')
import numpy as np
from utils.interface import FLAMEModel
import torch
import os
from queue import SimpleQueue, Queue
from threading import Thread
from math import sqrt
from torch import multiprocessing as mp
import random
import json
import time
import struct

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
        print('Write obj:', out_path, ' template=', template)
        
    def create(obj_str, template):
        return Mesh(Mesh.read_obj(obj_str), template_str=template)

def get_landmark_idx(template:str):
    if template.lower() == 'biwi':
        landmark_idx = [9620, 12613, 16045, 12592, 2447, 19062, 6116, 17243, 19698, 2012, 4592, 10283, 12490, 18031, 11658, 11320, 17394, 4133, 91, 11997, 
    12054, 9186, 10531, 7867, 22851, 19206, 17433, 3476, 3349, 16056, 15976, 426, 377, 463, 13622, 847, 1028, 13825, 21167, 9568, 9516, 21487, 20999, 
    6142, 12794, 12651, 10205, 13820, 15529, 10143, 304] # 16808 ,8016
    elif template.lower() == 'flame':
        landmark_idx = [3763, 3157, 335, 3153, 3712, 3868, 2134, 16, 2138, 3892, 3553, 3561, 3508, 3668, 2788, 2790, 3507, 1674, 1672, 2428, 2381, 3690, 2487, 
    2293, 2332, 3827, 1343, 1294, 1146, 955, 827, 3797, 2814, 2803, 3537, 1687, 1691, 1702, 1796, 1866, 3511, 2949, 2878, 2830, 2865, 3546, 1750, 1713, 1850, 3506, 2939] # 2884, 367
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

def get_front_face_idx():
    with open('quick_fit/front-face-idx.json', 'r') as f:
        idx_list = json.load(f)['front_face']
    return idx_list

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

'''
estimate and apply an approx transform: nose is at origin, Y+ is up, Z+ is front
input: mesh, source=other mesh, target=standard mesh
output: source mesh in zero pose
'''
def approx_transform(source:Mesh, target:Mesh, frac_scale=False):
    # step 0: extract landmarks
    lmk_s = get_landmark(source, mouth=False)
    lmk_t = get_landmark(target, mouth=False)
    # step 1: estimate scale
    scale = np.linalg.norm(lmk_s[9] - lmk_s[0])
    ori_scale = scale
    if not frac_scale:
        while scale > 0.15 or scale < 0.05:
            if scale > 0.15:
                scale = scale / 10
                source.v /= 10
                lmk_s /= 10
            elif scale < 0.05:
                scale = scale * 10
                source.v *= 10
                lmk_s *= 10
    else:
        tg_scale = np.linalg.norm(lmk_t[9] - lmk_t[0])
        source.v *= (tg_scale / ori_scale)
        lmk_s *= (tg_scale / ori_scale)
    #print('estimated scale coeff=', scale / ori_scale)
    for _ in range(5): # avoid percision problem when angle is near 180
        # step 2: rotate to zero position
        source.v, lmk_s = set_point_as_origin(source.v, lmk_s, 0)
        R1 = get_rotation_matrix(lmk_s[9] - lmk_s[0], lmk_t[9] - lmk_t[0]) # .obj file format: x,z,y
        lmk_s = np.matmul(lmk_s, R1.T) # Ra^T=b^T->(aR^T)=b
        source.v = np.matmul(source.v, R1.T)
        source.v, lmk_s = set_point_as_origin(source.v, lmk_s, 16)
        vec_s = np.cross(lmk_s[4]-lmk_s[16], lmk_s[5]-lmk_s[16])
        vec_t = np.cross(lmk_t[4]-lmk_t[16], lmk_t[5]-lmk_t[16])
        R2 = get_rotation_matrix(vec_s, vec_t)
        lmk_s = np.matmul(lmk_s, R2.T) 
        source.v = np.matmul(source.v, R2.T)
        source.v += (lmk_t[35] - lmk_s[35])
        lmk_s += (lmk_t[35] - lmk_s[35])
    # step 3: fix average lmk error
    avg_vec = np.average(lmk_t - lmk_s, axis=0)
    source.v, lmk_s = source.v + avg_vec, lmk_s + avg_vec
    return source, lmk_s

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

class Tracker:
    def __init__(self, map, n_grid, min_depth, max_depth):
        self.point = None
        self.map = map
        self.n_grid = n_grid
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.queue = SimpleQueue() # bfs
        self.found_set = set()
    
    def find_nearest(self, point, code):
        # init
        while not self.queue.empty():
            self.queue.get()
        self.found_set.clear()
        self.point = point
        # put first package
        code = [min(max(v,0), self.n_grid) for v in code] # clip
        self.queue.put((code, 0))
        result = None
        while not self.queue.empty():
            code, depth = self.queue.get()
            if depth > self.max_depth:
                continue

            if self.map.get(get_hash(code)) is None:
                x,y,z = code[0], code[1], code[2]
                next_code = [[x-1,y,z],[x+1,y,z],[x,y-1,z],[x,y+1,z],[x,y,z-1],[x,y,z+1]]
                random.shuffle(next_code)
                for new_code in next_code:
                    if [min(max(0,c),self.n_grid) for c in new_code] != new_code: # out of bound
                        continue
                    if get_hash(new_code) in self.found_set:
                        continue
                    self.queue.put((new_code, depth+1))
                    self.found_set.add(get_hash(new_code))
            else:
                if depth < self.min_depth:
                    result = None
                    break
                else:
                    result = self.map[get_hash(code)][0]
                    for v in self.map[get_hash(code)]:
                        result = result if (dist(result, self.point) < dist(v, self.point)) else v
                    break # stop when nearest point is found

        return result
            
def get_fit_config(phase:str):
    conf = {}
    if phase == 'lmk':
        conf['max_iter'] = 400
        conf['lr'] = 0.1
        conf['mask'] = list(range(0,125)) + [153]
        conf['decay'] = 1e-2
    elif phase == 'lmk_quick':
        conf['max_iter'] = 400
        conf['lr'] = 0.1
        conf['mask'] = list(range(0, 125)) + [153]
        conf['decay'] = 1e-2
    elif phase == 'shape':
        conf['max_iter'] = 15
        conf['lr'] = 0.05
        conf['stop_eps'] = 1e-3
        conf['mask'] = list(range(0,50)) + list(range(100, 125))
        conf['min_depth'] = 1
        conf['max_depth'] = 3
        conf['decay'] = 1e-2
    elif phase == 'shape_quick':
        conf['max_iter'] = 10
        conf['lr'] = 0.05
        conf['stop_eps'] = 1e-3
        conf['mask'] = list(range(0,50)) + list(range(100, 125)) # expression only
        conf['min_depth'] = 1
        conf['max_depth'] = 3
        conf['decay'] = 1e-2
    elif phase == 'detail':
        conf['max_iter'] = 5
        conf['lr'] = 0.05
        conf['stop_eps'] = 1e-3
        conf['mask'] = list(range(0, 125))
        conf['min_depth'] = -1
        conf['max_depth'] = 2
        conf['decay'] = 1e-3
    elif phase == 'detail_quick':
        conf['max_iter'] = 5
        conf['lr'] = 0.05
        conf['stop_eps'] = 1e-3
        conf['mask'] = list(range(0, 125))
        conf['min_depth'] = -1
        conf['max_depth'] = 2
        conf['decay'] = 1e-3
    elif phase == 'final':
        conf['max_iter'] = 5
        conf['lr'] = 0.005
        conf['stop_eps'] = 1e-3
        conf['mask'] = list(range(0, 125))
        conf['min_depth'] = -1
        conf['max_depth'] = 2
        conf['decay'] = 1e-3
    elif phase == 'final_quick':
        conf['max_iter'] = 5
        conf['lr'] = 0.005
        conf['stop_eps'] = 1e-3
        conf['mask'] = list(range(0, 125))
        conf['min_depth'] = -1
        conf['max_depth'] = 2
        conf['decay'] = 1e-3
    return conf

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

def fitting(flame:FLAMEModel, source:Mesh, init_param, flame_idx_list:list, biwi_idx_list:list):
    # init
    fitted_param = torch.zeros((1, 156)).to(flame.device)
    cri = torch.nn.MSELoss()
    if init_param is not None:
        fitted_param += init_param.to(flame.device)
    # first stage: fit landmark
    print('stage: fitting landmark')
    conf = get_fit_config('lmk' if init_param is None else 'lmk_quick')
    lmk_flame_idx = get_landmark_idx('flame')
    lmk_source = torch.from_numpy(get_landmark(source)).float().to(flame.device)
    params = torch.zeros((1, len(conf['mask'])), device=flame.device)
    params = fitted_param[:, conf['mask']]
    params.requires_grad=True
    opt = torch.optim.Adam([params], lr=conf['lr'], weight_decay=conf['decay'])
    for iter in range(conf['max_iter']):
        p_zero = torch.zeros((1, 156), device=flame.device)
        p_zero[:,conf['mask']] += params
        face = flame.forward({'shapecode':p_zero[:,0:100], 'expcode':p_zero[:,100:150], 'posecode':p_zero[:,150:156]}).squeeze(0)
        loss = 1e5 * cri(lmk_source, face[lmk_flame_idx,:])
        opt.zero_grad()
        loss.backward()
        opt.step()
        #if iter % 100 == 0:
        #    print('iter ', iter, 'loss=', loss.item(), ' params L1=', torch.mean(torch.abs(params)).item())
    fitted_param[:,conf['mask']] = params.detach().clone()
    # make grid
    min_list = [min(source.v[:,0]), min(source.v[:,1]), min(source.v[:,2])]
    max_list = [max(source.v[:,0]), max(source.v[:,1]), max(source.v[:,2])]
    #print('min xyz:', min_list)
    #print('max xyz=', max_list)
    n_grid = 50
    map = {} # hash(code) -> point
    for n in biwi_idx_list:
        code = get_code(source.v[n,:], min_list, max_list, n_grid)
        if get_hash(code) not in map.keys():
            map[get_hash(code)] = [source.v[n,:]]
        else:
            map[get_hash(code)].append(source.v[n,:])
    #print('average point per grid=', source.v.shape[0] / len(map))
    # configs
    phases = ['shape_quick', 'detail_quick', 'final_quick'] if init_param is not None else ['shape', 'detail', 'final']
    for phase in phases:
        print('stage: ', phase)
        conf = get_fit_config(phase)
        params = torch.zeros((1, len(conf['mask'])), device=flame.device)
        params += fitted_param[:,conf['mask']]
        params.requires_grad = True

        opt = torch.optim.Adam([params], lr=conf['lr'], weight_decay=conf['decay'])
        queue = Queue()
        for iter in range(conf['max_iter']):
            loss = 0
            n_active = 0
            n_out_of_bound = 0
            p_zero = fitted_param.detach().clone()
            p_zero[:, conf['mask']] = params
            face = flame.forward({'shapecode':p_zero[:,0:100], 'expcode':p_zero[:,100:150], 'posecode':p_zero[:,150:156]}).squeeze(0)
            face_np = face.detach().cpu().numpy()
            tracker = Tracker(map, n_grid, conf['min_depth'], conf['max_depth'])
            random.shuffle(flame_idx_list)
            data = cal_point_loss(conf, tracker, face, face_np, flame_idx_list, min_list, max_list, n_grid, cri, queue)
            loss += data['loss']
            n_out_of_bound += data['out_of_bound']
            n_active += data['n_active']
            loss /= n_active
            if loss.item() < conf['stop_eps']:
                print('stop at ', loss.item(), ' iter=', iter)
                break
            opt.zero_grad()
            loss.backward()
            opt.step()
            #print('iter:', iter, ' loss(mm)=', sqrt(loss.item()), ' active=', n_active, '/', face_np.shape[0], ' n_out=', n_out_of_bound, \
            #    ' params L1=', torch.mean(torch.abs(params)).item())
        fitted_param[:, conf['mask']] = params.detach().clone()
    
    print('fit Done')
    face = flame.forward({'shapecode':fitted_param[:,0:100], 'expcode':fitted_param[:,100:150], 'posecode':fitted_param[:,150:156]})
    return face.squeeze(0).detach().cpu().numpy(), fitted_param.detach().cpu()

def fit_one_mesh(init_param, flame_idx_list, biwi_idx_list, flame, f_std, source, output_obj_path):
    if flame_idx_list is None:
        flame_idx_list = get_used_vertex('./template/flame.json', 'flame')
    if biwi_idx_list is None:
        biwi_idx_list = get_used_vertex('./template/biwi.json', 'biwi')
    start = time.time()
    m, lmk_m = approx_transform(source, f_std) # fit flame
    #Mesh.write_obj('biwi', m.v, './output/scan_scaled.obj')
    v_fit, params = fitting(flame, m, init_param, flame_idx_list, biwi_idx_list)
    #print('fit completed in ', time.time() - start, ' sec.') # 24 sec
    if output_obj_path is not None:
        Mesh.write_obj('flame', v_fit, output_obj_path)
    return params
    

def fit_sequence(flame, f_std, source_path_list, output_dir):
    flame_idx_list = get_used_vertex('./template/flame.json', 'flame')
    biwi_idx_list = get_used_vertex('./template/biwi.json', 'biwi')
    # load files
    mesh_list = []
    for p in source_path_list:
        vertex = None
        if p.endswith('.vl'):
            vertex = read_vl(p)
            mesh_list.append(Mesh(vertex, 'biwi'))
    print('load', len(mesh_list), 'frames')
    if len(mesh_list) == 0:
        return
    param_list = []
    # fit idx 0
    param_list.append(fit_one_mesh(None, flame_idx_list, biwi_idx_list, flame, f_std, mesh_list[0], os.path.join(output_dir, '0.obj')))
    # fit sequence
    for idx in range(1, len(mesh_list)):
        param_list.append(fit_one_mesh(param_list[0], flame_idx_list, biwi_idx_list, flame, f_std, mesh_list[idx], os.path.join(output_dir, f'{idx}.obj')))
    # write params file
    with open(os.path.join(output_dir, 'params.json'), 'w') as fp:
        json.dump(torch.cat(param_list, dim=0).tolist(), fp)

def fit_sequence_mp(task_args):
    flame = FLAMEModel(device='cuda:' + str(task_args['device']))
    flame_std_path = './template/flame_std.obj'
    f_std = Mesh.create(flame_std_path,'flame')
    fit_sequence(flame, f_std, task_args['source_path_list'], task_args['output_dir'])

def fit_multi_sequences(source_dir_list, output_list):
    mp.set_start_method('spawn')
    arg_list = []
    print('fit multi seqs')
    n_device = 2
    k_process = 10
    for idx, p_dir in enumerate(source_dir_list):
        p_list = sorted(os.listdir(p_dir)) # XX/XX.vl
        p_list = [os.path.join(p_dir, pl) for pl in p_list]
        arg_list.append({
            'device' : (idx % n_device)+5, # 1-7
            'source_path_list' : p_list,
            'output_dir' : output_list[idx]
        })
    start_time = time.time()
    with mp.Pool(n_device*k_process, maxtasksperchild=n_device*k_process) as p:
        p.map(fit_sequence_mp, arg_list)
    print('multi processes ended in ', time.time() - start_time)


if __name__ == '__main__':
    # init
    #flame = FLAMEModel(device='cuda:0')
    #flame_std_path = './template/flame_std.obj'
    #if not os.path.exists(flame_std_path):
    #    print('creating flame standard obj')
    #    with torch.no_grad():
    #        temp = FLAMEModel.get_codedict()
    #        flame_std = flame.forward(temp).squeeze().cpu().numpy() # out: batch, 5023, 3
    #    Mesh.write_obj('flame', flame_std, './output/flame_std.obj')
    #f_std = Mesh.create(flame_std_path,'flame')
    #data_dir = './input/test_dataset'
    data_dir= r'/home/chenyutong/facialanimation/dataset_cache/BIWI/fit'
    output_dir = r'/home/chenyutong/facialanimation/dataset_cache/BIWI/fit_output'
    os.makedirs(output_dir, exist_ok=True)
    input_list = []
    output_list = []
    for person in os.listdir(data_dir):
        file_list = sorted(os.listdir(os.path.join(data_dir, person))) # fit/A1/01
        for file in file_list:
            if os.path.isdir(os.path.join(data_dir,person, file)):
                input_list.append(os.path.join(data_dir,person, file))
                os.makedirs(os.path.join(output_dir, person, file), exist_ok=True)
                output_list.append(os.path.join(output_dir, person, file))
    print('start fitting')
    fit_multi_sequences(input_list, output_list)
    
    #m = Mesh.create('./input/scan.obj', 'biwi')
    #fit_one_mesh(None, flame, f_std, m)
    
    
    
    
    
