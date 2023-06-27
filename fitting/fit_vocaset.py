import subprocess
import numpy as np
from utils.interface import FLAMEModel
import torch
import os
from torch import multiprocessing as mp
from fitting.fit_utils import *
from fitting.fit import get_landmark
import json
import time
from utils.config_loader import GBL_CONF, PATH

def approx_transform(source:Mesh, target:Mesh, frac_scale=False, return_lmk=True):
    '''
    estimate and apply an approx transform: nose is at origin, Y+ is up, Z+ is front
    input: mesh, source=other mesh, target=standard mesh
    output: source mesh in zero pose
    '''
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
    if return_lmk:
        return source, lmk_s
    else:
        return source

def fitting(flame:FLAMEModel, targets:list, init_param, return_v=False):
    # init
    seqs_len = len(targets)
    target_tensor = torch.as_tensor(np.asarray([target.v for target in targets]), dtype=torch.float32).to(flame.device)
    zero_tensor = torch.zeros((seqs_len, 156)).to(flame.device)
    o_mask = torch.ones((seqs_len, 156), requires_grad=False).to(flame.device)
    o_mask[:, 0:100] = 0.01
    o_mask[:, 100:103] = 0.6
    o_mask[:, 103:115] = 0.8
    o_mask[:, 150:] = 0
    fitted_param = torch.zeros((seqs_len, 156)).to(flame.device)
    fitted_param.requires_grad = True
    
    # configs
    phases = ['shape_quick','detail_quick'] if init_param is not None else ['shape', 'shape_2', 'detail']
    opt = torch.optim.Adam([fitted_param], lr=0, weight_decay=0)
    for phase in phases:
        print('stage: ', phase)
        conf = GBL_CONF['dataset']['vocaset']['fitting'][phase]
        for g in opt.param_groups:
            g['lr'] = conf['lr']
        for iter in range(conf['max_iter']):
            out = flame.forward({'shapecode':fitted_param[:,0:100], 'expcode':fitted_param[:,100:150], 'posecode':fitted_param[:,150:]})
            # print('target:', target_tensor.size(), 'out size:', out.size())
            assert(out.size() == target_tensor.size())
            loss_param = torch.nn.functional.l1_loss(out, target_tensor, reduction='mean') * 1000
            loss_reg = torch.nn.functional.l1_loss(fitted_param * o_mask, zero_tensor, reduction='mean') * 0.5
            loss = loss_reg + loss_param
            if iter % 50 == 0:
                print('iter:', iter, ' loss(mm)=', loss_param.item(), ' L1 loss=', 
                    loss_reg.item(), ' param L1=', torch.mean(torch.abs(fitted_param.detach()[:,100:])).item())
            opt.zero_grad()
            loss.backward()
            opt.step()
    
    print('fit Done')
    
    if return_v:
        face = flame.forward({'shapecode':fitted_param[:,0:100], 'expcode':fitted_param[:,100:150], 'posecode':fitted_param[:,150:]})
        if seqs_len == 1:
            return face.squeeze(0).detach().cpu().numpy(), fitted_param.detach().cpu()
        else:
            return face.detach().cpu().numpy(), fitted_param.detach().cpu()
    else:
        return fitted_param.detach().cpu()

def fit_mesh(init_param, flame, f_std:Mesh, target, output_obj_path:str):
    if isinstance(target, Mesh):
        target = [target]
    
    v_fit, params = fitting(flame, target, init_param, return_v=True)
    if len(target) == 1:
        Mesh.write_obj('flame', v_fit, output_obj_path)
    else:
        for idx in range(len(target)):
            Mesh.write_obj('flame', v_fit[idx], output_obj_path[idx])
    return params
    

def fit_sequence(flame, f_std, source_path_list, output_dir):
    if not isinstance(source_path_list[0], str):
        for idx, p_list in enumerate(source_path_list):
            fit_sequence(flame, f_std, p_list, output_dir[idx])
        return
    # load files
    mesh_list = []
    for p in source_path_list: # .ply
        vertex = None
        if p.endswith('.ply'):
            vertex = read_ply(p)
            mesh_list.append(Mesh(vertex, 'flame'))
    print('load', len(mesh_list), 'frames')
    if len(mesh_list) == 0:
        return
    os.makedirs(os.path.join(output_dir, 'ori'), exist_ok=True)
    for idx in range(len(mesh_list)):
        mesh_list[idx] = approx_transform(mesh_list[idx], f_std, return_lmk=False)
        Mesh.write_obj('flame', mesh_list[idx].v, os.path.join(output_dir, 'ori', f'{idx}.obj'))
    # fit sequence
    params_1 = fit_mesh(None, flame, f_std, mesh_list, [os.path.join(output_dir, f'{idx}.obj') for idx in range(0, len(mesh_list),1)])
    # write params file
    with open(os.path.join(output_dir, 'params.json'), 'w') as fp:
        json.dump(params_1.tolist(), fp)
    print('finished', output_dir)

def fit_sequence_mp(task_args):
    flame = FLAMEModel(device='cuda:' + str(task_args['device']))
    flame_std_path = './template/flame_std.obj'
    f_std = Mesh.create(flame_std_path,'flame')
    fit_sequence(flame, f_std, task_args['source_path_list'], task_args['output_dir'])

def fit_multi_sequences(source_dir_list, output_list, enable_mp=False):
    mp.set_start_method('spawn')
    arg_list = []
    print('fit multi seqs')
    device_list = [6,7]
    n_device = len(device_list)
    k_process = 4
    dir_packages = [source_dir_list[idx:min(idx+20, len(source_dir_list))] for idx in range(0, len(source_dir_list), 20)]
    out_packages = [output_list[idx:min(idx+20, len(output_list))] for idx in range(0, len(output_list), 20)]
    for idx, package in enumerate(dir_packages):
        p_list = [sorted([os.path.join(p_dir, pl) for pl in os.listdir(p_dir)]) for p_dir in package] # sentence01/sentence01.000001.ply
        arg_list.append({
            'device' : device_list[idx % n_device],
            'source_path_list' : p_list,
            'output_dir' : out_packages[idx]
        })
    start_time = time.time()
    if enable_mp:
        with mp.Pool(n_device*k_process, maxtasksperchild=n_device*k_process) as p:
            p.map(fit_sequence_mp, arg_list)
    else:
        for task_args in arg_list:
            flame = FLAMEModel(device='cuda:' + str(task_args['device']))
            flame_std_path = './template/flame_std.obj'
            f_std = Mesh.create(flame_std_path,'flame')
            fit_sequence(flame, f_std, task_args['source_path_list'], task_args['output_dir'])
    print('multi processes ended in ', time.time() - start_time)


if __name__ == '__main__':
    # init
    data_dir= PATH['dataset']['vocaset']
    output_dir = os.path.join(PATH['dataset']['cache'], 'vocaset', 'fit_output')
    subprocess.run(['rm','-rf', output_dir])
    os.makedirs(output_dir, exist_ok=True)
    input_list = []
    output_list = []
    for person in os.listdir(data_dir):
        if person.startswith('FaceTalk'):
            sentences = sorted(os.listdir(os.path.join(data_dir, person))) # sentenceXX
            for sentence in sentences: # sentence01
                sentence_dir = os.path.join(data_dir, person, sentence)
                if os.path.isdir(sentence_dir):
                    input_list.append(sentence_dir)
                    os.makedirs(os.path.join(output_dir, person, sentence), exist_ok=True)
                    output_list.append(os.path.join(output_dir, person, sentence))
    print('start fitting')
    fit_multi_sequences(input_list, output_list, enable_mp=True)
    
    
    
    
    
