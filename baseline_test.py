import sys
import os
import torch
from torch.utils.data import Dataset
from utils.interface import LSTMEMO, FaceFormerModel, VOCAModel, BaselineConverter
from dataset import BaselineBIWIDataset, BaselineVOCADataset
from utils.fitting.fit import Mesh, approx_transform, approx_transform_mouth, get_mouth_landmark
import torch.nn.functional as F
import subprocess
import argparse
import numpy as np
from utils.detail_fixer import DetailFixer

def baseline_test(model_name, model, model_output_type, dataset: Dataset, gt_output_type, device,save_obj=True):
    
    max_loss = 0
    avg_loss = 0
    lmk_idx_out = get_mouth_landmark(model_output_type)
    lmk_idx_gt = get_mouth_landmark(gt_output_type)
    print('load', len(dataset),'in test dataset')
    output_path = '/home/chenyutong/facialanimation/quick_fit/output'
    if save_obj:
        subprocess.run(['rm','-rf', output_path, 'baseline_eval'])
        subprocess.run(['rm','-rf', output_path, 'baseline_gt'])
    with torch.no_grad():
        for idx,data in enumerate(dataset):
            d = {'wav':data['wav'].to(device), 'code_dict':None , 'name':data['name'], 'seqs_len':data['seqs_len'],'verts':data['verts'].to(device),
                'flame_template':data['flame_template'], 'shapecode':data['shapecode'].to(device)}
            d['emo_tensor_conf'] = ['no_use']
            gt = d['verts']
            output =  model.forward(d) # 1, vertexm 3
            if model_name == 'lstm_emo': # seqs_len, 5023,3
                output = F.interpolate(output.unsqueeze(0).permute(0,2,1,3), size=(gt.size(0),3)).permute(0,2,1,3).squeeze(0)
            try:
                assert(output.size(0) == gt.size(0))
            except Exception as e:
                print('name=', data['name'], 'output size=', output.size(), ' gt size=', gt.size())
                min_len = min(output.size(0), gt.size(0))
                gt = gt[:min_len,:,:]
                output = output[:min_len,:,:]
            seq_len = gt.size(0)
            gt, output = gt.detach().cpu().numpy(), output.detach().cpu().numpy()
            seq_max_loss = 0
            seq_avg_loss = 0
            if save_obj:
                bt_out_p = os.path.join(output_path, 'baseline_eval', d['name'])
                bt_gt_p = os.path.join(output_path, 'baseline_gt', d['name'])
                os.makedirs(bt_out_p, exist_ok=True)
                os.makedirs(bt_gt_p, exist_ok=True)
                        
            if 'emo' in model_name:
                output = np.asarray(output)
                fixer = DetailFixer(d['flame_template']['ply'], target_area='mouth',fix_mesh=None)
                output = output + (fixer.template_mesh.v - output[0,:,:])
            
            
            for idx2 in range(seq_len):
                m_out = Mesh(output[idx2,:,:], model_output_type)
                m_gt = Mesh(gt[idx2,:,:], gt_output_type)
                #m_out,_ = approx_transform(m_out, m_gt, frac_scale=True)
                m_out = approx_transform_mouth(m_out, m_gt)
                
                # the scale of output is not corresponds to real scale
                #m_out = mesh_seq[idx2]
                delta = m_gt.v[lmk_idx_gt,:]-m_out.v[lmk_idx_out,:]
                delta = np.sqrt(np.power(delta[:,0],2) + np.power(delta[:,1],2) + np.power(delta[:,2],2))
                seq_avg_loss += np.mean(delta)
                seq_max_loss += np.max(delta)
                if save_obj:
                    Mesh.write_obj(m_out.template, m_out.v, os.path.join(bt_out_p, str(idx2) + '.obj'))
                    Mesh.write_obj(m_gt.template, m_gt.v, os.path.join(bt_gt_p, str(idx2) + '.obj'))
            max_loss += (seq_max_loss / seq_len)
            avg_loss += (seq_avg_loss / seq_len)
            if idx % 10 == 0:
                print('idx=',idx, 'mean loss=', avg_loss/(idx+1), 'max loss=', max_loss/(idx+1), 'name=', data['name'])

    print('='*10, 'test result', '='*10)
    print('model type:', type(model))
    print('average max vertex loss:', max_loss/len(dataset))
    print('average avg vertex loss:', avg_loss/len(dataset))
    print('Done')
    return max_loss / len(dataset)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='lstm_emo',)
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--save_obj', type=bool, default=False)
    args = parser.parse_args()
    
    # load biwi test dataset
    device = torch.device('cuda:' +str(args.device))
    test_voca_path = r'/home/chenyutong/facialanimation/dataset_cache/VOCASET'
    label_dict = None
    test_dataset = BaselineVOCADataset(test_voca_path, device=device)
    # load model
    if 'emo' in args.model:
        model = LSTMEMO(device, model_name=args.model)
        model_output_type = 'flame'
    elif args.model == 'convert':
        model = BaselineConverter(device)
        model_output_type = 'flame'
    elif args.model == 'faceformer_flame':
        model = FaceFormerModel(device)
        model_output_type = 'flame'
    elif args.model == 'voca':
        model = VOCAModel(device)
        model_output_type = 'flame'
    baseline_test(args.model, model, model_output_type, dataset=test_dataset, gt_output_type='flame', device=device, save_obj=args.save_obj)