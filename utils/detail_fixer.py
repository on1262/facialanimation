import torch
import numpy as np
import sys, os
from plyfile import PlyData, PlyElement
import numpy as np
sys.path.append('/home/chenyutong/facialanimation')
from fitting.fit_utils import Mesh, get_mouth_landmark, get_landmark_idx
from fitting.fit import Mesh, approx_transform, approx_transform_mouth
import torch.multiprocessing as mp

def normalize(vec):
    return vec / np.linalg.norm(vec)

def get_ax(source_mat):
    x_ax = source_mat[1,:]-source_mat[0,:]
    y_ax = source_mat[2,:]-source_mat[0,:]
    z_ax = np.cross(x_ax, y_ax)
    y_new_ax = np.cross(z_ax, x_ax)
    return np.asarray([normalize(x_ax), normalize(y_new_ax), normalize(z_ax)])


def vertices2nparray(vertex):
    return np.asarray([vertex['x'],vertex['y'],vertex['z']]).T


class DetailFixer():
    def __init__(self, template_path, target_area='mouth', fix_mesh=None):
        self.template = PlyData.read(template_path)
        if target_area == 'mouth':
            self.vert_idx = get_mouth_landmark('flame') + get_landmark_idx('flame')
        else:
            assert(0)
        self.verts_dict = {}
        vert_indices = self.template['face'].data['vertex_indices']
        if fix_mesh is not None:
            self.template_mesh = fix_mesh
            self.template['vertex']['x'] = fix_mesh.v[:,0]
            self.template['vertex']['y'] = fix_mesh.v[:,1]
            self.template['vertex']['z'] = fix_mesh.v[:,2]
        else:
            self.template_mesh = Mesh(vertices2nparray(self.template['vertex']),'flame')
        for face_idx in range(len(vert_indices)):
            for vert_idx in vert_indices[face_idx]:
                if vert_idx not in self.verts_dict.keys():
                    self.verts_dict[vert_idx] = [face_idx]
                elif face_idx not in self.verts_dict[vert_idx]:
                    self.verts_dict[vert_idx].append(face_idx)


    '''
    思路是获取三角面片, 然后按照原面片->点的相对位置关系来得到新的点. 如果点在多个面片中, 则取平均
    '''
    def fix_sequence(self, sequence:np.ndarray, cache_path=None, k_process=None):
        origin = sequence[0] # origin face
        print(f'Fixing length={sequence.shape[0]}', end='')
        # new_seq = np.zeros(sequence.shape)
        new_seq = sequence.copy()
        new_seq[:, self.vert_idx, :] = 0
        #for idx in range(sequence.shape[1]):
        for idx in self.vert_idx:
            # if idx % 200 == 0:
            #     print('.', end='')
            for face_idx in self.verts_dict[idx]:
                source_ori, arr = self.read_triangle(origin, face_idx, idx,arr=None)
                target_ori = np.asarray(list(self.template['vertex'][idx]))
                vec = target_ori - source_ori[0,:]
                old_ax = get_ax(source_ori)
                for s_idx in range(1, sequence.shape[0]):
                    mat, _ = self.read_triangle(sequence[s_idx], face_idx, idx, arr=arr)
                    new_seq[s_idx, idx, :] += self.estimate_point(
                        old_ax, vec, mat) / len(self.verts_dict[idx])
        new_seq[0] = vertices2nparray(self.template['vertex'])
        print('sequence Done')
        return new_seq

    def read_triangle(self, verts, face_idx, vert_idx, arr=None):
        if arr is not None:
            new_arr = arr
        else:
            arr_idx = self.template['face'].data['vertex_indices'][face_idx]
            new_arr = []
            for idx in arr_idx:
                if idx == vert_idx:
                    new_arr.insert(0, idx)
                else:
                    new_arr.append(idx)
        mat = np.asarray([verts[vid,:] for vid in new_arr])
        return mat, arr

    # target point should be correspond with source[0]
    # source: (3,3), source[0,:] => point 0
    def estimate_point(self, old_ax, vec, source_new):
        new_ax = get_ax(source_new)
        convert_mat = np.matmul(new_ax, old_ax.T)
        new_point = np.matmul(convert_mat ,vec) + source_new[0,:]

        return new_point

if __name__ == '__main__':
    test_path = r'/home/chenyutong/facialanimation/utils/detail_fixer_test'
    template_path = os.path.join(test_path, 'template.ply')
    sequence_list = ['0.obj', '1.obj', '2.obj','3.obj', '4.obj']
    sequence_list = [os.path.join(test_path, s) for s in sequence_list]
    sequence = np.asarray([Mesh.read_obj(s) for s in sequence_list])
    fixer = DetailFixer(template_path)
    sequence = fixer.fix_sequence(sequence)
    for idx in range(len(sequence_list)):
        Mesh.write_obj('flame', sequence[idx, ...], os.path.join(test_path, 'out', f'{idx}.obj'))







