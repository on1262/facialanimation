import sys
sys.path.append('/home/chenyutong/facialanimation')
import os
from DAN.networks.dan import DAN
from EMOCABasic import DecaModule, save_images, FLAME
from utils.converter import save_img
from fitting.fit_utils import Mesh
from torchvision import transforms
from PIL import Image
from threading import Thread
import numpy as np
import glob
import torch
import json


class FLAMEModel():
    def __init__(self, device):
        print('loading FLAME model')
        self.device = device
        self.flame = FLAME().to(device)
        for p in self.flame.parameters():
            p.requires_grad = False
    
    def forward(self, codedict):
        codedict = {key:val.to(self.device) for key,val in codedict.items()}
        verts,_,_ = self.flame(shape_params=codedict['shapecode'], \
            expression_params=codedict['expcode'], \
            pose_params=codedict['posecode'], eye_pose_params=None)
        return verts
    
    @classmethod
    def get_codedict(cls):
        return {'shapecode': torch.zeros(1,100), 'expcode': torch.zeros(1,50), 'posecode': torch.zeros(1,6)}


class EMOCAModel():
    def __init__(self, device, decoder_only=False):
        self.device = device
        self.emoca = DecaModule(decoder_only)
        self.emoca.load_state_dict(torch.load('/home/chenyutong/facialanimation/EMOCABasic/data/emoca_basic.ckpt'), strict=(not decoder_only))
        for p in self.emoca.parameters():
            p.requires_grad = False
        self.emoca.to(device=device)
        self.emoca.eval()
        self.max_batch_size=40 # 40

    """
    input: imgs(tensor), batch,3,224,224. format = convert_img( o_code='emoca')
    output: dict{'shapecode(100)', 'texcode','expcode(50)','posecode(6)','cam(9,3)','lightcode', 'images(optional)'}
    """
    def encode(self, imgs, return_img):
        if isinstance(imgs, torch.Tensor):
            return self.emoca.encode({'image':imgs.float().to(self.device)}, return_img) # emoca use 3hwRGB01 format
        elif isinstance(imgs, dict):
            assert('image' in imgs.keys())
            imgs['image'] = imgs['image'].float().to(self.device)
            return self.emoca.encode(imgs, return_img)

    def move_result(self, outs, result, device, start, end):
        for key in outs.keys():
            result[key][start:end,...] += outs[key].to(device)
        outs.clear()


    '''
    inputs:
        list(dict) or dict
        return_set:
            {'verts', 'geo', 'coarse', 'faces'}
    outputs: dict
        output_images_coarse: batch*3*224*224 image format=emoca
        verts: batch*5663
        geometry_coarse: grey image
    '''
    def decode(self, codedict_list, return_set:set, target_device=None):
        if target_device is None:
            target_device = self.device
        if 'geo' in return_set:
            return_set.add('verts')
        if isinstance(codedict_list, dict):
            codedict_list = [codedict_list]
        for codedict in codedict_list:
            try:
                for key in codedict.keys(): # assert seq length are the same
                    if codedict[key] is not None and key != 'decode_verts':
                        assert(codedict[key].size(0) == codedict['shapecode'].size(0))
                assert(codedict['shapecode'].size(1) == 100)
                #assert(codedict['lightcode'].size(-2) == 9 and codedict['lightcode'].size(-1) == 3)
                assert(codedict['expcode'].size(1) == 50)
                assert(codedict['posecode'].size(1) == 6)
                if 'coarse' in return_set:
                    assert(codedict['cam'].size(1) == 3)
            except AssertionError as e:
                print('EMOCA decode error, check tensor size in codedict')
                for key in codedict.keys(): # assert seq length are the same
                    if codedict[key] is not None:
                        print('key:', str(key), ' size:', codedict[key].size())

        # concat codedict
        len_list = [cd['shapecode'].size(0) for cd in codedict_list]
        for idx in range(len(len_list)):
            len_list[idx] += (len_list[idx-1] if idx > 0 else 0)
        # codedict is about 2 MB per batch(70imgs), imgs is 40 MB per batch
        codedict_keys = {'shapecode', 'expcode', 'posecode'}
        if 'coarse' in return_set:
            codedict_keys.update({'texcode', 'cam'})
        if 'geo' in return_set:
            codedict_keys.update({'cam'})
        if 'decode_verts' in return_set:
            codedict_keys.update({'decode_verts'})
        codedict = {key:torch.cat([cd[key] for cd in codedict_list], dim=0).to(self.device) for key in codedict_keys}
        
        # avoid re-split
        result = {}
        if 'coarse' in return_set:
            result['output_images_coarse'] = torch.zeros((len_list[-1],3,224,224), dtype=torch.float32, device=target_device)
        if 'verts' in return_set:
            result['verts'] = torch.zeros((len_list[-1], 5023, 3), dtype=torch.float32, device=target_device)
        if 'geo' in return_set:
            result['geometry_coarse'] = torch.zeros((len_list[-1],3,224,224), dtype=torch.float32, device=target_device)

        # check_num = result[-1,-1,-1,-1]
        batch_size = codedict['expcode'].size(0)
        if batch_size > self.max_batch_size:
            t_new = None
            now_batch = self.max_batch_size
            while now_batch < batch_size + self.max_batch_size:
                in_dict = {key:val[now_batch - self.max_batch_size:min(batch_size, now_batch),...] for key,val in codedict.items()}
                if t_new is not None:
                    t_new.join()
                    del t_new
                #print('before decode: ', torch.cuda.max_memory_allocated(self.device))
                out_dict = self._decode(in_dict, return_set)
                t_new = Thread(target=self.move_result, args= \
                    (out_dict, result, target_device, now_batch - self.max_batch_size, min(batch_size, now_batch)))
                t_new.start()
                now_batch += self.max_batch_size
            t_new.join()
        else:
            result = self._decode(codedict, return_set)
            for key in result:
                result[key] = result[key].to(target_device)
        
        if 'faces' in return_set:
            result['faces'] = self.emoca.deca.render.faces[0].detach().to(target_device) # topography is constant
        return result

            
    # 48 Bytes for 1 pixel, image(3*224*224) = 6.8 MB, 1 batch=1 GB
    def _decode(self, codedict, return_set:set):
        codedict = {key:val.to(self.device) for key,val in codedict.items()}
        values = self.emoca.decode(codedict, return_set=return_set)
        outdict = {}
        if 'coarse' in return_set:
            outdict['output_images_coarse'] = values['predicted_images'] # background rendered
        if 'verts' in return_set:
            outdict['verts'] = values['verts']
        if 'geo' in return_set:
            outdict['geometry_coarse'] = self.emoca.deca.render.render_shape(values['verts'], values['trans_verts'])
        for key in outdict.keys():
            if outdict[key] is not None:
                assert(outdict[key].size(0) == codedict['shapecode'].size(0))
        return outdict

    """
    input: img: dict
        keys= image: tensor, image_name: list(str)
    """
    def inference(self, img, save_img=False):
        # img cropped
        if len(img['image'].shape) == 3:
            img['image'] = img['image'].view(1,3,224,224)
        
        codedict = self.encode(img)
        vals, visdict = self.decode(codedict)
        # name = f"{i:02d}"
        if save_img:
            name =  img['image_name'][0]
            sample_output_folder = Path(output_folder) / name
            sample_output_folder.mkdir(parents=True, exist_ok=True)
            save_images(output_folder, name, visdict, with_detection=True, i=j)
        
        return codedict, vals, visdict


class DANModel():
    def __init__(self, device):
        self.device = device
        self.labels = ['NEU', 'HAP', 'SAD', 'SUR', 'FEA', 'DIS', 'ANG']
        if os.path.exists('/home/chenyutong/facialanimation/DAN/dan_norm.pt'):
            print('load from new_norm.pt')
            self.out_norm = torch.load('/home/chenyutong/facialanimation/DAN/dan_norm.pt')['norm'].to(self.device)
            print('DAN label:', self.labels)
            print('DAN [mean, std]:', self.out_norm)
        else:
            self.out_norm = None
        self.data_transforms = transforms.Compose([
                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                    std=[0.229, 0.224, 0.225])
                                ])
        self.max_batch_size = 40 # 40
        self.tensor2img = transforms.ToPILImage()
        self.img2tensor = transforms.ToTensor()
        self.model = DAN(num_head=4, num_class=7, pretrained=False)
        self.model.load_state_dict(torch.load('/home/chenyutong/facialanimation/DAN/checkpoints/affecnet7_epoch6_acc0.6569.pth', map_location=self.device)['model_state_dict'],strict=True)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.to(device=self.device)
        self.model.eval()
    
    def set_norm(self, norm:torch.Tensor):
        # :,0 = mean, :,1 = std
        assert(norm.size() == (2,7))
        self.out_norm = norm.to(self.device)

    """
    input: img: batch,3,224,224
    output:
        label: str
        out(tensor): batch,7 not norm
    """
    def inference(self, img:torch.Tensor, norm=True):
        # large batch will cause out of memory error
        batch_size = img.size(0)
        assert(img.size()[1:] == (3,224,224))

        if batch_size > self.max_batch_size:
            result = torch.empty((batch_size, 7), device=self.device)
            now_batch = self.max_batch_size
            while now_batch < batch_size + self.max_batch_size:
                in_img = img[now_batch - self.max_batch_size:min(batch_size, now_batch),...]
                result[now_batch - self.max_batch_size:min(batch_size, now_batch),...] = self._inference(in_img)
                now_batch += self.max_batch_size
        else:
            result = self._inference(img)
        if norm and self.out_norm is not None:
            self.out_norm = self.out_norm.to()
            result = (result - self.out_norm[0,:]) / self.out_norm[1,:]
        return result

    def _inference(self, img:torch.Tensor):
        # RGB
        img = self.data_transforms(img)
        img = img.to(self.device)

        out, features, out_heads = self.model(img)
        #print('input size: ', img.size()) 1,3,224,224
        _, pred = torch.max(out,1)
        ind_list = pred.long().tolist()
        label = torch.LongTensor(data=ind_list)# [batch_size]
        # label = self.labels[ind_list]
        # label is not tensor but str
        return out


class FaceFormerConfig:
    def __init__(self, device):
        self.model_name = 'vocaset'
        self.dataset = 'vocaset'
        self.fps = 30
        self.feature_dim = 64
        self.period =  30
        self.vertice_dim = 5023*3
        self.device = device
        self.template_path = 'templates.pkl'
        self.train_subjects = "FaceTalk_170728_03272_TA FaceTalk_170904_00128_TA FaceTalk_170725_00137_TA FaceTalk_170915_00223_TA FaceTalk_170811_03274_TA FaceTalk_170913_03279_TA FaceTalk_170904_03276_TA FaceTalk_170912_03278_TA"
        self.val_subjects = "FaceTalk_170811_03275_TA FaceTalk_170908_03277_TA"
        self.test_subjects = "FaceTalk_170809_00138_TA FaceTalk_170731_00024_TA"
        self.subject = None # vertex template
        self.condition = 'FaceTalk_170728_03272_TA' # speaking style


class FaceFormerModel:
    def __init__(self, device):
        # import
        import plyfile
        import pickle
        from FaceFormer.faceformer import Faceformer
        from transformers import Wav2Vec2Processor
        # config
        config = FaceFormerConfig(device)
        self.config = config
        self.device = device
        # build model
        self.model = Faceformer(config)
        self.model.load_state_dict(torch.load(os.path.join('FaceFormer', config.dataset, '{}.pth'.format(config.model_name)), map_location=self.device))
        self.model = self.model.to(torch.device(device))
        self.model.eval()
        # init
        self.processor = Wav2Vec2Processor.from_pretrained("FaceFormer/facebook/wav2vec2-base-960h")
        
        
        self.template = {}
        if config.dataset.lower() == 'biwi':
            template_file = os.path.join('FaceFormer', config.dataset, config.template_path)
            with open(template_file, 'rb') as fin:
                templates = pickle.load(fin,encoding='latin1')
            for key in templates.keys():
                temp = templates[key].reshape((-1))
                temp = np.reshape(temp,(-1,temp.shape[0]))
                self.template[key] = torch.FloatTensor(temp).to(device=device)
        else:
            # load ply and conver to dict
            template_dir = r'/home/chenyutong/facialanimation/FaceFormer/vocaset/template_pkl'
            plys = sorted(os.listdir(template_dir))
            for ply in plys:
                plydata = plyfile.PlyData.read(os.path.join(template_dir, ply))
                tmp_tensor = torch.as_tensor(
                    np.asarray([plydata['vertex']['x'], plydata['vertex']['y'], plydata['vertex']['z']]).T.reshape((1, self.config.vertice_dim)), 
                    device=self.device) # N, 3
                self.template[ply.split('.')[0]] = tmp_tensor
        self.train_subjects_list = [i for i in config.train_subjects.split(" ")]
        self.one_hot_labels = np.eye(len(self.train_subjects_list))
        iter = self.train_subjects_list.index(config.condition)
        one_hot = self.one_hot_labels[iter]
        one_hot = np.reshape(one_hot,(-1,one_hot.shape[0]))
        self.one_hot = torch.FloatTensor(one_hot).to(device=device)

    def forward(self, data):
        speech_array = data['wav'] # 1, seq_len*16000
        audio_feature = np.squeeze(self.processor(speech_array,sampling_rate=16000).input_values)
        audio_feature = np.reshape(audio_feature,(-1,audio_feature.shape[0]))
        audio_feature = torch.FloatTensor(audio_feature).to(device=self.device)
        # use ground truth when possible
        if 'name' in data.keys(): 
            tmp = self.template[data['name'].split('=')[0]]
        else:
            tmp = self.template[self.config.subject]
        prediction = self.model.predict(audio_feature, tmp, self.one_hot)
        pred = prediction.permute(1,2,0).view(prediction.size(1), 5023,3)
        return pred



class VOCAModel:
    def __init__(self, device):
        self.device = device
        self.out_path = r'/home/chenyutong/facialanimation/VOCA/baseline_test'

    def forward(self, data):
        # load obj
        person, idx_str = data['name'].split('=')
        m_path = os.path.join(self.out_path, person, idx_str, 'meshes')
        arrs = []
        sorted_list = sorted(os.listdir(m_path))
        # 60fps to 30 pfs
        sorted_list = sorted_list[::2]
        # print(sorted_list)
        for name in sorted_list:
            arrs.append(Mesh.read_obj(os.path.join(m_path, name)))
        arrs = np.stack(arrs, axis=0)
        return torch.as_tensor(arrs).to(self.device)


class LSTMEMO:
    def __init__(self, device, model_name='lstm_emo'):
        from inference import Inference
        import importlib
        Model = importlib.import_module('Model.' + model_name + '.model').Model
        print('init inference')
        self.model_path = os.path.join('/home/chenyutong/facialanimation/Model', model_name,'saved_model')
        print('load model from', self.model_path)
        list_of_files = glob.glob(os.path.join(self.model_path, '*.pth')) # * means all if need specific format then *.csv
        latest_file = max(list_of_files, key=os.path.getctime)
        print('load model from: ', latest_file)
        self.infer = Inference('3d', device, latest_file, Model)
        self.flame = FLAMEModel(device)
        self.device = device

        # load ply and conver to dict
        template_dir = r'/home/chenyutong/facialanimation/FaceFormer/vocaset/template_pkl'
 
    def forward(self, data):
        params = self.infer.forward(data)['params'].squeeze(0)
        seq_len = params.size(0)
        #print('params:', params.size())
        codedict = {'shapecode':torch.zeros((seq_len, 100), device=self.device), 'expcode':params[:,:50], 'posecode':params[:,50:]}
        codedict['shapecode'] = data['shapecode'].to(self.device)
        out = self.flame.forward(codedict)
        return out

'''
test mesh convertion precision.
use converted ground truth as model output
'''
class BaselineConverter:
    def __init__(self, device):
        self.device = device
        self.data_path = '/home/chenyutong/facialanimation/dataset_cache/BIWI/fit_output/'
        self.flame = FLAMEModel(device)

    def forward(self, data):
        person, idx_str = data['name'].split('_')
        json_path = os.path.join(self.data_path, person, idx_str, 'params.json')
        with open(json_path, 'r') as f:
            params = torch.as_tensor(json.load(f)).to(self.device)
        seq_len = params.size(0)
        # print('params size: ', params.size())
        codedict = {'shapecode':params[:,:100], 'expcode':params[:,100:150], 'posecode':params[:,150:156]}
        return self.flame.forward(codedict)



