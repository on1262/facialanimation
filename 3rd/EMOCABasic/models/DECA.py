"""
Author: Radek Danecek
Copyright (c) 2022, Radek Danecek
All rights reserved.

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# Using this computer program means that you agree to the terms 
# in the LICENSE file included with this software distribution. 
# Any use not explicitly granted by the LICENSE is prohibited.
#
# Copyright©2022 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# For comments or questions, please email us at emoca@tue.mpg.de
# For commercial licensing contact, please contact ps-license@tuebingen.mpg.de

Parts of the code were adapted from the original DECA release: 
https://github.com/YadiraF/DECA/ 
"""


import os, sys
import torch
import torchvision
import torch.nn.functional as F
import numpy as np
# from time import time
from skimage.io import imread
import cv2
from pathlib import Path
#from pytorch_lightning import LightningModule
import decautils.DecaUtils as util
from models.Renderer import SRenderY
from models.DecaEncoder import ResnetEncoder
from models.DecaFLAME import FLAME, FLAMETex



#torch.backends.cudnn.benchmark = True
from enum import Enum


class DecaMode(Enum):
    COARSE = 1 # when switched on, only coarse part of DECA-based networks is used
    DETAIL = 2 # when switched on, only coarse and detail part of DECA-based networks is used 


class DecaModule(torch.nn.Module):

    def __init__(self, decoder_only=False):
        """
        :param model_params: a DictConfig of parameters about the model itself
        :param learning_params: a DictConfig of parameters corresponding to the learning process (such as optimizer, lr and others)
        :param inout_params: a DictConfig of parameters about input and output (where checkpoints and visualizations are saved)
        """
        super().__init__()


        # instantiate the network
        self.deca = ExpDECA(decoder_only=decoder_only)

        self.mode = DecaMode.COARSE

    def uses_texture(self):
        """
        Check if the model uses texture
        """
        return self.deca.uses_texture()

    def forward(self, batch):
        values = self.encode(batch)
        values = self.decode(values)
        return values

    def _encode_flame(self, images):
        # forward pass with gradients (for coarse stage (used), or detail stage with coarse training (not used))
        parameters = self.deca._encode_flame(images)

        code_list = self.deca.decompose_code(parameters)
        shapecode, texcode, expcode, posecode, cam, lightcode = code_list
        return shapecode, texcode, expcode, posecode, cam, lightcode

    def encode(self, batch, return_img=True) -> dict:
        """
        Forward encoding pass of the model. Takes a batch of images and returns the corresponding latent codes for each image.
        :param batch: Batch of images to encode. batch['image'] [batch_size, 3, image_size, image_size]. ndarray
        """
        codedict = {}
        original_batch_size = batch['image'].shape[0]

        images = batch['image']
        

        # [B, 3, size, size]
        images = images.view(-1, images.shape[-3], images.shape[-2], images.shape[-1])


        # 1) COARSE STAGE
        # forward pass of the coarse encoder (use flame)
        shapecode, texcode, expcode, posecode, cam, lightcode = self._encode_flame(images) # ExpDECA encode, expcode replaced

        codedict['shapecode'] = shapecode
        codedict['texcode'] = texcode
        codedict['expcode'] = expcode
        codedict['posecode'] = posecode
        codedict['cam'] = cam
        codedict['lightcode'] = lightcode
        if return_img:
            codedict['images'] = images

        return codedict

    def decode(self, codedict, return_set:set) -> dict:
        """
        Forward decoding pass of the model. Takes the latent code predicted by the encoding stage and reconstructs and renders the shape.
        :param codedict: Batch dict of the predicted latent codes
        :param training: Whether the forward pass is for training or testing.
        """
        shapecode = codedict['shapecode']
        expcode = codedict['expcode']
        posecode = codedict['posecode']
        texcode = codedict['texcode'] if 'texcode' in codedict.keys() else None
        cam = codedict['cam'] if 'cam' in codedict.keys() else None
        lightcode = codedict['lightcode'] if 'lightcode' in codedict.keys() else None
        images = codedict['images'] if 'images' in codedict.keys() else None
        #print('img max', images.max(), 'min', images.min())
        effective_batch_size = shapecode.size(0) 

        # 1) Reconstruct the face mesh
        # FLAME - world space
        if 'decode_verts' in codedict.keys():
            verts = codedict['decode_verts']
            print('Warning: use input verts in EMOCA decoding process')
        else:
            verts, _, _ = self.deca.flame(shape_params=shapecode, expression_params=expcode,
                                                          pose_params=posecode)
        if 'verts' in return_set:
            codedict['verts'] = verts
        
        if 'geo' in return_set or 'coarse' in return_set:
            # world to camera
            trans_verts = util.batch_orth_proj(verts, cam)
            # camera to image space
            trans_verts[:, :, 1:] = -trans_verts[:, :, 1:]
        
        if 'geo' in return_set:
            codedict['trans_verts'] = trans_verts

        if 'coarse' in return_set: # performance consideration
            #print('EMOCA: use texture=', self.uses_texture())
            if self.uses_texture() and texcode is not None:
                albedo = self.deca.flametex(texcode) # 0->1
            else: 
                # if not using texture, default to gray
                albedo = torch.ones([effective_batch_size, 3, 256, 256], device=shapecode.device) * 0.5

            # 2) Render the coarse image
            ops = self.deca.render(verts, trans_verts, albedo, lightcode)

            if images is not None:
                # mask
                mask_face_eye = F.grid_sample(self.deca.uv_face_eye_mask.expand(effective_batch_size, -1, -1, -1),
                                              ops['grid'].detach(),
                                              align_corners=False)

                masks = mask_face_eye * ops['alpha_images']
                #print('masks max', masks.max())

            # images
            predicted_images = ops['images'] # 0-1
            #print('pred image max', predicted_images.max())

            # add background or black background
            if images is not None:
                if images.shape[-1] != predicted_images.shape[-1] or images.shape[-2] != predicted_images.shape[-2]:
                    ## special case only for inference time if the rendering image sizes have been changed
                    images_resized = F.interpolate(images, size=predicted_images.shape[-2:], mode='bilinear')
                    predicted_images = (1. - masks) * images_resized + masks * predicted_images
                else:
                    predicted_images = (1. - masks) * images + masks * predicted_images


            # populate the value dict for metric computation/visualization
            codedict['predicted_images'] = predicted_images

            if images is not None:
                codedict['mask_face_eye'] = mask_face_eye
                codedict['masks'] = masks


        return codedict
    
class DECA(torch.nn.Module):
    """
    The original DECA class which contains the encoders, FLAME decoder and the detail decoder.
    """

    def __init__(self, decoder_only):
        """
        :config corresponds to a model_params from DecaModule
        """
        super().__init__()
        self.flame = FLAME()
        self.flametex = FLAMETex()
        if not decoder_only:
            self._create_model()
        self._setup_renderer()


    def _setup_renderer(self):
        self.render = SRenderY(224, obj_filename='/home/chenyutong/facialanimation/EMOCABasic/data/FLAME/geometry/head_template.obj', uv_size=256)  # .to(self.device)
        # face mask for rendering details
        mask = imread('/home/chenyutong/facialanimation/EMOCABasic/data/FLAME/mask/uv_face_mask.png').astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        self.uv_face_mask = F.interpolate(mask, [256,256])
        mask = imread('/home/chenyutong/facialanimation/EMOCABasic/data/FLAME/mask/uv_face_eye_mask.png').astype(np.float32) / 255.
        mask = torch.from_numpy(mask[:, :, 0])[None, None, :, :].contiguous()
        uv_face_eye_mask = F.interpolate(mask, [256, 256])
        self.register_buffer('uv_face_eye_mask', uv_face_eye_mask)


    def uses_texture(self): 
        return True # true by default

    def _create_model(self):
        self.E_flame = ResnetEncoder(outsize=236)
    
    def _get_coarse_trainable_parameters(self):
        print("Add E_flame.parameters() to the optimizer")
        return list(self.E_flame.parameters())


    def _encode_flame(self, images):
        return self.E_flame(images)

    def decompose_code(self, code):
        '''
        config.n_shape + config.n_tex + config.n_exp + config.n_pose + config.n_cam + config.n_light
        '''
        code_list = []
        num_list = [100, 50, 50, 6, 3, 27]
        start = 0
        for i in range(len(num_list)):
            code_list.append(code[:, start:start + num_list[i]])
            start = start + num_list[i]
        # shapecode, texcode, expcode, posecode, cam, lightcode = code_list
        code_list[-1] = code_list[-1].reshape(code.shape[0], 9, 3)
        return code_list


class ExpDECA(DECA):
    """
    This is the EMOCA class (previously ExpDECA). This class derives from DECA and add EMOCA-related functionality. 
    Such as a separate expression decoder and related.
    """

    def _create_model(self):
        # 1) Initialize DECA
        super()._create_model()
        # E_flame should be fixed for expression EMOCA
        self.E_flame.requires_grad_(False)
        
        # 2) add expression decoder
        ## b) Clones the original DECA coarse decoder (and the entire decoder will be trainable) - This is in final EMOCA.
        #TODO this will only work for Resnet. Make this work for the other backbones (Swin) as well.
        self.E_expression = ResnetEncoder(50)
        # clone parameters of the ResNet
        self.E_expression.encoder.load_state_dict(self.E_flame.encoder.state_dict())
        # expression module is fixed
        self.E_expression.requires_grad_(False)

    def _get_coarse_trainable_parameters(self):
        print("Add E_expression.parameters() to the optimizer")
        return list(self.E_expression.parameters())

    def _encode_flame(self, images):
        # other regressors have to do a separate pass over the image
        deca_code = super()._encode_flame(images)
        exp_deca_code = self.E_expression(images)
        return deca_code, exp_deca_code

    def decompose_code(self, code):
        deca_code = code[0]
        expdeca_code = code[1]

        deca_code_list = super().decompose_code(deca_code)
        # shapecode, texcode, expcode, posecode, cam, lightcode = deca_code_list

        deca_code_list[2] = expdeca_code

        return deca_code_list