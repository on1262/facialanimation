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
"""
import sys
from ImageTestDataset import TestData
from .models.DECA import DecaModule
import numpy as np
import os
import torch
from skimage.io import imsave, imread
from pathlib import Path
from tqdm import auto
import argparse
import logging
from decautils.lightning_logging import _fix_image


def torch_img_to_np(img):
    return img.detach().cpu().numpy().transpose(1, 2, 0)

def save_images(outfolder, name, vis_dict, i = 0, with_detection=False):
    prefix = None
    final_out_folder = Path(outfolder) / name
    final_out_folder.mkdir(parents=True, exist_ok=True)

    if with_detection:
        imsave(final_out_folder / f"inputs.png",  _fix_image(torch_img_to_np(vis_dict['inputs'][i])))
    imsave(final_out_folder / f"geometry_coarse.png",  _fix_image(torch_img_to_np(vis_dict['geometry_coarse'][i])))
    imsave(final_out_folder / f"out_im_coarse.png", _fix_image(torch_img_to_np(vis_dict['output_images_coarse'][i])))


def decode(emoca, values, training=False):
    with torch.no_grad():
        values = emoca.decode(values)
        batch_size = values['verts'].shape[0]
        visind = np.arange(batch_size)
        visdict = {}
        visdict['inputs'] = values['images'][visind]
        visdict['output_images_coarse'] = values['predicted_images'][visind]
        shape_images = emoca.deca.render.render_shape(values['verts'], values['trans_verts'])
        visdict['geometry_coarse'] = shape_images[visind]

    return values, visdict

def main():
    logging.getLogger("lightning").setLevel(logging.ERROR)
    parser = argparse.ArgumentParser()
    # add the input folder arg 
    parser.add_argument('--input_folder', type=str)
    parser.add_argument('--output_folder', type=str, default="image_output")

    args = parser.parse_args()
    input_folder = args.input_folder
    output_folder = args.output_folder

    # 1) Load the model
    #emoca = DecaModule.load_from_checkpoint(checkpoint_path='data/emoca.ckpt', strict=False)
    #torch.save(emoca.state_dict(), 'data/emoca_basic.ckpt')
    emoca = DecaModule()
    emoca.load_state_dict(torch.load('data/emoca_basic.ckpt'))
    emoca.cuda()
    emoca.eval()

    # 2) Create a dataset
    dataset = TestData(input_folder, face_detector="fan", max_detection=20, iscrop=True)

    ## 4) Run the model on the data

    #img['image'] = torch.from_numpy(imread(input_folder)).float().permute(2,0,1).unsqueeze(0) / 255 # 3hwRGB01
    img = dataset[0]
    img['image_name'] = 'test_image'
    img['image'] = img['image'].cuda()
    if len(img['image'].shape) == 3:
        img['image'] = img['image'].view(1,3,224,224)
    #img['image'] = img['image'].expand(2,-1,-1,-1)
    vals = emoca.encode(img)
    vals, visdict = decode(emoca, vals)
    # name = f"{i:02d}"
    current_bs = img['image'].shape[0]

    for j in range(current_bs):
        name =  img["image_name"]

        sample_output_folder = Path(output_folder) / name
        sample_output_folder.mkdir(parents=True, exist_ok=True)

        save_images(output_folder, name, visdict, with_detection=True, i=j)

    print("Done")


if __name__ == '__main__':
    main()
