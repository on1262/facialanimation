from facenet_pytorch import MTCNN
import torch
import os
from PIL import Image
import torchvision.transforms.functional as F
from skimage.transform import rescale, estimate_transform, warp
import numpy as np
from threading import Thread

# FAN is a multi-scale face detector, which is better than MTCNN
class FANDetector():

    def __init__(self, device, crop_size=224, scaling_factor=1.0, scale=1.25, threshold=0.5):
        import face_alignment
        self.face_detector = 'sfd'
        self.face_detector_kwargs = {
            "filter_threshold": threshold
        }
        self.device = device
        self.scaling_factor = scaling_factor
        self.crop_size = crop_size
        self.max_thread = 1
        self.scale = scale
        self.resolution_inp = crop_size
        self.model = face_alignment.FaceAlignment(face_alignment.LandmarksType._2D,
                                                  device=str(device),
                                                  flip_input=False,
                                                  face_detector=self.face_detector,
                                                  face_detector_kwargs=self.face_detector_kwargs)
    def process_item(self, idx, img, result_dict):
        result_dict[idx] = self._crop(img)

    def crop(self, imgs):
        if isinstance(imgs, (torch.Tensor, np.ndarray)) and imgs.dim() == 3:
            return self._crop(imgs).unsqueeze(0)

        if isinstance(imgs, list):
            imgs = torch.stack(imgs, dim=0)
            if imgs.dim() > 4:
                imgs = imgs.squeeze()
        # multi-process
        result_dict = {}
        process_list = [Thread(target=self.process_item, args=(idx, imgs[idx,...], result_dict)) for idx in range(imgs.size(0))]
        for idx in range(imgs.size(0)):
            if idx < self.max_thread:
                process_list[idx].start()
            else:
                process_list[idx-self.max_thread].join()
                process_list[idx].start()
        for idx in range(imgs.size(0)):
            process_list[idx-self.max_thread].join()
        for idx in range(imgs.size(0)):
            if result_dict[idx] is None:
                return None
        
        result_list = [result_dict[idx] for idx in range(imgs.size(0))]

        return torch.stack(result_list, dim=0)

    
    def _crop(self, image):
        if len(image.shape) == 3 and image.shape[2] > 3:
            image = image[:, :, :3]
        if self.scaling_factor != 1.:
            image = rescale(image, (self.scaling_factor, self.scaling_factor, 1))*255.
        h, w, _ = image.shape
        # bbox, bbox_type, landmarks = self.face_detector.run(image)
        bbox, bbox_type = self.run(image)
        if len(bbox) < 1:
            print('no face detected! return None')
            return None
            left = 0
            right = h - 1
            top = 0
            bottom = w - 1
            old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)
        else:
            bbox = bbox[0]
            left = bbox[0]
            right = bbox[2]
            top = bbox[1]
            bottom = bbox[3]
            old_size, center = self.bbox2point(left, right, top, bottom, type=bbox_type)

        size = int(old_size * self.scale)
        src_pts = np.array(
            [[center[0] - size / 2, center[1] - size / 2], [center[0] - size / 2, center[1] + size / 2],
            [center[0] + size / 2, center[1] - size / 2]])

        image = image / 255.
        DST_PTS = np.array([[0, 0], [0, self.resolution_inp - 1], [self.resolution_inp - 1, 0]])
        tform = estimate_transform('similarity', src_pts, DST_PTS)
        dst_image = warp(image, tform.inverse, output_shape=(self.resolution_inp, self.resolution_inp))
        dst_image = dst_image.transpose(2, 0, 1)
        dst_image = torch.tensor(dst_image).float()
        return dst_image

    def run(self, image, with_landmarks=False, detected_faces=None):
        '''
        image: 0-255, uint8, rgb, [h, w, 3]
        return: detected box list
        '''
        out = self.model.get_landmarks(image, detected_faces=detected_faces)
        #torch.cuda.empty_cache()
        if out is None:
            del out
            if with_landmarks:
                return [], 'kpt68', []
            else:
                return [], 'kpt68'
        else:
            boxes = []
            kpts = []
            for i in range(len(out)):
                kpt = out[i].squeeze()
                left = np.min(kpt[:, 0])
                right = np.max(kpt[:, 0])
                top = np.min(kpt[:, 1])
                bottom = np.max(kpt[:, 1])
                bbox = [left, top, right, bottom]
                boxes += [bbox]
                kpts += [kpt]
            del out # attempt to prevent memory leaks
            if with_landmarks:
                return boxes, 'kpt68', kpts
            else:
                return boxes, 'kpt68'

    def bbox2point(self, left, right, top, bottom, type='bbox'):
        ''' bbox from detector and landmarks are different
        '''
        if type == 'kpt68':
            old_size = (right - left + bottom - top) / 2 * 1.1
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0])
        elif type == 'bbox':
            old_size = (right - left + bottom - top) / 2
            center = np.array([right - (right - left) / 2.0, bottom - (bottom - top) / 2.0 + old_size * 0.12])
        else:
            raise NotImplementedError
        return old_size, center

class MTCNNDetector():
    def __init__(self, device, scale_up=1.0):
        self.device = device
        # poseprocess: convert [0,255] to [-127, 127], not used
        self.scale_up = scale_up
        self.mtcnn = MTCNN(image_size=224, device=device, min_face_size=100, select_largest=False, post_process=False)
    
    # input format: hw3RGB255_tu
    # output: tensor
    def crop(self, input):
        '''
        input: PIL image, tensor, ndarray, list of above, dir path, file path
        output: batch*3*224*224 (single: batch=1)
        note: int tensor has no grad, do not use crop after flame output
        '''
        bbox = None
        if isinstance(input, list):
            if isinstance(input[0], torch.Tensor):
                input = torch.stack(input, dim=0) # b3wh->bwh3, fit for mtcnn
        elif isinstance(input, str) and os.path.isdir(input):
            result = []
            for root, _, files in os.walk(input):
                for file in sorted(files):
                    result.append(torch.as_tensor(Image.open(os.path.join(root, file))))
            input = torch.stack(result, dim=0)
        elif isinstance(input, str) and os.path.isfile(input):
            input = torch.as_tensor(Image.open(os.path.join(root, file))).unsqueeze(0)
        elif isinstance(input, np.ndarray):
            input = torch.as_tensor(input)
            input = input.unsqueeze(0) if input.dim() == 3 else input
        else: # tensor
            input = input.unsqueeze(0) if input.dim() == 3 else input
        
        if input.size(1) == 3:
            input = input.permute(0, 2, 3, 1) # hw3
        

        input = input.cpu()
        #print('input size', input.size())
        # Detect faces
        batch_boxes, batch_probs, batch_points = self.mtcnn.detect(input, landmarks=True)
        # Select faces
        batch_boxes, batch_probs, batch_points = self.mtcnn.select_boxes(
            batch_boxes, batch_probs, batch_points, input, method='probability')
        # maually crop to avoid different scale factors between x and y axis
        #print(batch_boxes, type(batch_boxes))
        for idx in range(batch_boxes.shape[0]):
            if batch_boxes[idx] is not None:
                c_h, c_w = (batch_boxes[idx][0, 3] + batch_boxes[idx][0,1]) / 2, (batch_boxes[idx][0, 2] + batch_boxes[idx][0,0]) / 2
            else:
                print('Warning: no face detected in frame ', idx, ' use nearby image')
                batch_boxes[idx] = np.asarray([[0,0,0,0]])
                if idx >= 1 and batch_boxes[idx-1] is not None:
                    c_h, c_w = (batch_boxes[idx-1][0, 3] + batch_boxes[idx-1][0,1]) / 2, (batch_boxes[idx-1][0, 2] + batch_boxes[idx-1][0,0]) / 2
                elif idx+1 < batch_boxes.shape[0] and batch_boxes[idx+1] is not None:
                    c_h, c_w = (batch_boxes[idx+1][0, 3] + batch_boxes[idx+1][0,1]) / 2, (batch_boxes[idx+1][0, 2] + batch_boxes[idx+1][0,0]) / 2
                else:
                    print('Error in using nearby image, use center image')
                    c_h, c_w = input.size(-3) / 2, input.size(-2) / 2
            batch_boxes[idx][0,1] = c_h - 112
            batch_boxes[idx][0,3] = c_h + 112
            batch_boxes[idx][0,2] = c_w + 112
            batch_boxes[idx][0,0] = c_w - 112

        #img[box[1]:box[3], box[0]:box[2]]
        # Extract faces
        out = self.mtcnn.extract(input, batch_boxes, save_path=None)

        # scale up
        if isinstance(out, list):
            out = torch.stack(out, dim=0)
        if out.size(-1) == 3:
            out = out.permute(0, 3, 1, 2)

        img_up = F.resize(out, round(224*self.scale_up))
        crop_min = round(112*self.scale_up) - 112
        crop_max = crop_min + 224
        #print('img up:', img_up.size())
        offset = round(23.5 * self.scale_up) # make sure jaw part is included
        out = img_up[:,:, crop_min+offset:crop_max+offset, crop_min:crop_max].detach().clone()
        del img_up
        return out
        
        
if __name__ == '__main__':
    from converter import video2sequence, save_img, convert_img
    imgs_list = video2sequence('./test/test_0.jpg', sample_step=10, return_path=False, o_code='mtcnn')
    imgs_list += video2sequence('./test/test_1.jpg', sample_step=10, return_path=False, o_code='mtcnn')

    img_tensor = torch.stack(imgs_list, dim=0) # 3hwRGB
    print(img_tensor.size())
    mctnn = MTCNNDetector(device=torch.device('cuda:0'), scale_up=1.7)
    out = mctnn.crop(img_tensor)
    save_img(convert_img(out, i_code='mtcnn', o_code='tvsave'), './test/frame0_crop.png')
    print('Done')
