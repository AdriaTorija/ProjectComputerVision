import pathlib
import torch
import tqdm
import click
import numpy as np
import cv2
import tops
from ssd import utils
from tops.config import instantiate
from PIL import Image
from vizer.draw import draw_boxes
from tops.checkpointer import load_checkpoint
from pathlib import Path
import torch.nn as nn
from collections import OrderedDict
    
#CAM IMPORTS
import torchvision
from pytorch_grad_cam import AblationCAM, EigenCAM, EigenGradCAM
from pytorch_grad_cam.ablation_layer import AblationLayerFasterRCNN
from pytorch_grad_cam.utils.model_targets import FasterRCNNBoxScoreTarget
from pytorch_grad_cam.utils.image import show_cam_on_image, scale_accross_batch_and_channels, scale_cam_image
from pytorch_grad_cam.utils.reshape_transforms import fasterrcnn_reshape_transform
import torch

tags = {0: 'background', 1: 'car', 2: 'truck', 3: 'bus', 4: 'motorcycle', 5: 'bicycle', 6: 'scooter', 7: 'person', 8: 'rider'}

def renormalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam):
    """Normalize the CAM to be in the range [0, 1] 
    inside every bounding boxes, and zero outside of the bounding boxes. """
    renormalized_cam = np.zeros(grayscale_cam.shape, dtype=np.int)
    images = []
    for x1, y1, x2, y2 in boxes:
        img = renormalized_cam * 0
        x1=abs(x1)
        x2=abs(x2)
        y1=abs(y1)
        y2=abs(y2)

        img[y1:y2, x1:x2] = scale_cam_image(grayscale_cam[y1:y2, x1:x2].copy())    
        images.append(img)
    
    renormalized_cam = np.max(np.float32(images), axis = 0)
    renormalized_cam = scale_cam_image(renormalized_cam)
    eigencam_image_renormalized = show_cam_on_image(image_float_np, renormalized_cam, use_rgb=True)
    #image_with_bounding_boxes = draw_boxes(eigencam_image_renormalized,boxes, labels, tags)
    return eigencam_image_renormalized


def reshape_transform(x):
  
    target_size = x['feat5'].size()[-2 : ]
    activations = []
    for key in x:
        value=x[key]
        activations.append(torch.nn.functional.interpolate(torch.abs(value), target_size, mode='bilinear'))
    activations = torch.cat(activations, axis=1)
    return activations
   

class RetinaNetModelOutputWrapper(nn.Module):
    def __init__(self, model): 
        super(RetinaNetModelOutputWrapper, self).__init__()
        self.model = model
        
    def forward(self, x):
        b, l, s =self.model(x)[0]
        d = OrderedDict()
        d["boxes"]=b
        d["labels"]=l
        d["scores"]=s

        return [d]
    
@torch.no_grad()
@click.command()
@click.argument("config_path", type=click.Path(exists=True, dir_okay=False, path_type=str))
@click.argument("video_path", type=click.Path(dir_okay=True, path_type=str))
@click.argument("output_path", type=click.Path(dir_okay=True, path_type=str))
@click.option("-s", "--score_threshold", type=click.FloatRange(min=0, max=1), default=.3)
def run_demo(config_path: str, score_threshold: float, video_path: str, output_path: str):
    cfg = utils.load_config(config_path)
    model = tops.to_cuda(instantiate(cfg.model))
    
    wrapper_model= RetinaNetModelOutputWrapper(model)
    wrapper_model.model.eval()
    ckpt = load_checkpoint(cfg.output_dir.joinpath("checkpoints"), map_location=tops.get_device())
    wrapper_model.model.load_state_dict(ckpt["model"])
    width, height = 1024, 128

    reader = cv2.VideoCapture(video_path) 
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    writer = cv2.VideoWriter(output_path, fourcc, 30, (width, height))
    cpu_transform = instantiate(cfg.data_val.dataset.transform)
    gpu_transform = instantiate(cfg.data_val.gpu_transform)
    video_length = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

    cam = EigenCAM(wrapper_model,
            [wrapper_model.model.feature_extractor.fpn], 
            use_cuda=torch.cuda.is_available(),
            reshape_transform=reshape_transform)
    cam.uses_gradients=False
    assert reader.isOpened()
    for frame_idx in tqdm.trange(video_length, desc="Predicting on video"):
        ret, frame = reader.read()
        assert ret, "An error occurred"
        frame = np.ascontiguousarray(frame[:, :, ::-1])
        image_float_np = np.float32(frame) / 255
        img = cpu_transform({"image": frame})["image"].unsqueeze(0)
        img = tops.to_cuda(img)
        img = gpu_transform({"image": img})["image"]
        
        #CAM 
        d= wrapper_model(img)
        targets = [FasterRCNNBoxScoreTarget(labels=d[0]['labels'], bounding_boxes=d[0]['boxes'])]

        #Here it crashes
        grayscale_cam = cam(img, targets=targets)   
        grayscale_cam = grayscale_cam[0, :]
        cam_image = show_cam_on_image(image_float_np, grayscale_cam, use_rgb=True)
        
        boxes=d[0]['boxes']
        scores=d[0]["scores"]
        labels=d[0]['labels']
        boxes[:, [0, 2]] *= width
        boxes[:, [1, 3]] *= height
        boxes, labels, scores = [_.cpu().numpy() for _ in [boxes, labels, scores]]
        boxes = boxes.astype(int)
        
        eng=renormalize_cam_in_bounding_boxes(boxes, image_float_np, grayscale_cam)
        frame = draw_boxes(
            eng, boxes, labels, scores).astype(np.uint8)
        writer.write(frame[:, :, ::-1])
    print("Video saved to:", pathlib.Path(output_path).absolute())
        
        
if __name__ == '__main__':
    run_demo()