# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import argparse
import glob
import multiprocessing as mp
import os
import time
import cv2
import tqdm

import torch
import numpy as np

import json
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.data import MetadataCatalog
from detectron2.data.detection_utils import read_image
from detectron2.utils.logger import setup_logger
from detectron2.structures.boxes import Boxes, pairwise_iou
from detectron2.structures.instances import Instances
from detectron2.utils.visualizer import Visualizer, _create_text_labels
from detectron2.utils.colormap import colormap
from detectron2.structures import BoxMode, PolygonMasks, Boxes
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
)

import sys
sys.path.append('/home/irecorte/sianms/frustum-pointnets/nuscenes/')
import nuscenes_object
import secrets

bev_im = np.ones((1400,1400,3)) * 255

# constants
WINDOW_NAME = "nuscenes val"
# category_dict = {'Pedestrian' : 0,
#               'Car' :  1,
#               'Cyclist' : 2,
#               # 'Motorcycle' :  2,
#               # 'Motorcyclist' : 3,
#               # 'Bicycle' :  4,
#               # 'Bicyclist' : 5,
#               # 'Bus' :  6,
#               # 'Truck' : 7, 
#               # 'Construction' :  8,
#               # # 'Van' :  9,
#               # 'Trailer' : 9,
#               # 'Barrier' :  10,
#               # 'Cone' : 11
#               }
# if nuscyc:
category_dict = {'Pedestrian' : 0,
          'Car' :  1,
          'Cycle' :  2,
          'Cyclist' : 3,
          'Bus' :  4,
          'Truck' : 5, 
          'Construction' :  6,
          # 'Van' :  9,
          'Trailer' : 7,
          'Barrier' :  8,
          'Cone' : 9
          }


def setup_cfg(args):
    # load config from file and command-line arguments
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    # Set score_threshold for builtin models
    cfg.MODEL.RETINANET.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = args.confidence_threshold
    cfg.MODEL.PANOPTIC_FPN.COMBINE.INSTANCES_CONFIDENCE_THRESH = args.confidence_threshold
    cfg.freeze()
    return cfg


def get_parser():
    parser = argparse.ArgumentParser(description="Detectron2 Demo")
    parser.add_argument(
        "--config-file",
        default="configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
        metavar="FILE",
        help="path to config file",
    )
    parser.add_argument("--input", nargs="+", help="A list of space separated input images")
    parser.add_argument("--output_dir", default=None)
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.5,
        help="Minimum score for instance predictions to be shown",
    )
    parser.add_argument(
        "--opts",
        help="Modify config options using the command-line 'KEY VALUE' pairs",
        default=[],
        nargs=argparse.REMAINDER,
    )
    return parser

def get2d2box(box):

    xmin = min(box[0])
    xmax = max(box[0])
    ymin = min(box[1])
    ymax = max(box[1])

    return Boxes(torch.as_tensor([[xmin, ymin, xmax, ymax]], dtype=torch.float32, device='cuda'))

def get2d2boxes(boxes):
    list_boxes = []

    for box in boxes:
        xmin = min(box[0])
        xmax = max(box[0])
        ymin = min(box[1])
        ymax = max(box[1])
        list_boxes.append([xmin, ymin, xmax, ymax])

    return Boxes(torch.as_tensor(list_boxes, dtype=torch.float32, device='cuda'))

def main(args):
    global bev_im
    mp.set_start_method("spawn", force=True)
    args = get_parser().parse_args()
    logger = setup_logger()
    logger.info("Arguments: " + str(args))

    cfg = setup_cfg(args)
    view = True

    predictor = DefaultPredictor(cfg)
    metadata = MetadataCatalog.get(cfg.DATASETS.TEST)

    f_rgb_detections = open('nuscenes_rgb_detections.txt', "a")


    dataset = nuscenes_object.nuscenes_object('/raid/datasets/extracted_nuscenes', split='val', velo_kind='lidar_top')
    if not os.path.exists(os.path.join(args.output_dir)):
        os.mkdir(os.path.join(args.output_dir))
    if not os.path.exists(os.path.join(args.output_dir, 'data')):
        os.mkdir(os.path.join(args.output_dir, 'data'))

    current_scene = 0
    current_time = 0

    for idx in range(0,len(dataset)):
        name = dataset.get_idx_name(idx)[1:]
        
        if current_scene == int(name[:4]) and current_time >= int(name[-4:]):
            continue
        else:
            current_scene = int(name[:4])
            current_time = int(name[-4:])
   
        print(name)

        ims = []
        for ii in range(0,6):
            # print(ii)
            im = dataset.get_image_by_name(str(ii)+name)
            im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)

            predictions = predictor(im)[0]
            print(predictions)
     
            instances = predictions["instances"].to(torch.device("cpu"))
            #draw the detections over the original images
            visualizer = Visualizer(im, metadata)

            classes = instances.pred_classes.cpu().numpy()
            class_names = visualizer.metadata.get("thing_classes")
            # print(instances.pred_classes)
            # print(class_names)
            # print(classes)
            labels = [class_names[i] for i in classes]
            vis_colors = [colormap(rgb=True, maximum=1)[i] if i < 74 else (0,0,0) for i in classes]

            if (view):
                visualizer.overlay_instances(
                    boxes=instances.pred_boxes,
                    # masks=instances.pred_masks,
                    labels=_create_text_labels(instances.pred_classes, \
                        instances.scores, \
                        visualizer.metadata.get("thing_classes", None)),
                        # ['Car', 'Pedestrian', 'Cyclist', 'Motorcyclist']),
                    assigned_colors=vis_colors,
                    alpha=0.5,
                )
            
            for jj in range(len(instances)):

                bbox = instances.pred_boxes[jj].get_numpy()[0]
                output_str = os.path.join(dataset.image_dir, '%s.jpg'%(str(ii)+name)) + " %s %f %.2f %.2f %.2f %.2f\n" %(
                    labels[jj], instances.scores[jj].cpu().numpy(), bbox[0], bbox[1], bbox[2], bbox[3])
                
                # print(output_str)
                f_rgb_detections.write(output_str)

                det_filename = os.path.join(args.output_dir, 'data', '%s.txt' %(str(ii)+name)) 
                with open(det_filename,'a+') as f:
                    bbox = instances.pred_boxes[jj].get_numpy()[0]
                    output_eval = '%s -1 -1 -10 %.3f %.3f %.3f %.3f -1 -1 -1 -1 -1 -1 -1 %.3f\n' %\
                                (labels[jj], bbox[0], bbox[1], bbox[2], bbox[3], instances.scores[jj].cpu().numpy())
                    f.write(output_eval)
                    # print(output_eval)

            if (view):
                im_view = np.array(visualizer.output.get_image()[:, :, ::-1])
                # im_v = cv2.rectangle(im_view, (0,0), (im_view.shape[1], im_view.shape[0]), colors[ii], thickness = 30)
                if (ii == 0):
                    ims = []
                ims.append(im_view)

        if (view):
            h1 = cv2.hconcat((ims[1], ims[0], ims[2]))
            h2 = cv2.hconcat((ims[5], ims[3], ims[4]))
            v1 = cv2.vconcat((h1, h2))

            cv2.namedWindow('6im', cv2.WINDOW_NORMAL)
            cv2.imshow('6im', v1)

        if (view):
            key = cv2.waitKey(0)

            if key == 115:
                cv2.imwrite('%s_6im.png' %name, v1)
                cv2.imwrite('%s_bev_im.png' %name, bev_im)
                print('SAVING IMAGES')
            if key == 27:
                break  # esc to quit

if __name__ == "__main__":
    main([])
