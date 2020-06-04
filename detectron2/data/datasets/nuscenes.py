# -*- coding: utf-8 -*-
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import numpy as np
import os
import xml.etree.ElementTree as ET
from fvcore.common.file_io import PathManager

from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.structures import BoxMode
from PIL import Image
import math
import json

__all__ = ["register_nuscenes"]


# fmt: off
CLASS_NAMES = [
    "Pedestrian", "Car", "Cycle", "Cyclist", "Bus", 
    "Truck", "Construction", "Trailer",
    "Barrier", "Cone",
]
# fmt: on
category_dict = {'Pedestrian' : 0,
              'Car' :  1,
              'Bicycle' :  2,
              'Motorcycle' :  2,
              'Bicyclist' : 3,
              'Motorcyclist' : 3,
              'Bus' :  4,
              'Truck' : 5, 
              'Construction' :  6,
              'Trailer' : 7,
              'Barrier' :  8,
              'Cone' : 9,
              'Van' :  -1,
              }

def is_vehicle(num):
    return category_dict['Barrier'] != num and category_dict['Cone'] != num

def load_nuscenes_instances(data_dir, out_dir, subsets, subsets_files):
    print("VAL FILE: %s" % val_file)
    print("TRAIN FILE: %s\n" % train_file)

    json_name = 'annotations_nuscenes_%s.json'

    img_id = 0
    ann_id = 0

    for sub_set, subset_name in zip(subsets, subsets_files):
        ann_dir = os.path.join(data_dir, 'label')
        im_dir = os.path.join(data_dir, 'image')  #images_dataset
        print('Starting %s' % ann_dir)
        print('Starting %s\n' % im_dir)

        dataset_dicts = []
        rings = {}

        with open(subset_name, "r") as f:
            im_list = f.read().splitlines()
            for filename in im_list:
                if not anns_in_ring(ann_dir, filename[1:]):
                  # print('Empty ring, continuing.')
                  continue
                complete_name_im = os.path.join(im_dir, filename + '.jpg')
                complete_name_ann = os.path.join(ann_dir, filename + '.txt')
                
                if not os.path.exists(complete_name_im): 
                    continue

                record = {}
                record['image_id'] = img_id
                img_id += 1

                im = Image.open(complete_name_im)
                # print(im.size)
                record['width'] = int(im.size[0])
                record['height'] = int(im.size[1])

                record['file_name'] = complete_name_im
                # images.append(image)                             

                if os.path.exists(complete_name_ann): 
                  pre_objs = np.genfromtxt(complete_name_ann, delimiter=' ',
                      names=['token', 'type', 'truncated', 'occluded', 'alpha', 'bbox_xmin', 'bbox_ymin',
                      'bbox_xmax', 'bbox_ymax', 'dimensions_1', 'dimensions_2', 'dimensions_3',
                      'location_1', 'location_2', 'location_3', 'rotation_y'], dtype=None, encoding='ascii')

                  if (pre_objs.ndim < 1):
                      pre_objs = np.array(pre_objs, ndmin=1)
                else:
                  pre_objs= []
                annotations = []

                for obj in pre_objs:

                    if (category_dict[obj['type']] != -1):
                        ann = {}
                        ann_id += 1
                        ann['token'] = obj['token']
                        ann['category_id'] = int(category_dict[obj['type']])
                        ann['bbox'] = [int(obj['bbox_xmin']), int(obj['bbox_ymin']), math.fabs(obj['bbox_xmax'] - obj['bbox_xmin']), math.fabs(obj['bbox_ymax'] - obj['bbox_ymin'])]
                        ann['segmentation'] = [[int(obj['bbox_xmin']), int(obj['bbox_ymin']), 
                                        int(obj['bbox_xmin']), int(obj['bbox_ymax']), 
                                        int(obj['bbox_xmax']), int(obj['bbox_ymax']), 
                                        int(obj['bbox_xmax']), int(obj['bbox_ymin'])]]
                        ann['iscrowd'] = 0
                        annotations.append(ann)

                if len(dataset_dicts) % 500 == 0:
                    print("Processed %s images" % (len(dataset_dicts)))

                if not filename[1:] in rings:
                    rings[filename[1:]] = {}
                rings[filename[1:]][filename[0]] = len(annotations)
                record['annotations'] = annotations
                dataset_dicts.append(record)

        check_rings(rings)

        outfile_name = os.path.join(out_dir, json_name % (sub_set + '_nuscyc'))
        print("Processed %s images" % (len(dataset_dicts)))

        with open(outfile_name, 'w') as outfile:
            outfile.write(json.dumps(dataset_dicts))

def anns_in_ring(ann_dir, filename):
  anns = 0
  for i in range(0,6):
    complete_name_ann = os.path.join(ann_dir, str(i) + filename + '.txt')
    if os.path.exists(complete_name_ann): 
      anns += 1
  return anns > 0

def check_rings(ann_dict):
  for ring in ann_dict:
    print('ring', len(ann_dict[ring]), ring)
    assert len(ann_dict[ring]) == 6

    sum_ring = 0
    for cam in ann_dict[ring]:
        sum_ring += ann_dict[ring][cam]

    assert sum_ring > 0
    print(sum_ring)

def load_nuscenes_instance(data_dir, split):
    img_id = 0
    ann_id = 0
    json_name = 'annotations_nuscyc_%s_token.json'

    ann_dir = os.path.join(data_dir, 'label')
    im_dir = os.path.join(data_dir, 'image')  #images_dataset
    print('Starting %s' % ann_dir)
    print('Starting %s\n' % im_dir)

    dataset_dicts = []
    rings = {}

    split_file = os.path.join(data_dir, '%s.txt' %split)
    with open(split_file, "r") as f:
        im_list = f.read().splitlines()
        for filename in im_list:
            if not anns_in_ring(ann_dir, filename[1:]):
              # print('Empty ring, continuing.')
              continue
            complete_name_im = os.path.join(im_dir, filename + '.jpg')
            complete_name_ann = os.path.join(ann_dir, filename + '.txt')
            
            if not os.path.exists(complete_name_im): 
                continue

            record = {}
            record['image_id'] = img_id
            img_id += 1

            im = Image.open(complete_name_im)
            record['width'] = int(im.size[0])
            record['height'] = int(im.size[1])
            record['file_name'] = complete_name_im

            if os.path.exists(complete_name_ann): 
              pre_objs = np.genfromtxt(complete_name_ann, delimiter=' ',
                  names=['token', 'type', 'truncated', 'occluded', 'alpha', 'bbox_xmin', 'bbox_ymin',
                  'bbox_xmax', 'bbox_ymax', 'dimensions_1', 'dimensions_2', 'dimensions_3',
                  'location_1', 'location_2', 'location_3', 'rotation_y'], dtype=None, encoding='ascii')

              if (pre_objs.ndim < 1):
                  pre_objs = np.array(pre_objs, ndmin=1)
            else:
              pre_objs = []
            annotations = []

            for obj in pre_objs:
                if (category_dict[obj['type']] != -1):
                    ann = {}
                    ann_id += 1
                    ann['token'] = obj['token']
                    ann['category_id'] = int(category_dict[obj['type']])
                    ann['bbox'] = [int(obj['bbox_xmin']), int(obj['bbox_ymin']), math.fabs(obj['bbox_xmax'] - obj['bbox_xmin']), math.fabs(obj['bbox_ymax'] - obj['bbox_ymin'])]
                    ann['segmentation'] = [[int(obj['bbox_xmin']), int(obj['bbox_ymin']), 
                                    int(obj['bbox_xmin']), int(obj['bbox_ymax']), 
                                    int(obj['bbox_xmax']), int(obj['bbox_ymax']), 
                                    int(obj['bbox_xmax']), int(obj['bbox_ymin'])]]
                    ann['iscrowd'] = 0
                    ann["bbox_mode"] = BoxMode.XYWH_ABS
                    annotations.append(ann)

            if len(dataset_dicts) % 500 == 0:
                print("Processed %s images" % (len(dataset_dicts)))

            if not filename[1:] in rings:
                rings[filename[1:]] = {}
            rings[filename[1:]][filename[0]] = len(annotations)
            record['annotations'] = annotations
            dataset_dicts.append(record)

    check_rings(rings)
    print("Processed %s images" % (len(dataset_dicts)))

    outfile_name = os.path.join(data_dir, json_name % (split))
    with open(outfile_name, 'w') as outfile:
        outfile.write(json.dumps(dataset_dicts))

    return dataset_dicts

def upload_nuscenes():
    files_list = [
    '/raid/datasets/token_nuscenes/train_ring.txt',
    '/raid/datasets/token_nuscenes/val_ring.txt', 
    '/raid/datasets/token_nuscenes/trainval_ring.txt', 
    '/raid/datasets/token_nuscenes/minitrain_ring.txt', 
    '/raid/datasets/token_nuscenes/minival_ring.txt', 
    ]

    splits = ['train', 'val', 'trainval', 'minitrain', 'minival']
    load_nuscenes_instances('/raid/datasets/token_nuscenes', out_dir, splits, files_list)

def register_nuscenes(name, dirname, split):
    json_name = 'annotations_nuscyc_%s_token.json'
    outfile_name = os.path.join(dirname, json_name % split)

    if not os.path.exists(outfile_name):
        load_and_register_nuscenes(name, dirname, split)
    else:
        with open(outfile_name) as json_file:
            data = json.load(json_file)

            for record in data:
                for ann in record['annotations']:
                    ann["bbox_mode"] = BoxMode.XYWH_ABS

            DatasetCatalog.register(name, lambda: data)
            MetadataCatalog.get(name).set(
            thing_classes=CLASS_NAMES, dirname=dirname, split=split
            )

def load_and_register_nuscenes(name, dirname, split):
    DatasetCatalog.register(name, lambda: load_nuscenes_instance(dirname, split))
    MetadataCatalog.get(name).set(
    thing_classes=CLASS_NAMES, dirname=dirname, split=split
    )
