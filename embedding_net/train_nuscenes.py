# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
Detection Training Script.

This scripts reads a given config file and runs the training or evaluation.
It is an entry point that is made to train standard models in detectron2.

In order to let one script support training of many models,
this script contains logic that are specific to these built-in models and therefore
may not be suitable for your own project.
For example, your research project perhaps only needs a single "evaluator".

Therefore, we recommend you to use detectron2 as an library and take
this file as an example of how to use the library.
You may want to write your own script with your datasets and other customizations.
"""


# from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch

import logging
import os
import json
from collections import OrderedDict
import torch
from torch.nn.parallel import DistributedDataParallel

from detectron2.structures import BoxMode, PolygonMasks, Boxes
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer, PeriodicCheckpointer
from detectron2.config import get_cfg
from detectron2.data import (
    DatasetCatalog,
    MetadataCatalog,
    build_detection_test_loader,
    build_detection_train_loader,
)
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.evaluation import (
    CityscapesEvaluator,
    COCOEvaluator,
    COCOPanopticEvaluator,
    DatasetEvaluators,
    LVISEvaluator,
    PascalVOCDetectionEvaluator,
    SemSegEvaluator,
    inference_on_dataset,
    print_csv_format,
)
from detectron2.modeling import build_model
from detectron2.solver import build_lr_scheduler, build_optimizer
from detectron2.utils.events import (
    CommonMetricPrinter,
    EventStorage,
    JSONWriter,
    TensorboardXWriter,
)

nuscyc=True
carpedcyc=False
nuscenes=False

if carpedcyc:
    category_dict = {'Pedestrian' : 0,
              'Car' :  1,
              'Cyclist' : 2,
              }
if nuscyc:
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
if nuscenes:
    category_dict = {'Pedestrian' : 0,
              'Car' :  1,
              'Motorcycle' :  2,
              'Motorcyclist' : 3,
              'Bicycle' :  4,
              'Bicyclist' : 5,
              'Bus' :  6,
              'Truck' : 7, 
              'Construction' :  8,
              # 'Van' :  9,
              'Trailer' : 9,
              'Barrier' :  10,
              'Cone' : 11
              }

logger = logging.getLogger("detectron2")

def do_test(cfg, model):
    results = OrderedDict()
    for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results


def do_train(cfg, model, resume=False):
    model.train()
    optimizer = build_optimizer(cfg, model)
    scheduler = build_lr_scheduler(cfg, optimizer)


    checkpointer = DetectionCheckpointer(
        model, cfg.OUTPUT_DIR, optimizer=optimizer, scheduler=scheduler
    )
    start_iter = (
        checkpointer.resume_or_load('', resume=resume).get("iteration", -1) + 1
    )
    checkpoint = torch.load(cfg.MODEL.WEIGHTS)
    # print(checkpoint['model'].keys())
    model.load_state_dict(checkpoint['model'])

    max_iter = cfg.SOLVER.MAX_ITER

    periodic_checkpointer = PeriodicCheckpointer(
        checkpointer, cfg.SOLVER.CHECKPOINT_PERIOD, max_iter=max_iter
    )

    writers = (
        [
            CommonMetricPrinter(max_iter),
            JSONWriter(os.path.join(cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(cfg.OUTPUT_DIR),
        ]
        if comm.is_main_process()
        else []
    )

    # compared to "train_net.py", we do not support accurate timing and
    # precise BN here, because they are not trivial to implement
    data_loader = build_detection_train_loader(cfg)
    logger.info("Starting training from iteration {}".format(start_iter))
    with EventStorage(start_iter) as storage:
        for data, iteration in zip(data_loader, range(start_iter, max_iter)):
            iteration = iteration + 1
            storage.step()

            # print(data[0].get('file_name'))
            # print(data[1].get('file_name'))
            # print(data[2].get('file_name'))
            # print(data[3].get('file_name'))
            # print(data[4].get('file_name'))
            # print(data[5].get('file_name'))

            loss_dict = model(data)
            losses = sum(loss for loss in loss_dict.values())
            assert torch.isfinite(losses).all(), loss_dict

            loss_dict_reduced = {k: v.item() for k, v in comm.reduce_dict(loss_dict).items()}
            losses_reduced = sum(loss for loss in loss_dict_reduced.values())
            if comm.is_main_process():
                storage.put_scalars(total_loss=losses_reduced, **loss_dict_reduced)

            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            storage.put_scalar("lr", optimizer.param_groups[0]["lr"], smoothing_hint=False)
            scheduler.step()

            if (
                cfg.TEST.EVAL_PERIOD > 0
                and iteration % cfg.TEST.EVAL_PERIOD == 0
                and iteration != max_iter
            ):
                do_test(cfg, model)
                # Compared to "train_net.py", the test results are not dumped to EventStorage
                comm.synchronize()

            if iteration - start_iter > 5 and (iteration % 20 == 0 or iteration == max_iter):
                for writer in writers:
                    writer.write()
            periodic_checkpointer.step(iteration)


def do_test(cfg, model):
    results = OrderedDict()
    for idx, dataset_name in enumerate(cfg.DATASETS.TEST):
        data_loader = build_detection_test_loader(cfg, dataset_name)
        evaluator = get_evaluator(
            cfg, dataset_name, os.path.join(cfg.OUTPUT_DIR, "inference", dataset_name)
        )
        results_i = inference_on_dataset(model, data_loader, evaluator)
        results[dataset_name] = results_i
        if comm.is_main_process():
            logger.info("Evaluation results for {} in csv format:".format(dataset_name))
            print_csv_format(results_i)
    if len(results) == 1:
        results = list(results.values())[0]
    return results

def register_nuscenes(out_dir, names, dataset='nuscenes'):
    for name in names:
        if dataset == 'carpedcyc':
            json_name = 'annotations_nuscenes_%s_carpedcyc.json'
        if dataset == 'nuscyc':
            json_name = 'annotations_nuscyc_%s_token.json'
        if dataset == 'nuscenes':
            json_name = 'annotations_nuscenes_%s.json'
        
        outfile_name = os.path.join(out_dir, json_name % name)
        with open(outfile_name) as json_file:
            data = json.load(json_file)

            for record in data:
                for ann in record['annotations']:
                    ann["bbox_mode"] = BoxMode.XYWH_ABS

            if dataset == 'carpedcyc':
                DatasetCatalog.register("ns_carpedcyc_" + name, lambda: data)
                MetadataCatalog.get("ns_carpedcyc_" + name).set(thing_classes=list(category_dict.keys()))
            if dataset == 'nuscyc':
                DatasetCatalog.register("nuscyc_" + name, lambda: data)
                MetadataCatalog.get("nuscyc_" + name).set(thing_classes=list(category_dict.keys()))
            if dataset == 'nuscenes':
                DatasetCatalog.register("nuscenes_" + name, lambda: data)
                MetadataCatalog.get("nuscenes_" + name).set(thing_classes=list(category_dict.keys()))


def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg


def main(args):
    # register_nuscenes('/home/irecorte/sianms/detectron2_sianms/nuscenes_train/json', ['val', 'train'], dataset='nuscyc')
    # register_nuscenes('/home/irecorte/sianms/detectron2_sianms/nuscenes_train/json', ['val', 'train'], dataset='nuscenes')
    # register_nuscenes('/home/irecorte/sianms/detectron2_sianms/nuscenes_train/json', ['val', 'train'], dataset='carpedcyc')
    register_nuscenes('/home/irecorte/sianms/detectron2_sianms/nuscenes_train/json', ['val', 'train'], dataset='nuscyc')
    cfg = setup(args)

    model = build_model(cfg)
    # logger.info("Model:\n{}".format(model))

    distributed = comm.get_world_size() > 1
    if distributed:
        model = DistributedDataParallel(
            model, device_ids=[comm.get_local_rank()], broadcast_buffers=False
        )

    do_train(cfg, model)
    return do_test(cfg, model)

if __name__ == "__main__":
    args = default_argument_parser().parse_args()
    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
