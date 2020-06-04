# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import logging
import numpy as np
import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F

from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.utils.events import get_event_storage
from detectron2.utils.registry import Registry
from termcolor import colored
from detectron2.data.datasets.nuscenes import is_vehicle

EMBEDDING_HEAD_REGISTRY = Registry("EMBEDDING_HEAD")
EMBEDDING_HEAD_REGISTRY.__doc__ = """
Registry for box heads, which make box predictions from per-region features.

The registered object will be called with `obj(cfg, input_shape)`.
"""

logger = logging.getLogger(__name__)

@EMBEDDING_HEAD_REGISTRY.register()
class DoubleMarginContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin_p, margin_n):
        super(DoubleMarginContrastiveLoss, self).__init__()
        self.margin_p = margin_p #1
        self.margin_n = margin_n #3
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * F.relu((distances + self.eps).sqrt() - self.margin_p).pow(2) +
                        (1 - target.float()) * F.relu(self.margin_n - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()

@EMBEDDING_HEAD_REGISTRY.register()
class DoubleMarginContrastiveLossOHEM(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    """

    def __init__(self, margin_p, margin_n):
        super(DoubleMarginContrastiveLossOHEM, self).__init__()
        self.margin_p = margin_p #1
        self.margin_n = margin_n #3
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        num_pos = torch.sum(target).cpu().numpy()
        print(num_pos)
        losses_p = 0.5 * (F.relu((distances[target.bool()] + self.eps).sqrt() - self.margin_p).pow(2))

        print('losses_p', losses_p.size())
        losses_n = 0.5 * (F.relu(self.margin_n - (distances[~target.bool()] + self.eps).sqrt()).pow(2))
        print('losses_n', losses_n.size())

        num_topk = int(np.min((np.max((1, num_pos)), losses_n.size()[0])))
        
        if losses_n.size()[0] > 0:
            losses_n = torch.topk(losses_n, num_topk)[0]

        print('losses_n', losses_n.size())

        losses = torch.cat((losses_p, losses_n))
        return losses.mean() if size_average else losses.sum()

@EMBEDDING_HEAD_REGISTRY.register()
class EmbeddingHead(nn.Module):
    """
    A head with several 3x3 conv layers (each followed by norm & relu) and
    several fc layers (each followed by relu).
    """

    def __init__(self, cfg, input_shape: ShapeSpec):
        super().__init__()

        n_output = cfg.MODEL.EMBEDDING_HEAD.NUM_OUTPUT

        self._output_size = (input_shape.channels, input_shape.height, input_shape.width)
        self.convnet = nn.Sequential(nn.Conv2d(1, 32, kernel_size=5), 
                                     nn.PReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2),
                                     nn.Conv2d(32, 64, kernel_size=5), 
                                     nn.PReLU(),
                                     nn.MaxPool2d(kernel_size=2, stride=2))

        self.fc = nn.Sequential(nn.Linear(64 * 61 * 9, 1024),
                                nn.PReLU(),
                                nn.Linear(1024, 256),
                                nn.PReLU(),
                                nn.Linear(256, 256),
                                nn.PReLU(),
                                nn.Linear(256, n_output)
                                )
        self._output_size = n_output

        self.loss_fn = cfg.MODEL.EMBEDDING_HEAD.LOSS_FN


    def forward(self, x):
        output = x.view(x.size()[0],1,256,-1)
        # logger.info('x: ')
        # logger.info(output.size())
        output = self.convnet(output)
        # logger.info('convnet: ')
        # logger.info(output.size())
        # logger.info(output)
        output = output.view(output.size()[0], -1)
        # logger.info('view: ')
        # logger.info(output.size())
        output = self.fc(output)
        # logger.info('fc: ')
        # logger.info(output.size())
        return output

    def get_embedding(self, x):
        return self.forward(x)

    def embedding_loss(self, predictions, instances, image_names):
        margin_p = 1.
        margin_n = 3.
        # loss_fn = DoubleMarginContrastiveLoss(margin_p, margin_n)
        loss_fn =  EMBEDDING_HEAD_REGISTRY.get(self.loss_fn)(margin_p, margin_n)
        #anchors, positivos + negativos, targets

        tokens_dict = []
        for idx, ins in enumerate(instances):
            tokens_dict.append({})
            if ins.has('gt_tokens'):
                for jdx, (token, ins_class) in enumerate(zip(ins.gt_tokens, ins.gt_classes)):
                    if not is_vehicle(ins_class.cpu().numpy()):
                        # print('Not a vehicle')
                        continue
                    if token.get_str() in tokens_dict[idx]:
                        if ins.objectness_logits[jdx] > tokens_dict[idx][token.get_str()][1]:
                            tokens_dict[idx][token.get_str()] = (jdx, ins.objectness_logits[jdx])
                    else:
                        tokens_dict[idx][token.get_str()] = (jdx, ins.objectness_logits[jdx])

        camera_order = [0,2,5,3,4,1]
        anchors = torch.empty((0,100)).cuda()
        pos_neg = torch.empty((0,100)).cuda()
        targets = torch.empty((0)).cuda()
        one = torch.tensor([1.]).cuda()
        zero = torch.tensor([0.]).cuda()

        num_pos = 0
        num_neg = 0
        for i, idx in enumerate(camera_order):
            other_idx = camera_order[(i+1)%len(instances)]
            ins = instances[idx]
            other_ins = instances[other_idx]
            for token in tokens_dict[idx]:
                for other_token in tokens_dict[other_idx]:
                    feat_idx = tokens_dict[idx][token][0]
                    other_feat_idx = tokens_dict[other_idx][other_token][0]
                    anchors = torch.cat((anchors, torch.unsqueeze(predictions[idx][feat_idx], 0)))
                    pos_neg = torch.cat((pos_neg, torch.unsqueeze(predictions[other_idx][other_feat_idx], 0)))
                    if token == other_token:
                        targets = torch.cat((targets, one))
                        num_pos += 1
                        # print('    ', colored(other_token, 'green'))
                    else:
                        targets = torch.cat((targets, zero))
                        num_neg += 1
                        # print('    ', other_token)

        if len(anchors) == 0:
            return torch.tensor([0.0]).cuda()

        storage = get_event_storage()
        storage.put_scalar("embedding_head/num_positive_pairs", num_pos)
        storage.put_scalar("embedding_head/num_negative_pairs", num_neg)

        # embd_loss = loss_fn(anchors, pos_neg, targets)
        embd_loss =  loss_fn(anchors, pos_neg, targets)
        
        # for idx in range(len(image_names)):
        #     im = cv2.imread(image_names[idx])
        #     ins = instances[idx]

        #     im = cv2.resize(im, (1333, 750))
        #     cv2.rectangle(im, (0, 0), (1333, 750), (255, 255, 255), 2)

        #     for token in tokens_dict[idx]:
        #         box_cv = ins.proposal_boxes[tokens_dict[idx][token][0]].get_numpy()[0]
        #         cv2.rectangle(im, (int(box_cv[0]), int(box_cv[1])), (int(box_cv[2]), int(box_cv[3])), (0, 0, 255), 4)

        #     for box in ins.gt_boxes:
        #         box_cv = box.cpu().numpy()
        #         cv2.rectangle(im, (int(box_cv[0]), int(box_cv[1])), (int(box_cv[2]), int(box_cv[3])), (0, 255, 0), 4)

            # ims.append(im)
            # cv2.imshow('gt_boxes', im)
        
        # h1 = cv2.hconcat((ims[1], ims[0], ims[2]))
        # h2 = cv2.hconcat((ims[5], ims[3], ims[4]))
        # v1 = cv2.vconcat((h1, h2))

        # cv2.namedWindow('6im', cv2.WINDOW_NORMAL)
        # cv2.imshow('6im', v1)
        # cv2.waitKey(0)

        return embd_loss


    @property
    def output_size(self):
        return self._output_size

def build_embedding_head(cfg, input_shape):
    """
    Build a box head defined by `cfg.MODEL.EMBEDDING_HEAD.NAME`.
    """
    name = cfg.MODEL.EMBEDDING_HEAD.NAME
    return EMBEDDING_HEAD_REGISTRY.get(name)(cfg, input_shape)
