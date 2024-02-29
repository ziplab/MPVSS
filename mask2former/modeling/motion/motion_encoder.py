# Copyright (c) Facebook, Inc. and its affiliates.
# Modifications copyright (c) 2022 ZIP Group
import torch.nn as nn
from . import MOTIONENC_REGISTRY
from detectron2.config import configurable
from typing import Tuple
from mmcv.cnn.bricks import ConvModule
from mmcv.runner import BaseModule


@MOTIONENC_REGISTRY.register()
class MotionEnc(BaseModule):

     default_arch_setting = {
        'conv_layers': {
            'inplanes': (6, 64, 128, 256, 512, 512),
            'kernel_size': (7, 5, 5, 3, 3, 3),
            'num_convs': (1, 1, 2, 2, 2, 2)
        },
        'deconv_layers': {
            'inplanes': (386, 770, 1026, 1024)
        }
     }

     @configurable
     def __init__(self, 
                    *,
                    inplanes: Tuple[int],
                    kernel_sizes: Tuple[int],
                    num_convs: Tuple[int],
                    out_indices: Tuple[int]
                    ):
          super().__init__()
          
          self.inplanes = inplanes
          self.kernel_sizes = kernel_sizes
          self.num_convs = num_convs
          self.out_indices = out_indices
          
          self.conv_layers = []

          for i in range(len(inplanes)):
               num_convs = self.num_convs[i]
               kernel_size = self.kernel_sizes[i]
               inplanes = self.inplanes[i]
               if i == len(self.inplanes) - 1:
                    planes = 2 * inplanes
               else:
                    planes = self.inplanes[i + 1]

               conv_layer = nn.ModuleList()
               conv_layer.append(
                    ConvModule(
                         in_channels=inplanes,
                         out_channels=planes,
                         kernel_size=kernel_size,
                         stride=2,
                         padding=(kernel_size - 1) // 2,
                         bias=True,
                         conv_cfg=dict(type='Conv'),
                         act_cfg=dict(type='LeakyReLU', negative_slope=0.1)))
               for j in range(1, num_convs):
                    kernel_size = 3 if i == 2 else kernel_size
                    conv_layer.append(
                         ConvModule(
                         in_channels=planes,
                         out_channels=planes,
                         kernel_size=kernel_size,
                         stride=1,
                         padding=(kernel_size - 1) // 2,
                         bias=True,
                         conv_cfg=dict(type='Conv'),
                         act_cfg=dict(type='LeakyReLU', negative_slope=0.1)))

               self.add_module(f'conv{i+1}', conv_layer)
               self.conv_layers.append(f'conv{i+1}')

     @classmethod
     def from_config(cls, cfg):
          return {
               "inplanes": cfg.MODEL.MOTION.ENCODER_INPLANES,
               "kernel_sizes": cfg.MODEL.MOTION.ENCODER_KERNEL_SIZE,
               "num_convs": cfg.MODEL.MOTION.ENCODER_NUM_CONVS, 
               "out_indices": cfg.MODEL.MOTION.ENCODER_OUTINDICES
          }

     def forward(self, x):
          # x: prepared_imgs
          conv_outs = []
          for i, conv_name in enumerate(self.conv_layers, 1):
               conv_layer = getattr(self, conv_name)
               for module in conv_layer:
                    x = module(x)
               if i in self.out_indices:
                    conv_outs.append(x)

          return conv_outs