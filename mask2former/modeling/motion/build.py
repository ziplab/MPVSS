# Copyright (c) Facebook, Inc. and its affiliates.
from typing import Dict,  Tuple

import torch
from torch import nn

from detectron2.layers import ShapeSpec
from detectron2.config import configurable
from detectron2.utils.registry import Registry
from detectron2.structures import ImageList

__all__ = [
    "build_motion",
    "MOTION_REGISTRY",
    "MOTIONENC_REGISTRY",
    "MOTIONDEC_REGISTRY",
    "BaseFlowEstimator",
    "FlowEstimator"
]

MOTION_REGISTRY = Registry("MOTION")
MOTION_REGISTRY.__doc__ = """
Registry for motion model, which make flow prediction for VSS
"""

def build_motion(cfg, input_shape: Dict[str, ShapeSpec]):
    """
    Build a motion model from `cfg.MODEL.MOTION.NAME`.

    Returns:
        an instance of :class:`Motion`
     """

    motion_name = cfg.MODEL.MOTION.NAME
    motion = MOTION_REGISTRY.get(motion_name)(cfg, input_shape)
    return motion

def build_teacher_motion(cfg, input_shape: Dict[str, ShapeSpec]):
    """
    Build a motion model from `cfg.MODEL.MOTION.NAME`.

    Returns:
        an instance of :class:`Motion`
     """

    motion_name = cfg.MODEL.TEACHER_MOTION.NAME
    motion = MOTION_REGISTRY.get(motion_name)(cfg, input_shape)
    return motion



MOTIONENC_REGISTRY = Registry("MOTIONENC")
MOTIONENC_REGISTRY.__doc__ = """
Registry for motion feature encoder, which encode cost map for flow prediction
"""
MOTIONDEC_REGISTRY = Registry("MOTIONDEC")
MOTIONDEC_REGISTRY.__doc__ = """
Registry for motion decoder, which make flow prediction
"""

def build_motion_encoder(cfg):
    """
    Build a motion model from `cfg.MODEL.MOTION.NAME`.

    Returns:
        an instance of :class:`Motion`
     """

    enc_name = cfg.MODEL.MOTION.ENCODER_NAME
    enc = MOTIONENC_REGISTRY.get(enc_name)(cfg)
    return enc



def build_motion_decoder(cfg, input_shape: Dict[str, ShapeSpec]):
    """
    Build a motion model from `cfg.MODEL.MOTION.NAME`.

    Returns:
        an instance of :class:`Motion`
     """

    dec_name = cfg.MODEL.MOTION.DECODER_NAME
    dec = MOTIONDEC_REGISTRY.get(dec_name)(cfg, input_shape)
    return dec


@MOTION_REGISTRY.register()
class BaseFlowEstimator(nn.Module):
    
    @configurable
    def __init__(self, 
                 *,
                 encoder: nn.Module,
                 decoder: nn.Module,
                 size_divisibility: int, 
                 img_scale_factor: float, 
                 input_shape: Dict[str, ShapeSpec],
                 flow_img_norm_std: Tuple[float],
                 flow_img_norm_mean: Tuple[float],
                 flow_scale_factor: float, 
                 flow_interpolate_factor: float,
                 ):
        super().__init__()
        self.size_divisibility = size_divisibility
        self.img_scale_factor = img_scale_factor
        self.input_shape = input_shape
        self.flow_scale_factor  = flow_scale_factor
        self.flow_interpolate_factor = flow_interpolate_factor

        self.register_buffer("flow_img_norm_mean", torch.Tensor(flow_img_norm_mean).repeat(2)[:, None, None], False)
        self.register_buffer("flow_img_norm_std", torch.Tensor(flow_img_norm_std).repeat(2)[:, None, None], False)
        
        self.encoder = encoder
        self.decoder = decoder


    @classmethod
    def from_config(cls, cfg, input_shape: Dict[str, ShapeSpec]):
        encoder = build_motion_encoder(cfg)
        decoder = build_motion_decoder(cfg, input_shape)
        return {
            "encoder": encoder,
            "decoder": decoder,
            "input_shape": input_shape, 
            "size_divisibility": cfg.MODEL.MASK_FORMER.SIZE_DIVISIBILITY,
            "img_scale_factor": cfg.MODEL.MOTION.IMG_SCALE_FACTOR,
            "flow_img_norm_std": cfg.MODEL.MOTION.FLOW_IMG_NORM_STD,
            "flow_img_norm_mean": cfg.MODEL.MOTION.FLOW_IMG_NORM_MEAN,
            "flow_scale_factor": cfg.MODEL.MOTION.FLOW_SCALE_FACTOR,
            "flow_interpolate_factor": cfg.MODEL.MOTION.FLOW_INTERPOLATE_FACTOR,
        }

    @property
    def device(self):
        return self.flow_img_norm_mean.device

    def prepare_imgs(self, batched_inputs):
        """Preprocess images pairs for computing flow.

        Args:
            imgs (Tensor): of shape (N, 6, H, W) encoding input images pairs.
                Typically these should be mean centered and std scaled.
            img_metas (list[dict]): list of image information dict where each
                dict has: 'img_shape', 'scale_factor', 'flip', and may also
                contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

        Returns:
            Tensor: of shape (N, 6, H, W) encoding the input images pairs for
            FlowNetSimple.
        """
        flow_img = [torch.cat([x["image"].to(self.device), x["ref_image"].to(self.device)], dim=0) for x in batched_inputs]

        # flow_img = imgs * self.img_norm_std + self.img_norm_mean
        flow_img = [(x / self.flow_img_norm_std - self.flow_img_norm_mean) for x in flow_img]
        flow_img = ImageList.from_tensors(flow_img, int(self.size_divisibility))
        
        flow_img = flow_img.tensor
        flow_img[:, :, batched_inputs[0]["height"]:, :] = 0.0
        flow_img[:, :, :, batched_inputs[0]["width"]:] = 0.0

        flow_img = torch.nn.functional.interpolate(
            flow_img,
            scale_factor=self.img_scale_factor,
            mode='bilinear',
            align_corners=False)
        return flow_img


    def forward(self, x):
        """
        Input: the concat image pair
        Output: the predicted flow
        """
        flow_imgs = self.prepare_imgs(x)
        out = self.encoder(flow_imgs)
        out = self.decoder(out)
        return out


@MOTION_REGISTRY.register()
class FlowEstimator(BaseFlowEstimator):
    def forward(self, x, query_init):
        flow_imgs = self.prepare_imgs(x)
        out = self.encoder(flow_imgs)
        out = self.decoder(out, query_init)
        return out