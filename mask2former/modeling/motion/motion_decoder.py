# Copyright (c) Facebook, Inc. and its affiliates.
# Modifications copyright (c) 2022 ZIP Group
from . import MOTIONDEC_REGISTRY
from detectron2.config import configurable
from typing import Dict, List, Tuple
from mmcv.cnn.bricks import ConvModule
import torch.nn as nn
from detectron2.layers import ShapeSpec
import torch
from ..transformer_decoder.position_encoding import PositionEmbeddingSine
from detectron2.layers import Conv2d
import fvcore.nn.weight_init as weight_init
from .decoder_layer import SelfAttentionLayer, MLP, FFNLayer



@MOTIONDEC_REGISTRY.register()
class MotionDec(nn.Module):

    @configurable
    def __init__(self, 
                    *, 
                    inplanes: Tuple[int], 
                    query_input_indices: List[int],
                    # flow_in_channels: List[int],
                    in_channels: List[int],
                    hidden_dim: int,
                    num_queries: int,
                    nheads: int,
                    dim_feedforward: int,
                    dec_layers: int,
                    pre_norm: bool,
                    enforce_input_project: bool, 
                    flow_scale_factor=2.5,
                    flow_interpolate_factor=0.5, 
                    input_shape: Dict[str, ShapeSpec],
                    ):
        super().__init__()
        self.in_channels = in_channels
        self.deconv_inchannels = [386, 386, 386, 256]
        self.deconv_outchannels = [64, 128, 128, 128]
        self.flowlayer_inchannels = [386, 386, 386, 256] # inplanes = [inplanes]
        self.query_input_indices = query_input_indices
        # self.flow_in_channels = flow_in_channels
        self.input_shape = sorted(input_shape.items(), key=lambda x: x[1].stride)
        self.out_channels = [v.channels for k, v in self.input_shape]
        self.hidden_dim = hidden_dim

        self.flow_interpolate_factor = flow_interpolate_factor
        self.flow_scale_factor = flow_scale_factor

        self.build_pixel_decoder()

        # positional encoding
        N_steps = hidden_dim // 2
        self.pe_layer = PositionEmbeddingSine(N_steps, normalize=True)
        #self.channel_mapper = ChannelMapper(in_channels=self.flow_in_channels, out_channels=N_steps)
        
        # define Transformer decoder here
        self.num_heads = nheads
        self.num_layers = dec_layers
        self.transformer_self_attention_layers = nn.ModuleList()
        # self.transformer_cross_attention_layers = nn.ModuleList()
        self.transformer_ffn_layers = nn.ModuleList()

        for _ in range(self.num_layers):
            self.transformer_self_attention_layers.append(
                SelfAttentionLayer(
                    d_model=hidden_dim,
                    nhead=nheads,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

            self.transformer_ffn_layers.append(
                FFNLayer(
                    d_model=hidden_dim,
                    dim_feedforward=dim_feedforward,
                    dropout=0.0,
                    normalize_before=pre_norm,
                )
            )

        self.decoder_norm = nn.LayerNorm(hidden_dim)
        self.output_head = MLP(hidden_dim, hidden_dim*2, hidden_dim, 3)
        self.scale_layer = MLP(128, hidden_dim*2, hidden_dim*2, 3)
        self.hidden_dim = hidden_dim

        self.num_queries = num_queries
        self.query_embed = nn.Embedding(num_queries, hidden_dim)

        self.fuse_layer = ConvModule(
            in_channels=4,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=True,
            conv_cfg=dict(type='Conv'),
            act_cfg=None)
        nn.init.constant_(self.fuse_layer.conv.weight, 0)
        nn.init.constant_(self.fuse_layer.conv.bias, 1)


        # level embedding (we always use 3 scales)
        self.num_feature_levels = 3
        self.level_embed = nn.Embedding(self.num_feature_levels, hidden_dim)
        self.input_proj = nn.ModuleList()
        for i in range(self.num_feature_levels-1, -1, -1):
            if self.in_channels[i] != hidden_dim or enforce_input_project:
                self.input_proj.append(Conv2d(self.in_channels[i], hidden_dim, kernel_size=1))
                weight_init.c2_xavier_fill(self.input_proj[-1])
            else:
                self.input_proj.append(nn.Sequential())

    @classmethod
    def from_config(cls, cfg, input_shape):
        ret = {}
        ret["inplanes"] = cfg.MODEL.MOTION.DECODER_INPLANES
        ret['query_input_indices'] = [cfg.MODEL.MOTION.ENCODER_OUTINDICES.index(i) for i in cfg.MODEL.MOTION.DECODER_QUERYINPUTINDECES]
        ret["hidden_dim"] = cfg.MODEL.MOTION.HIDDEN_DIM
        ret["num_queries"] = cfg.MODEL.MOTION.NUM_FLOW_QUERIES
        ret["nheads"] = cfg.MODEL.MOTION.NHEADS
        ret["dim_feedforward"] = cfg.MODEL.MOTION.DIM_FEEDFORWARD
        ret["dec_layers"] = cfg.MODEL.DECODER_LAYERS
        ret["pre_norm"] = cfg.MODEL.MOTION.PRE_NORM
        ret["enforce_input_project"] = cfg.MODEL.MOTION.ENFORCE_INPUT_PROJ
        ret["in_channels"] = cfg.MODEL.MOTION.IN_CHANNELS
        ret["flow_scale_factor"] =  cfg.MODEL.MOTION.FLOW_SCALE_FACTOR
        ret["flow_interpolate_factor"] = cfg.MODEL.MOTION.FLOW_INTERPOLATE_FACTOR
        ret["input_shape"] = input_shape
        return ret


    def build_pixel_decoder(self):
        self.deconv_layers = []
        self.flow_layers = []
        self.upflow_layers = []
        self.scale_layers = []
        # planes = self.inplanes[-1] // 2
        for i in range(len(self.flowlayer_inchannels) - 1, -1, -1):
            deconv_layer = ConvModule(
                in_channels=self.deconv_inchannels[i],
                out_channels=self.deconv_outchannels[i],
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
                conv_cfg=dict(type='deconv'),
                act_cfg=dict(type='LeakyReLU', negative_slope=0.1))
            self.add_module(f'deconv{i+2}', deconv_layer)
            self.deconv_layers.insert(0, f'deconv{i+2}')

            flow_layer = ConvModule(
                in_channels=self.flowlayer_inchannels[i],
                out_channels=2,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
                conv_cfg=dict(type='Conv'),
                act_cfg=None)
            self.add_module(f'predict_flow{i+3}', flow_layer)
            self.flow_layers.insert(0, f'predict_flow{i+3}')

            upflow_layer = ConvModule(
                in_channels=2,
                out_channels=2,
                kernel_size=4,
                stride=2,
                padding=1,
                bias=False,
                conv_cfg=dict(type='deconv'),
                act_cfg=None)
            self.add_module(f'upsample_flow{i+2}', upflow_layer)
            self.upflow_layers.insert(0, f'upsample_flow{i+2}')
            # planes = planes // 2

        self.predict_flow = ConvModule(
            in_channels=194,
            out_channels=2,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False,
            conv_cfg=dict(type='Conv'),
            act_cfg=None)

    def crop_like(self, input, target):
        """Crop `input` as the size of `target`."""
        if input.size()[2:] == target.size()[2:]:
            return input
        else:
            return input[:, :, :target.size(2), :target.size(3)]

    def forward(self, x, query_init): 

        conv_outs = x
        query_x = [x[i] for i in self.query_input_indices]

        assert len(query_x) == self.num_feature_levels
        src = []
        pos = []
        size_list = []

        motion_map = conv_outs[0].clone()
        motion_map = self.scale_layer(motion_map.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        query_x.reverse()
        for i in range(self.num_feature_levels):
            size_list.append(query_x[i].shape[-2:])
            pos.append(self.pe_layer(query_x[i], None).flatten(2))
            src.append(self.input_proj[i](query_x[i]).flatten(2) + self.level_embed.weight[i][None, :, None])

            # flatten NxCxHxW to HWxNxC
            pos[-1] = pos[-1].permute(2, 0, 1)
            src[-1] = src[-1].permute(2, 0, 1)

        _, bs, _ = src[0].shape

        # QxNxC
        query_embed = self.query_embed.weight.unsqueeze(1).repeat(1, bs, 1)
        query_feat = query_init

        for i in range(self.num_layers):
            level_index = i % self.num_feature_levels
            output = torch.cat([query_feat + query_embed, src[level_index] + pos[level_index]], dim=0) # (N+HW, B, C)

            output = self.transformer_self_attention_layers[i](
                output, tgt_mask=None,
                tgt_key_padding_mask=None,
                query_pos=None
            )
            
            # FFN
            output = self.transformer_ffn_layers[i](
                output
            )
            # update query feature and the src for the current level
            query_feat, src[level_index] = torch.split(output, [self.num_queries, size_list[level_index][0]*size_list[level_index][1]], dim=0)
        
        refined_out = [src[i].permute(1, 2, 0).reshape(bs, self.hidden_dim, size_list[i][0], size_list[i][1]) for i in range(self.num_feature_levels)]
        refined_out.reverse()
        conv_outs = conv_outs[:-len(refined_out)] + refined_out
        
        ## pixel-wise flow
        num_outs = len(conv_outs)
        for i, deconv_name, flow_name, upflow_name in zip(
                range(1, num_outs)[::-1], self.deconv_layers[::-1],
                self.flow_layers[::-1], self.upflow_layers[::-1]):
            deconv_layer = getattr(self, deconv_name)
            flow_layer = getattr(self, flow_name)
            upflow_layer = getattr(self, upflow_name)

            if i == num_outs - 1:
                concat_out = conv_outs[i]
            flow = flow_layer(concat_out)
            upflow = self.crop_like(upflow_layer(flow), conv_outs[i - 1])
            deconv_out = self.crop_like(
                deconv_layer(concat_out), conv_outs[i - 1])
            concat_out = torch.cat((conv_outs[i - 1], deconv_out, upflow),
                                    dim=1)
        
        concat_out = torch.nn.functional.interpolate(
            concat_out,
            scale_factor=self.flow_interpolate_factor,
            mode='bilinear',
            align_corners=False) 

        flow = self.predict_flow(concat_out) 
        flow *= self.flow_scale_factor 

        query_feat = self.decoder_norm(query_feat)
        query_feat = self.output_head(query_feat) 
        bs, _, h, w = motion_map.shape
        mx = torch.einsum("bqc,bchw->bqhw", query_feat.transpose(0, 1), motion_map[:, self.hidden_dim:, ...]).reshape(bs*self.num_queries, 1, h, w)
        my = torch.einsum("bqc,bchw->bqhw", query_feat.transpose(0, 1), motion_map[:, :self.hidden_dim, ...]).reshape(bs*self.num_queries, 1, h, w)
        outputs_motion = torch.cat([mx, my], dim=1)

        flow = flow.unsqueeze(1).repeat(1, self.num_queries, 1, 1, 1).reshape(bs*self.num_queries, 2, h, w)  # bs, 2, H, W
        final_flow = self.fuse_layer(torch.cat([flow, outputs_motion], dim=1))

        return final_flow 