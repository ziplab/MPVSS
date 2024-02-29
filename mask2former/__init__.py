# Copyright (c) Facebook, Inc. and its affiliates.
from . import data  # register all new datasets
from . import modeling

# config
from .config import add_maskformer2_config
from .utils import DetectionCheckpointerMC, SingleGPUInferenceSampler, inference_on_vspw_dataset, CityscapesSemSegEvaluator

# dataset loading
from .data.dataset_mappers.mask_former_semantic_dataset_mapper import (
    MaskFormerSemanticDatasetMapper,
)

from .data.dataset_mappers.mask_former_video_semantic_dataset_mapper import (
    MaskFormerCSVideoSemanticDatasetTrainMapper, 
    MaskFormerCSVideoSemanticDatasetTestMapper,
    MaskFormerVSPWVideoSemanticDatasetTrainMapper, 
    MaskFormerVSPWVideoSemanticDatasetTestMapper,
    MaskFormerVSPWImageSemanticDatasetTestMapper
)
# evaluation
from .evaluation.instance_evaluation import InstanceSegEvaluator

# models
from .maskformer_model import MaskFormer
from .test_time_augmentation import SemanticSegmentorWithTTA
from .mpvss import MPVSS
