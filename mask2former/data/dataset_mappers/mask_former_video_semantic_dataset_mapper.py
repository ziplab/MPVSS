# Copyright (c) Facebook, Inc. and its affiliates.
# Modified by Bowen Cheng from https://github.com/facebookresearch/detr/blob/master/d2/detr/dataset_mapper.py
# Modifications copyright (c) 2022 ZIP Group
import copy
import logging
import random
import numpy as np
import torch
from torch.nn import functional as F
from typing import List, Optional, Union
from detectron2.config import configurable
from detectron2.data import MetadataCatalog
from detectron2.data import detection_utils as utils
from detectron2.data import transforms as T
from detectron2.projects.point_rend import ColorAugSSDTransform
from detectron2.structures import BitMasks, Instances
from fvcore.transforms.transform import (
    HFlipTransform,
    NoOpTransform,
    VFlipTransform,
)
from .augmentation import build_augmentation

__all__ = ["MaskFormerVSPWVideoSemanticDatasetTrainMapper", 
            "MaskFormerVSPWVideoSemanticDatasetTestMapper",
            "MaskFormerVSPWImageSemanticDatasetTestMapper"]


def check_image_size(dataset_dict, image):
    """
    Raise an error if the image does not match the size specified in the dict.
    """
    if "width" in dataset_dict or "height" in dataset_dict:
        image_wh = (image.shape[1], image.shape[0])
        expected_wh = (dataset_dict["width"], dataset_dict["height"])
        if not image_wh == expected_wh:
            dataset_dict.update({"width": image.shape[1]})
            dataset_dict.update({"height": image.shape[0]})
    # To ensure bbox always remap to original image size
    if "width" not in dataset_dict:
        dataset_dict["width"] = image.shape[1]
    if "height" not in dataset_dict:
        dataset_dict["height"] = image.shape[0]
    return dataset_dict


class MaskFormerVSPWVideoSemanticDatasetTrainMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by MaskFormer for semantic segmentation.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies geometric transforms to the image and annotation
    3. Find and applies suitable cropping to the image and annotation
    4. Prepare image and annotation to Tensors
    """

    @configurable
    def __init__(
        self,
        is_train=True,
        *,
        augmentations,
        image_format,
        ignore_label,
        size_divisibility,
        ref_frame_sampling: bool, 
        sampling_frame_num: int = 1,
        sampling_frame_range: int = 5,
    ):
        """
        NOTE: this interface is experimental.
        Args:
            is_train: for training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            ignore_label: the label that is ignored to evaluation
            size_divisibility: pad image size to be divisible by this value
        """
        self.is_train = is_train
        self.augmentations = augmentations # T.AugmentationList(augmentations)
        self.img_format = image_format
        self.ignore_label = ignore_label
        self.size_divisibility = size_divisibility
        self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        self.ref_frame_sampling = ref_frame_sampling

        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[{self.__class__.__name__}] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train=True):
        
        augs = build_augmentation(cfg, is_train)
        # Assume always applies to the training set.
        dataset_names = cfg.DATASETS.TRAIN
        meta = MetadataCatalog.get(dataset_names[0])
        ignore_label = meta.ignore_label
        ref_frame_sampling = cfg.INPUT.TRAIN_REF_FRAME_SAMPLING

        sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.TRAIN_SAMPLING_FRAME_RANGE

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "ignore_label": ignore_label,
            "size_divisibility": cfg.INPUT.SIZE_DIVISIBILITY,
            "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
            "ref_frame_sampling": ref_frame_sampling
        }
        return ret



    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one video, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        assert self.is_train, "MaskFormerVSPWVideoSemanticDatasetTrainMapper should only be used for training!"

        # read image from file
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below

        video_length = dataset_dict['length']
        if self.ref_frame_sampling:
            cur_frame = random.randrange(video_length)
            start_idx = max(0, cur_frame - self.sampling_frame_range)
            end_idx = min(video_length, cur_frame + self.sampling_frame_range+1)

            add_frame = np.random.choice(
                np.array(list(range(start_idx, cur_frame)) + list(range(cur_frame+1, end_idx))),
                self.sampling_frame_num,)
            sampled_frame = sorted([cur_frame, int(add_frame)])
            image = utils.read_image(dataset_dict['img_list'][sampled_frame[1]], format=self.img_format)
            ref_img = utils.read_image(dataset_dict['img_list'][sampled_frame[0]], format=self.img_format)
            dataset_dict = check_image_size(dataset_dict, image)

            # read segmentation from file 
            if "ann_list" in dataset_dict:
                # PyTorch transformation not implemented for uint16, so converting it to double first
                sem_seg_gt = utils.read_image(dataset_dict['ann_list'][sampled_frame[1]]).astype("double")
                #sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
            else:
                sem_seg_gt = None
            if sem_seg_gt is None:
                raise ValueError(
                    "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                        dataset_dict["file_name"]
                    )
            )
            # augmentation for image
            aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
            aug_input, _ = T.apply_transform_gens(self.augmentations, aug_input)
            image = aug_input.image
            sem_seg_gt = aug_input.sem_seg


            ### augmentation for ref_img
            aug_input_ref = T.AugInput(ref_img)
            aug_input_ref, _ = T.apply_transform_gens(self.augmentations, aug_input_ref)
            ref_img = aug_input_ref.image


            # Pad image and segmentation label here!
            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)).copy())
            ref_img = torch.as_tensor(np.ascontiguousarray(ref_img.transpose(2, 0, 1)).copy())


            if sem_seg_gt is not None:
                sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

            if self.size_divisibility > 0:
                image_size = (image.shape[-2], image.shape[-1])
                padding_size = [
                    0,
                    self.size_divisibility - image_size[1],
                    0,
                    self.size_divisibility - image_size[0],
                ]
                image = F.pad(image, padding_size, value=128).contiguous()
                if sem_seg_gt is not None:
                    sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()
                
                ## additionally pad ref_img
                ref_img_size = (ref_img.shape[-2], ref_img.shape[-1])
                padding_size = [
                    0,
                    self.size_divisibility - ref_img_size[1],
                    0,
                    self.size_divisibility - ref_img_size[0],
                ]
                ref_img = F.pad(ref_img, padding_size, value=128).contiguous()

            image_shape = (image.shape[-2], image.shape[-1])  # h, w

            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"] = image
            dataset_dict["ref_image"] = ref_img
            dataset_dict.pop("img_list")
            dataset_dict.pop("ann_list")
            dataset_dict.pop("vid_dir")
            dataset_dict.pop("ann_dir")

        else:
            cur_frame = random.randrange(video_length)
            image = utils.read_image(dataset_dict['img_list'][cur_frame], format=self.img_format)
            dataset_dict = check_image_size(dataset_dict, image)
            if "ann_list" in dataset_dict:
                # PyTorch transformation not implemented for uint16, so converting it to double first
                sem_seg_gt = utils.read_image(dataset_dict['ann_list'][cur_frame]).astype("double")
                #sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name")).astype("double")
            else:
                sem_seg_gt = None
            if sem_seg_gt is None:
                raise ValueError(
                    "Cannot find 'sem_seg_file_name' for semantic segmentation dataset {}.".format(
                        dataset_dict["file_name"]
                    )
            )
            # augmentation for image
            aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
            aug_input, _ = T.apply_transform_gens(self.augmentations, aug_input)
            image = aug_input.image
            sem_seg_gt = aug_input.sem_seg

            image = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)).copy())

            if sem_seg_gt is not None:
                sem_seg_gt = torch.as_tensor(sem_seg_gt.astype("long"))

            if self.size_divisibility > 0:
                image_size = (image.shape[-2], image.shape[-1])
                padding_size = [
                    0,
                    self.size_divisibility - image_size[1],
                    0,
                    self.size_divisibility - image_size[0],
                ]
                image = F.pad(image, padding_size, value=128).contiguous()
                if sem_seg_gt is not None:
                    sem_seg_gt = F.pad(sem_seg_gt, padding_size, value=self.ignore_label).contiguous()

            image_shape = (image.shape[-2], image.shape[-1])  # h, w

            # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
            # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
            # Therefore it's important to use torch.Tensor.
            dataset_dict["image"] = image
            dataset_dict.pop("img_list")
            dataset_dict.pop("ann_list")
            dataset_dict.pop("vid_dir")
            dataset_dict.pop("ann_dir")

        
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = sem_seg_gt.long()

        if "annotations" in dataset_dict:
            raise ValueError("Semantic segmentation dataset should not have 'annotations'.")

        # Prepare per-category binary masks
        if sem_seg_gt is not None:
            sem_seg_gt = sem_seg_gt.numpy()

            #### reduce zero label
            sem_seg_gt[sem_seg_gt == 0] = 255
            sem_seg_gt = sem_seg_gt - 1
            sem_seg_gt[sem_seg_gt == 254] = 255

            instances = Instances(image_shape)
            classes = np.unique(sem_seg_gt)
            # remove ignored region
            classes = classes[classes != self.ignore_label]
            instances.gt_classes = torch.tensor(classes, dtype=torch.int64)

            masks = []
            for class_id in classes:
                masks.append(sem_seg_gt == class_id)

            if len(masks) == 0:
                # Some image does not have annotation (all ignored)
                instances.gt_masks = torch.zeros((0, sem_seg_gt.shape[-2], sem_seg_gt.shape[-1]))
            else:
                masks = BitMasks(
                    torch.stack([torch.from_numpy(np.ascontiguousarray(x.copy())) for x in masks])
                )
                instances.gt_masks = masks.tensor

            dataset_dict["instances"] = instances

        return dataset_dict



class MaskFormerVSPWVideoSemanticDatasetTestMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
        #sampling_frame_num: int = 1,
        sampling_frame_range: int = 5,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = augmentations # T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes

        # self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = False):
        augs = build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        #sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.TEST_SAMPLING_FRAME_RANGE

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            # "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        # After transforms such as cropping are applied, the bounding box may no longer
        # tightly bound the object. As an example, imagine a triangle object
        # [(0,0), (2,0), (0,2)] cropped by a box [(1,0),(2,2)] (XYXY format). The tight
        # bounding box of the cropped triangle should be [(1,0),(2,1)], which is not equal to
        # the intersection of original bounding box and the cropping box.
        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        video_name = dataset_dict["video_name"]
        #height = dataset_dict["height"]
        #width = dataset_dict["width"]
        length = dataset_dict["length"]
        img_list = dataset_dict["img_list"]
        ann_list = dataset_dict["ann_list"]
        image_list = []
        sem_seg_list = []
        is_key_frame_list = []
        for i in range(length):
            image = utils.read_image(dataset_dict["img_list"][i], format=self.image_format)
            dataset_dict = check_image_size(dataset_dict, image)
            
            # frame_id = dataset_dict["frame_id"]
            if i % self.sampling_frame_range == 0 or i == 0:
                is_key_frame = True
            else:
                is_key_frame = False
            sem_seg_gt = utils.read_image(dataset_dict["ann_list"][i], "L").squeeze(2)


            aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
            transforms, _ = T.apply_transform_gens(self.augmentations, aug_input)
            image, sem_seg_gt = aug_input.image, aug_input.sem_seg
            #### reduce zero label
            sem_seg_gt[sem_seg_gt == 0] = 255
            sem_seg_gt = sem_seg_gt - 1
            sem_seg_gt[sem_seg_gt == 254] = 255


            image_list.append(torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)).copy()))
            sem_seg_list.append(torch.as_tensor(sem_seg_gt.astype("long")))
            is_key_frame_list.append(is_key_frame)

        dataset_dict.update({
            "video_name": video_name,
            "img_list": img_list,
            "ann_list": ann_list, 
            "image_list": image_list,
            "sem_seg_list": sem_seg_list,
            "is_key_frame_list": is_key_frame_list,
            "length": length, 
        })

        return dataset_dict





class MaskFormerVSPWImageSemanticDatasetTestMapper:
    """
    A callable which takes a dataset dict in Detectron2 Dataset format,
    and map it into a format used by the model.

    This is the default callable to be used to map your dataset dict into training data.
    You may need to follow it to implement your own one for customized logic,
    such as a different way to read or transform images.
    See :doc:`/tutorials/data_loading` for details.

    The callable currently does the following:

    1. Read the image from "file_name"
    2. Applies cropping/geometric transforms to the image and annotations
    3. Prepare data and annotations to Tensor and :class:`Instances`
    """

    @configurable
    def __init__(
        self,
        is_train: bool,
        *,
        augmentations: List[Union[T.Augmentation, T.Transform]],
        image_format: str,
        use_instance_mask: bool = False,
        use_keypoint: bool = False,
        instance_mask_format: str = "polygon",
        keypoint_hflip_indices: Optional[np.ndarray] = None,
        precomputed_proposal_topk: Optional[int] = None,
        recompute_boxes: bool = False,
        #sampling_frame_num: int = 1,
        sampling_frame_range: int = 5,
    ):
        """
        NOTE: this interface is experimental.

        Args:
            is_train: whether it's used in training or inference
            augmentations: a list of augmentations or deterministic transforms to apply
            image_format: an image format supported by :func:`detection_utils.read_image`.
            use_instance_mask: whether to process instance segmentation annotations, if available
            use_keypoint: whether to process keypoint annotations if available
            instance_mask_format: one of "polygon" or "bitmask". Process instance segmentation
                masks into this format.
            keypoint_hflip_indices: see :func:`detection_utils.create_keypoint_hflip_indices`
            precomputed_proposal_topk: if given, will load pre-computed
                proposals from dataset_dict and keep the top k proposals for each image.
            recompute_boxes: whether to overwrite bounding box annotations
                by computing tight bounding boxes from instance mask annotations.
        """
        if recompute_boxes:
            assert use_instance_mask, "recompute_boxes requires instance masks"
        # fmt: off
        self.is_train               = is_train
        self.augmentations          = augmentations # T.AugmentationList(augmentations)
        self.image_format           = image_format
        self.use_instance_mask      = use_instance_mask
        self.instance_mask_format   = instance_mask_format
        self.use_keypoint           = use_keypoint
        self.keypoint_hflip_indices = keypoint_hflip_indices
        self.proposal_topk          = precomputed_proposal_topk
        self.recompute_boxes        = recompute_boxes

        # self.sampling_frame_num     = sampling_frame_num
        self.sampling_frame_range   = sampling_frame_range
        # fmt: on
        logger = logging.getLogger(__name__)
        mode = "training" if is_train else "inference"
        logger.info(f"[DatasetMapper] Augmentations used in {mode}: {augmentations}")

    @classmethod
    def from_config(cls, cfg, is_train: bool = False):
        augs = build_augmentation(cfg, is_train)
        if cfg.INPUT.CROP.ENABLED and is_train:
            augs.insert(0, T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))
            recompute_boxes = cfg.MODEL.MASK_ON
        else:
            recompute_boxes = False

        #sampling_frame_num = cfg.INPUT.SAMPLING_FRAME_NUM
        sampling_frame_range = cfg.INPUT.TEST_SAMPLING_FRAME_RANGE

        ret = {
            "is_train": is_train,
            "augmentations": augs,
            "image_format": cfg.INPUT.FORMAT,
            "use_instance_mask": cfg.MODEL.MASK_ON,
            "instance_mask_format": cfg.INPUT.MASK_FORMAT,
            "use_keypoint": cfg.MODEL.KEYPOINT_ON,
            "recompute_boxes": recompute_boxes,
            # "sampling_frame_num": sampling_frame_num,
            "sampling_frame_range": sampling_frame_range,
        }

        if cfg.MODEL.KEYPOINT_ON:
            ret["keypoint_hflip_indices"] = utils.create_keypoint_hflip_indices(cfg.DATASETS.TRAIN)

        if cfg.MODEL.LOAD_PROPOSALS:
            ret["precomputed_proposal_topk"] = (
                cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TRAIN
                if is_train
                else cfg.DATASETS.PRECOMPUTED_PROPOSAL_TOPK_TEST
            )
        return ret

    def _transform_annotations(self, dataset_dict, transforms, image_shape):
        # USER: Modify this if you want to keep them for some reason.
        for anno in dataset_dict["annotations"]:
            if not self.use_instance_mask:
                anno.pop("segmentation", None)
            if not self.use_keypoint:
                anno.pop("keypoints", None)

        # USER: Implement additional transformations if you have other types of data
        annos = [
            utils.transform_instance_annotations(
                obj, transforms, image_shape, keypoint_hflip_indices=self.keypoint_hflip_indices
            )
            for obj in dataset_dict.pop("annotations")
            if obj.get("iscrowd", 0) == 0
        ]
        instances = utils.annotations_to_instances(
            annos, image_shape, mask_format=self.instance_mask_format
        )

        if self.recompute_boxes:
            instances.gt_boxes = instances.gt_masks.get_bounding_boxes()
        dataset_dict["instances"] = utils.filter_empty_instances(instances)

    def __call__(self, dataset_dict):
        """
        Args:
            dataset_dict (dict): Metadata of one image, in Detectron2 Dataset format.

        Returns:
            dict: a format that builtin models in detectron2 accept
        """
        dataset_dict = copy.deepcopy(dataset_dict)  # it will be modified by code below
        # USER: Write your own image loading if it's not from a file
        image = utils.read_image(dataset_dict["file_name"], format=self.image_format)
        utils.check_image_size(dataset_dict, image)

        
        frame_id = dataset_dict["frame_id"]
        if frame_id % self.sampling_frame_range == 0 or frame_id == 0:
            dataset_dict["is_key_frame"] = True
        else:
            dataset_dict["is_key_frame"] = False
    

        # USER: Remove if you don't do semantic/panoptic segmentation.
        if "sem_seg_file_name" in dataset_dict:
            sem_seg_gt = utils.read_image(dataset_dict.pop("sem_seg_file_name"), "L").squeeze(2)
        else:
            sem_seg_gt = None

        aug_input = T.AugInput(image, sem_seg=sem_seg_gt)
        transforms, _ = T.apply_transform_gens(self.augmentations, aug_input)
        image, sem_seg_gt = aug_input.image, aug_input.sem_seg

        #### reduce zero label
        sem_seg_gt[sem_seg_gt == 0] = 255
        sem_seg_gt = sem_seg_gt - 1
        sem_seg_gt[sem_seg_gt == 254] = 255


        image_shape = image.shape[:2]  # h, w
        # Pytorch's dataloader is efficient on torch.Tensor due to shared-memory,
        # but not efficient on large generic data structures due to the use of pickle & mp.Queue.
        # Therefore it's important to use torch.Tensor.
        dataset_dict["image"] = torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1)).copy())
        # dataset_dict["ref_image"] = torch.as_tensor(np.ascontiguousarray(ref_img.transpose(2, 0, 1)))
        
        if sem_seg_gt is not None:
            dataset_dict["sem_seg"] = torch.as_tensor(sem_seg_gt.astype("long"))

        # USER: Remove if you don't use pre-computed proposals.
        # Most users would not need this feature.
        if self.proposal_topk is not None:
            utils.transform_proposals(
                dataset_dict, image_shape, transforms, proposal_topk=self.proposal_topk
            )

        if not self.is_train:
            # USER: Modify this if you want to keep them for some reason.
            dataset_dict.pop("annotations", None)
            dataset_dict.pop("sem_seg_file_name", None)
            return dataset_dict

        if "annotations" in dataset_dict:
            self._transform_annotations(dataset_dict, transforms, image_shape)

        return dataset_dict
