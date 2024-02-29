# Copyright (c) Facebook, Inc. and its affiliates.
import numpy as np
import logging
import sys
from fvcore.transforms.transform import (
    CropTransform,
    HFlipTransform,
    NoOpTransform,
    VFlipTransform,
)
import cv2
import random
from PIL import Image

from detectron2.data import transforms as T


class ResizeShortestEdge(T.Augmentation):
    """
    Scale the shorter edge to the given size, with a limit of `max_size` on the longer edge.
    If `max_size` is reached, then downscale so that the longer edge does not exceed max_size.
    """

    def __init__(
        self, short_edge_length, max_size=sys.maxsize, sample_style="range", interp=Image.BILINEAR, clip_frame_cnt=1
    ):
        """
        Args:
            short_edge_length (list[int]): If ``sample_style=="range"``,
                a [min, max] interval from which to sample the shortest edge length.
                If ``sample_style=="choice"``, a list of shortest edge lengths to sample from.
            max_size (int): maximum allowed longest edge length.
            sample_style (str): either "range" or "choice".
        """
        super().__init__()
        assert sample_style in ["range", "choice", "range_by_clip", "choice_by_clip"], sample_style

        self.is_range = ("range" in sample_style)
        if isinstance(short_edge_length, int):
            short_edge_length = (short_edge_length, short_edge_length)
        if self.is_range:
            assert len(short_edge_length) == 2, (
                "short_edge_length must be two values using 'range' sample style."
                f" Got {short_edge_length}!"
            )
        self._cnt = 0
        self._init(locals())

    def get_transform(self, image):
        if self._cnt % self.clip_frame_cnt == 0:
            if self.is_range:
                self.size = np.random.randint(self.short_edge_length[0], self.short_edge_length[1] + 1)
            else:
                self.size = np.random.choice(self.short_edge_length)
            if self.size == 0:
                return NoOpTransform()

            self._cnt = 0   # avoiding overflow
        self._cnt += 1

        h, w = image.shape[:2]

        scale = self.size * 1.0 / min(h, w)
        if h < w:
            newh, neww = self.size, scale * w
        else:
            newh, neww = scale * h, self.size
        if max(newh, neww) > self.max_size:
            scale = self.max_size * 1.0 / max(newh, neww)
            newh = newh * scale
            neww = neww * scale
        neww = int(neww + 0.5)
        newh = int(newh + 0.5)
        return T.ResizeTransform(h, w, newh, neww, self.interp)


class RandomFlip(T.Augmentation):
    """
    Flip the image horizontally or vertically with the given probability.
    """

    def __init__(self, prob=0.5, *, horizontal=True, vertical=False, clip_frame_cnt=1):
        """
        Args:
            prob (float): probability of flip.
            horizontal (boolean): whether to apply horizontal flipping
            vertical (boolean): whether to apply vertical flipping
        """
        super().__init__()

        if horizontal and vertical:
            raise ValueError("Cannot do both horiz and vert. Please use two Flip instead.")
        if not horizontal and not vertical:
            raise ValueError("At least one of horiz or vert has to be True!")
        self._cnt = 0

        self._init(locals())

    def get_transform(self, image):
        if self._cnt % self.clip_frame_cnt == 0:
            self.do = self._rand_range() < self.prob
            self._cnt = 0   # avoiding overflow
        self._cnt += 1

        h, w = image.shape[:2]

        if self.do:
            if self.horizontal:
                return HFlipTransform(w)
            elif self.vertical:
                return VFlipTransform(h)
        else:
            return NoOpTransform()


# def build_augmentation(cfg, is_train):
#     logger = logging.getLogger(__name__)
#     aug_list = []
#     if is_train:
#         # Crop
#         if cfg.INPUT.CROP.ENABLED:
#             aug_list.append(T.RandomCrop(cfg.INPUT.CROP.TYPE, cfg.INPUT.CROP.SIZE))

#         # Resize
#         min_size = cfg.INPUT.MIN_SIZE_TRAIN
#         max_size = cfg.INPUT.MAX_SIZE_TRAIN
#         sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
#         ms_clip_frame_cnt = cfg.INPUT.SAMPLING_FRAME_NUM if "by_clip" in cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING else 1
#         aug_list.append(ResizeShortestEdge(min_size, max_size, sample_style, clip_frame_cnt=ms_clip_frame_cnt))

#         # Flip
#         if cfg.INPUT.RANDOM_FLIP != "none":
#             if cfg.INPUT.RANDOM_FLIP == "flip_by_clip":
#                 flip_clip_frame_cnt = cfg.INPUT.SAMPLING_FRAME_NUM
#             else:
#                 flip_clip_frame_cnt = 1

#             aug_list.append(
#                 # NOTE using RandomFlip modified for the support of flip maintenance
#                 RandomFlip(
#                     horizontal=(cfg.INPUT.RANDOM_FLIP == "horizontal") or (cfg.INPUT.RANDOM_FLIP == "flip_by_clip"),
#                     vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
#                     clip_frame_cnt=flip_clip_frame_cnt,
#                 )
#             )

#         # Additional augmentations : brightness, contrast, saturation, rotation
#         augmentations = cfg.INPUT.AUGMENTATIONS
#         if "brightness" in augmentations:
#             aug_list.append(T.RandomBrightness(0.9, 1.1))
#         if "contrast" in augmentations:
#             aug_list.append(T.RandomContrast(0.9, 1.1))
#         if "saturation" in augmentations:
#             aug_list.append(T.RandomSaturation(0.9, 1.1))
#         if "rotation" in augmentations:
#             aug_list.append(
#                 T.RandomRotation(
#                     [-15, 15], expand=False, center=[(0.4, 0.4), (0.6, 0.6)], sample_style="range"
#                 )
#             )
#     else:
#         # Resize
#         min_size = cfg.INPUT.MIN_SIZE_TEST
#         max_size = cfg.INPUT.MAX_SIZE_TEST
#         sample_style = "choice"
#         aug_list.append(T.ResizeShortestEdge(min_size, max_size, sample_style))

#     return aug_list


class RandomCrop(T.Augmentation):
    """
    Randomly crop a rectangle region out of an image.
    """

    def __init__(self, crop_type: str, crop_size, clip_frame_cnt=1):
        """
        Args:
            crop_type (str): one of "relative_range", "relative", "absolute", "absolute_range".
            crop_size (tuple[float, float]): two floats, explained below.

        - "relative": crop a (H * crop_size[0], W * crop_size[1]) region from an input image of
          size (H, W). crop size should be in (0, 1]
        - "relative_range": uniformly sample two values from [crop_size[0], 1]
          and [crop_size[1]], 1], and use them as in "relative" crop type.
        - "absolute" crop a (crop_size[0], crop_size[1]) region from input image.
          crop_size must be smaller than the input image size.
        - "absolute_range", for an input of size (H, W), uniformly sample H_crop in
          [crop_size[0], min(H, crop_size[1])] and W_crop in [crop_size[0], min(W, crop_size[1])].
          Then crop a region (H_crop, W_crop).
        """
        # TODO style of relative_range and absolute_range are not consistent:
        # one takes (h, w) but another takes (min, max)
        super().__init__()
        assert crop_type in ["relative_range", "relative", "absolute", "absolute_range"]
        self._cnt = 0
        self._init(locals())

    def get_transform(self, image):
        if self._cnt % self.clip_frame_cnt == 0:
            h, w = image.shape[:2]
            self.croph, self.cropw = self.get_crop_size((h, w))
            assert h >= self.croph and w >= self.cropw, "Shape computation in {} has bugs.".format(self)
            self.h0 = np.random.randint(h - self.croph + 1)
            self.w0 = np.random.randint(w - self.cropw + 1)
            self._cnt = 0
        self._cnt += 1
        return CropTransform(self.w0, self.h0, self.cropw, self.croph)

    def get_crop_size(self, image_size):
        """
        Args:
            image_size (tuple): height, width

        Returns:
            crop_size (tuple): height, width in absolute pixels
        """
        h, w = image_size
        if self.crop_type == "relative":
            ch, cw = self.crop_size
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "relative_range":
            crop_size = np.asarray(self.crop_size, dtype=np.float32)
            ch, cw = crop_size + np.random.rand(2) * (1 - crop_size)
            return int(h * ch + 0.5), int(w * cw + 0.5)
        elif self.crop_type == "absolute":
            return (min(self.crop_size[0], h), min(self.crop_size[1], w))
        elif self.crop_type == "absolute_range":
            assert self.crop_size[0] <= self.crop_size[1]
            ch = np.random.randint(min(h, self.crop_size[0]), min(h, self.crop_size[1]) + 1)
            cw = np.random.randint(min(w, self.crop_size[0]), min(w, self.crop_size[1]) + 1)
            return ch, cw
        else:
            raise NotImplementedError("Unknown crop type {}".format(self.crop_type))


class RandomCrop_CategoryAreaConstraint(T.Augmentation):
    """
    Similar to :class:`RandomCrop`, but find a cropping window such that no single category
    occupies a ratio of more than `single_category_max_area` in semantic segmentation ground
    truth, which can cause unstability in training. The function attempts to find such a valid
    cropping window for at most 10 times.
    """

    def __init__(
        self,
        crop_type: str,
        crop_size,
        single_category_max_area: float = 1.0,
        ignored_category: int = None,
        clip_frame_cnt=1
    ):
        """
        Args:
            crop_type, crop_size: same as in :class:`RandomCrop`
            single_category_max_area: the maximum allowed area ratio of a
                category. Set to 1.0 to disable
            ignored_category: allow this category in the semantic segmentation
                ground truth to exceed the area ratio. Usually set to the category
                that's ignored in training.
        """
        self.crop_aug = RandomCrop(crop_type, crop_size, clip_frame_cnt)
        self._cnt = 0
        self._init(locals())

    def get_transform(self, image, sem_seg):
        if self.single_category_max_area >= 1.0:
            return self.crop_aug.get_transform(image)
        else:
            h, w = sem_seg.shape
            for _ in range(10):
                crop_size = self.crop_aug.get_crop_size((h, w))
                y0 = np.random.randint(h - crop_size[0] + 1)
                x0 = np.random.randint(w - crop_size[1] + 1)
                sem_seg_temp = sem_seg[y0 : y0 + crop_size[0], x0 : x0 + crop_size[1]]
                labels, cnt = np.unique(sem_seg_temp, return_counts=True)
                if self.ignored_category is not None:
                    cnt = cnt[labels != self.ignored_category]
                if len(cnt) > 1 and np.max(cnt) < np.sum(cnt) * self.single_category_max_area:
                    break
            crop_tfm = CropTransform(x0, y0, crop_size[1], crop_size[0])
            return crop_tfm


class ColorAugSSDTransform(T.Transform):
    """
    A color related data augmentation used in Single Shot Multibox Detector (SSD).

    Wei Liu, Dragomir Anguelov, Dumitru Erhan, Christian Szegedy,
       Scott Reed, Cheng-Yang Fu, Alexander C. Berg.
       SSD: Single Shot MultiBox Detector. ECCV 2016.

    Implementation based on:

     https://github.com/weiliu89/caffe/blob
       /4817bf8b4200b35ada8ed0dc378dceaf38c539e4
       /src/caffe/util/im_transforms.cpp

     https://github.com/chainer/chainercv/blob
       /7159616642e0be7c5b3ef380b848e16b7e99355b/chainercv
       /links/model/ssd/transforms.py
    """

    def __init__(
        self,
        img_format,
        brightness_delta=32,
        contrast_low=0.5,
        contrast_high=1.5,
        saturation_low=0.5,
        saturation_high=1.5,
        hue_delta=18,
        clip_frame_cnt=1
    ):
        super().__init__()
        assert img_format in ["BGR", "RGB"]
        self.is_rgb = img_format == "RGB"
        del img_format
        self._cnt = 0
        self._set_attributes(locals())

    def apply_coords(self, coords):
        return coords

    def apply_segmentation(self, segmentation):
        return segmentation

    def apply_image(self, img, interp=None):
        if self.is_rgb:
            img = img[:, :, [2, 1, 0]]
        if self._cnt % self.clip_frame_cnt == 0:
            self.random_order = random.randrange(2)
            self.random_b = random.randrange(2)
            self.random_c = random.randrange(2)
            self.random_s = random.randrange(2)
            self.random_h = random.randrange(2)
            self.b_beta = random.uniform(-self.brightness_delta, self.brightness_delta)
            self.c_alpha = random.uniform(self.contrast_low, self.contrast_high)
            self.s_alpha = random.uniform(self.saturation_low, self.saturation_high)
            self.h_delta = random.randint(-self.hue_delta, self.hue_delta)
            self._cnt = 0
        self._cnt += 1
        
        img = self.brightness(img)
        if self.random_order:
            img = self.contrast(img)
            img = self.saturation(img)
            img = self.hue(img)
        else:
            img = self.saturation(img)
            img = self.hue(img)
            img = self.contrast(img)
        if self.is_rgb:
            img = img[:, :, [2, 1, 0]]
        
        return img

    def convert(self, img, alpha=1, beta=0):
        img = img.astype(np.float32) * alpha + beta
        img = np.clip(img, 0, 255)
        return img.astype(np.uint8)

    def brightness(self, img):
        if self.random_b:
            return self.convert(
                img, beta=self.b_beta
            )
        return img

    def contrast(self, img):
        if self.random_c:
            return self.convert(img, alpha=self.c_alpha)
        return img

    def saturation(self, img):
        if self.random_s:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 1] = self.convert(
                img[:, :, 1], alpha=self.s_alpha
            )
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img

    def hue(self, img):
        if self.random_h:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            img[:, :, 0] = (
                img[:, :, 0].astype(int) + self.h_delta
            ) % 180
            return cv2.cvtColor(img, cv2.COLOR_HSV2BGR)
        return img



def build_augmentation(cfg, is_train=True):
    # Build augmentation
    if is_train:
        augs = [
            ResizeShortestEdge(
                cfg.INPUT.MIN_SIZE_TRAIN,
                cfg.INPUT.MAX_SIZE_TRAIN,
                cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING,
                clip_frame_cnt=cfg.INPUT.SAMPLING_FRAME_NUM + 1
            )
        ]
        if cfg.INPUT.CROP.ENABLED:
            augs.append(
                RandomCrop_CategoryAreaConstraint(
                    cfg.INPUT.CROP.TYPE,
                    cfg.INPUT.CROP.SIZE,
                    cfg.INPUT.CROP.SINGLE_CATEGORY_MAX_AREA,
                    cfg.MODEL.SEM_SEG_HEAD.IGNORE_VALUE,
                    clip_frame_cnt=cfg.INPUT.SAMPLING_FRAME_NUM + 1
                )
            )
        if cfg.INPUT.COLOR_AUG_SSD:
            augs.append(ColorAugSSDTransform(img_format=cfg.INPUT.FORMAT, clip_frame_cnt=cfg.INPUT.SAMPLING_FRAME_NUM + 1))
        
        
        if cfg.INPUT.RANDOM_FLIP != "none":
            if cfg.INPUT.RANDOM_FLIP == "flip_by_clip":
                flip_clip_frame_cnt = cfg.INPUT.SAMPLING_FRAME_NUM + 1
            else:
                flip_clip_frame_cnt = 1
        augs.append(
                # NOTE using RandomFlip modified for the support of flip maintenance
                RandomFlip(
                    horizontal=(cfg.INPUT.RANDOM_FLIP == "horizontal") or (cfg.INPUT.RANDOM_FLIP == "flip_by_clip"),
                    vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
                    clip_frame_cnt=flip_clip_frame_cnt,
                )
            )
    else:

        # if is_train:
        #     min_size = cfg.INPUT.MIN_SIZE_TRAIN
        #     max_size = cfg.INPUT.MAX_SIZE_TRAIN
        #     sample_style = cfg.INPUT.MIN_SIZE_TRAIN_SAMPLING
        # else:
        min_size = cfg.INPUT.MIN_SIZE_TEST
        max_size = cfg.INPUT.MAX_SIZE_TEST
        sample_style = "choice"
        augs = [ResizeShortestEdge(min_size, max_size, sample_style, clip_frame_cnt=cfg.INPUT.SAMPLING_FRAME_NUM + 1)]
        # if is_train and cfg.INPUT.RANDOM_FLIP != "none":
        #     augs.append(
        #         RandomFlip(
        #             horizontal=cfg.INPUT.RANDOM_FLIP == "horizontal" or (cfg.INPUT.RANDOM_FLIP == "flip_by_clip"),
        #             vertical=cfg.INPUT.RANDOM_FLIP == "vertical",
        #         )
        #     )

    return augs
