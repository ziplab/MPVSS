# Copyright (c) Facebook, Inc. and its affiliates.
# Modifications copyright (c) 2022 ZIP Group
import os
import json
import logging

# try:
#     import cv2  # noqa
# except ImportError:
#     # OpenCV is an optional dependency at the moment
#     pass

from detectron2.utils.file_io import PathManager
from detectron2.data.datasets.builtin_meta import _get_builtin_metadata
from detectron2.data import DatasetCatalog, MetadataCatalog

logger = logging.getLogger(__name__)

# ==== Predefined splits for raw cityscapes images ===========
CLASSES = {"others": "0", 
            "wall": "1", 
            "ceiling": "2", 
            "door": "3", 
            "stair": "4", 
            "ladder": "5", 
            "escalator": "6", 
            "Playground_slide": "7", 
            "handrail_or_fence": "8", 
            "window": "9", 
            "rail": "10", 
            "goal": "11", 
            "pillar": "12", 
            "pole": "13", 
            "floor": "14",
            "ground": "15", 
            "grass": "16", 
            "sand": "17", 
            "athletic_field": "18", 
            "road": "19", 
            "path": "20",
            "crosswalk": "21", 
            "building": "22", 
            "house": "23", 
            "bridge": "24", 
            "tower": "25", 
            "windmill": "26",
            "well_or_well_lid": "27", 
            "other_construction": "28", 
            "sky": "29", 
            "mountain": "30", 
            "stone": "31",
            "wood": "32", 
            "ice": "33", 
            "snowfield": "34", 
            "grandstand": "35", 
            "sea": "36", 
            "river": "37", 
            "lake": "38", 
            "waterfall": "39", 
            "water": "40", 
            "billboard_or_Bulletin_Board": "41", 
            "sculpture": "42",
            "pipeline": "43", 
            "flag": "44", 
            "parasol_or_umbrella": "45", 
            "cushion_or_carpet": "46", 
            "tent": "47",
            "roadblock": "48", 
            "car": "49", 
            "bus": "50", 
            "truck": "51", 
            "bicycle": "52", 
            "motorcycle": "53",
            "wheeled_machine": "54", 
            "ship_or_boat": "55", 
            "raft": "56", 
            "airplane": "57", 
            "tyre": "58",
            "traffic_light": "59", 
            "lamp": "60", 
            "person": "61", 
            "cat": "62", 
            "dog": "63", 
            "horse": "64",
            "cattle": "65", 
            "other_animal": "66", 
            "tree": "67", 
            "flower": "68", 
            "other_plant": "69", 
            "toy": "70",
            "ball_net": "71", 
            "backboard": "72", 
            "skateboard": "73", 
            "bat": "74", 
            "ball": "75",
            "cupboard_or_showcase_or_storage_rack": "76", 
            "box": "77", 
            "traveling_case_or_trolley_case": "78",
            "basket": "79", 
            "bag_or_package": "80", 
            "trash_can": "81", 
            "cage": "82", 
            "plate": "83",
            "tub_or_bowl_or_pot": "84", 
            "bottle_or_cup": "85", 
            "barrel": "86", 
            "fishbowl": "87", 
            "bed": "88",
            "pillow": "89", 
            "table_or_desk": "90", 
            "chair_or_seat": "91", 
            "bench": "92", 
            "sofa": "93",
            "shelf": "94", 
            "bathtub": "95", 
            "gun": "96", 
            "commode": "97", 
            "roaster": "98", 
            "other_machine": "99",
            "refrigerator": "100", 
            "washing_machine": "101", 
            "Microwave_oven": "102", 
            "fan": "103", 
            "curtain": "104",
            "textiles": "105", 
            "clothes": "106", 
            "painting_or_poster": "107", 
            "mirror": "108", 
            "flower_pot_or_vase": "109",
            "clock": "110", 
            "book": "111", 
            "tool": "112", 
            "blackboard": "113", 
            "tissue": "114", 
            "screen_or_television": "115",
            "computer": "116", 
            "printer": "117", 
            "Mobile_phone": "118", 
            "keyboard": "119", 
            "other_electronic_product": "120",
            "fruit": "121", 
            "food": "122", 
            "instrument": "123",
            "train": "124"}

FG_CLASSES = CLASSES
FG_CLASSES.pop("others")
# CLASSES = dict(list(CLASSES.keys())[1:])
# len()=124

# len()=124
PALETTE = [[120, 120, 120], [180, 120, 120], [6, 230, 230], [80, 50, 50],
            [4, 200, 3], [120, 120, 80], [140, 140, 140], [204, 5, 255],
            [230, 230, 230], [4, 250, 7], [224, 5, 255], [235, 255, 7],
            [150, 5, 61], [120, 120, 70], [8, 255, 51], [255, 6, 82],
            [143, 255, 140], [204, 255, 4], [255, 51, 7], [204, 70, 3],
            [0, 102, 200], [61, 230, 250], [255, 6, 51], [11, 102, 255],
            [255, 7, 71], [255, 9, 224], [9, 7, 230], [220, 220, 220],
            [255, 9, 92], [112, 9, 255], [8, 255, 214], [7, 255, 224],
            [255, 184, 6], [10, 255, 71], [255, 41, 10], [7, 255, 255],
            [224, 255, 8], [102, 8, 255], [255, 61, 6], [255, 194, 7],
            [255, 122, 8], [0, 255, 20], [255, 8, 41], [255, 5, 153],
            [6, 51, 255], [235, 12, 255], [160, 150, 20], [0, 163, 255],
            [140, 140, 140], [250, 10, 15], [20, 255, 0], [31, 255, 0],
            [255, 31, 0], [255, 224, 0], [153, 255, 0], [0, 0, 255],
            [255, 71, 0], [0, 235, 255], [0, 173, 255], [31, 0, 255],
            [11, 200, 200], [255, 82, 0], [0, 255, 245], [0, 61, 255],
            [0, 255, 112], [0, 255, 133], [255, 0, 0], [255, 163, 0],
            [255, 102, 0], [194, 255, 0], [0, 143, 255], [51, 255, 0],
            [0, 82, 255], [0, 255, 41], [0, 255, 173], [10, 0, 255],
            [173, 255, 0], [0, 255, 153], [255, 92, 0], [255, 0, 255],
            [255, 0, 245], [255, 0, 102], [255, 173, 0], [255, 0, 20],
            [255, 184, 184], [0, 31, 255], [0, 255, 61], [0, 71, 255],
            [255, 0, 204], [0, 255, 194], [0, 255, 82], [0, 10, 255],
            [0, 112, 255], [51, 0, 255], [0, 194, 255], [0, 122, 255],
            [0, 255, 163], [255, 153, 0], [0, 255, 10], [255, 112, 0],
            [143, 255, 0], [82, 0, 255], [163, 255, 0], [255, 235, 0],
            [8, 184, 170], [133, 0, 255], [0, 255, 92], [184, 0, 255],
            [255, 0, 31], [0, 184, 255], [0, 214, 255], [255, 0, 112],
            [92, 255, 0], [0, 224, 255], [112, 224, 255], [70, 184, 160],
            [163, 0, 255], [153, 0, 255], [71, 255, 0], [255, 0, 163],
            [255, 204, 0], [255, 0, 143], [0, 255, 235], [133, 255, 0]
            ]


_RAW_VSPW_SPLITS = {
    "vspw_{task}_train": ("vspw/train.txt", "vspw/data"), 
    "vspw_{task}_val": ("vspw/val.txt", "vspw/data"),
    "vspw_{task}_test": ("vspw/test.txt", "vspw/data"), 
}


def load_vspw_semantic_video(split_list_file, video_dir):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".

    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []
    with open(split_list_file) as f:
        lines = f.readlines()
        videonames = [line[:-1] for line in lines] # video name is recorded in 127_-hIVCYO4C90
        
    videopathlists = [os.path.join(video_dir, v, "origin") for v in videonames] # data/xxxxx/origin
    annpathlists = [os.path.join(video_dir, v, "mask") for v in videonames] # data/xxxxx/mask 

    # if "train" in split_list_file:  # sample in video manner
    for vid_name, vid_dir, ann_dir in zip(videonames, videopathlists, annpathlists):
        img_list = sorted(PathManager.ls(vid_dir))
        ann_list = sorted(PathManager.ls(ann_dir))
        assert len(img_list) == len(ann_list), "Number of video frames and annotations for video {vid_name} not match."
        
        ret.append({
            "video_name": vid_name, 
            "vid_dir": vid_dir,
            "ann_dir": ann_dir,
            "img_list": [os.path.join(vid_dir, i) for i in img_list] ,
            "ann_list": [os.path.join(ann_dir, a) for a in ann_list] , 
            "length": len(img_list),
            "height": 480,
            "width": 853            
            })
    if "val" in split_list_file:
        def take_length(elem): 
            return elem["length"]
        ret.sort(key=take_length)  # sort list by video length

    assert len(ret), f"No videos found in {video_dir}"
    return ret
    

def register_vspw_video_semantic(root):

    for key, (list_file, data_root) in _RAW_VSPW_SPLITS.items():
        # meta = _get_builtin_metadata("cityscapes")
        split_list_file = os.path.join(root, list_file)
        video_dir = os.path.join(root, data_root)

        sem_key = key.format(task="sem_seg_video")
        DatasetCatalog.register(
            sem_key, lambda x=split_list_file, y=video_dir: load_vspw_semantic_video(x, y)
        )
        #### to do 
        stuff_ids = [v for k,v in FG_CLASSES.items()]

        # For semantic segmentation, this mapping maps from contiguous stuff id
        # (in [0, 91], used in models) to ids in the dataset (used for processing results)
        stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
        stuff_classes = [k for k, v in FG_CLASSES.items()]
        MetadataCatalog.get(sem_key).set(
            video_dir=video_dir,
            evaluator_type="vspw_sem_seg", 
            ignore_label=255,
            stuff_classes=stuff_classes,
            stuff_ids=stuff_ids,
            stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
            plattes=PALETTE,
        )


def load_vspw_semantic_image(split_list_file, video_dir):
    """
    Args:
        image_dir (str): path to the raw dataset. e.g., "~/cityscapes/leftImg8bit/train".
        gt_dir (str): path to the raw annotations. e.g., "~/cityscapes/gtFine/train".

    Returns:
        list[dict]: a list of dict, each has "file_name" and
            "sem_seg_file_name".
    """
    ret = []
    with open(split_list_file) as f:
        lines = f.readlines()
        videonames = [line[:-1] for line in lines] # video name is recorded in 127_-hIVCYO4C90
        
    videopathlists = [os.path.join(video_dir, v, "origin") for v in videonames] # data/xxxxx/origin
    annpathlists = [os.path.join(video_dir, v, "mask") for v in videonames] # data/xxxxx/mask 

    if "train" in split_list_file:  # sample in video manner
        for vid_name, vid_dir, ann_dir in zip(videonames, videopathlists, annpathlists):
            img_list = sorted(PathManager.ls(vid_dir))
            ann_list = sorted(PathManager.ls(ann_dir))
            assert len(img_list) == len(ann_list), "Number of video frames and annotations for video {vid_name} not match."
            
            ret.append({
                "video_name": vid_name, 
                "vid_dir": vid_dir,
                "ann_dir": ann_dir,
                "img_list": [os.path.join(vid_dir, i) for i in img_list] ,
                "ann_list": [os.path.join(ann_dir, a) for a in ann_list] , 
                "length": len(img_list),
                "height": 480,
                "width": 853            
                })

        assert len(ret), f"No videos found in {video_dir}"
    else:  # sample in image manner
        ret = []
        for vid_name, vid_dir, ann_dir in zip(videonames, videopathlists, annpathlists):
            img_list = sorted(PathManager.ls(vid_dir))
            ann_list = sorted(PathManager.ls(ann_dir))
            assert len(img_list) == len(ann_list), "Number of video frames and annotations for video {vid_name} not match."
            
            img_list = [os.path.join(vid_dir, i) for i in img_list]
            ann_list = [os.path.join(ann_dir, a) for a in ann_list]
            video_length = len(img_list)
            for i, (img, ann) in enumerate(zip(img_list, ann_list)):
                ret.append({
                    "file_name": img,
                    "sem_seg_file_name": ann,
                    "frame_id": i,
                    "video_name": vid_name,
                    "video_length": video_length
                })

    return ret

def register_vspw_image_semantic(root):

    for key, (list_file, data_root) in _RAW_VSPW_SPLITS.items():
        # meta = _get_builtin_metadata("cityscapes")
        split_list_file = os.path.join(root, list_file)
        video_dir = os.path.join(root, data_root)
        sem_key = key.format(task="sem_seg_image")
        DatasetCatalog.register(
            sem_key, lambda x=split_list_file, y=video_dir: load_vspw_semantic_image(x, y)
        )
        #### to do 
        stuff_ids = [v for k,v in FG_CLASSES.items()]

        # For semantic segmentation, this mapping maps from contiguous stuff id
        # (in [0, 91], used in models) to ids in the dataset (used for processing results)
        stuff_dataset_id_to_contiguous_id = {k: i for i, k in enumerate(stuff_ids)}
        stuff_classes = [k for k, v in FG_CLASSES.items()]
        MetadataCatalog.get(sem_key).set(
            video_dir=video_dir,
            evaluator_type="sem_seg",   
            ignore_label=255,
            stuff_classes=stuff_classes,
            stuff_ids=stuff_ids,
            stuff_dataset_id_to_contiguous_id=stuff_dataset_id_to_contiguous_id,
            plattes=PALETTE,
        )


_root = os.getenv("DETECTRON2_DATASETS", "datasets")
# for video models; in config file, using "vspw_sem_seg_video_train" or "vspw_sem_seg_video_val"
register_vspw_video_semantic(_root) 

# for base Mask2Former; in config file, using "vspw_sem_seg_image_train" or "vspw_sem_seg_image_val"
register_vspw_image_semantic(_root) 