#!/bin/bash

# train and evaluate per-frame baseline
MODEL=Base-VSPW-swin_base
python train_net_vss.py  --num-gpus=8 \
      --config-file=configs/vspw/semantic-segmentation/"$MODEL".yaml \
      OUTPUT_DIR ./experiments/mask2former/vspw/"$MODEL" \
      SOLVER.IMS_PER_BATCH 16

python train_net_vss.py  --num-gpus=8 \
      --eval-only \
      --config-file=configs/vspw/semantic-segmentation/"$MODEL".yaml \
      OUTPUT_DIR ./experiments/mask2former/vspw/"$MODEL" \
      MODEL.WEIGHTS /path/to/your/trained/model

# train and evaluate mpvss
MODEL=MPVSS-VSPW-swin_base
python train_net_vss.py  --num-gpus=8 \
      --config-file=configs/vspw/video-semantic-segmentation/"$MODEL".yaml \
      OUTPUT_DIR ./experiments/mpvss/vspw/"$MODEL" \
      SOLVER.IMS_PER_BATCH 16

python train_net_vss.py --num-gpus=8 \
      --eval-only \
      --config-file=configs/vspw/video-semantic-segmentation/"$MODEL".yaml \
      MODEL.WEIGHTS /path/to/your/trained/model \
      OUTPUT_DIR ./experiments/mpvss/vspw/"$MODEL"