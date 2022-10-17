#!/usr/bin/env bash
# python main.py tracking --exp_id mot17_half_sc --dataset custom --custom_dataset_ann_path ../data/mot17/annotations/train_half.json --custom_dataset_img_path ../data/mot17/train/ --input_h 544 --input_w 960 --num_classes 1 --pre_hm --ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0
python main.py tracking --exp_id sdd_sample_0 --dataset custom --custom_dataset_ann_path \
~/download/sdd_sample_coco/videos/bookstore/video0/annotations/train.json --custom_dataset_img_path \
~/download/sdd_sample_coco/videos/bookstore/video0/ --input_h 1088 --input_w 1440 --num_classes 1 --pre_hm \
--ltrb_amodal --same_aug --hm_disturb 0.05 --lost_disturb 0.4 --fp_disturb 0.1 --gpus 0 --batch_size=3
