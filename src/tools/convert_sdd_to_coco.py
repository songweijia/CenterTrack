#!/usr/bin/env python3
import cv2
import os
import json
import numpy as np

# STEP 1: convert SDD videos to images
# STEP 2: extract information and create coco data sets

# sdd data
# SDD_PATH/videos/${scene}/videoX/video.mov
SDD_PATH='/root/download/stanford_drone_dataset'
# dest data
# OUT_PATH/videos/${scene}/videoX/framesX.jpg
OUT_PATH='/root/download/stanford_drone_dataset_coco'
# SCENES=['bookstore','coupa','deathCircle','gates','hyang','little','nexus','quad']
SCENES=['bookstore']
# OUT_PATH/annotations/{scene}/{#vid}/train|val|test.json
SPLITS = ['train', 'val', 'test']
HALF_VIDEO = True
CREATE_SPLITTED_ANN = True
CATEGORIES = ['Biker','Bus','Car','Cart','Pedestrian','Skater']

def video_to_frames(src,des):
    """
    Convert SDD video to frames
    @param src - The source path of the sdd data
    @param des - The output path of the coco data
    """
    for scene in SCENES:
        for vdir in os.listdir(f"{SDD_PATH}/videos/{scene}"):
            mov_file = f"{SDD_PATH}/videos/{scene}/{vdir}/video.mov"
            des_dir = f"{OUT_PATH}/videos/{scene}/{vdir}"
            if not os.path.exists(des_dir):
                os.makedirs(des_dir)
            vidcap = cv2.VideoCapture(mov_file)
            success,image = vidcap.read()
            count = 0
            while success:
                cv2.imwrite(f"{des_dir}/frame{count}.jpg",image)
                success,image = vidcap.read()
                count += 1

def load_sdd_annotation(sdd_ann_file):
    """
    Load sdd annotation as numpy
    format:
    1 - Track ID.
    2 - xmin. (left)
    3 - ymin. (top)
    4 - xmax. (right)
    5 - ymax. (bottom)
    6 - frame.
    7 - lost. (if 1, the annotation is outside of the view screen.)
    8 - occluded.   (if 1, the annotation is occluded.)
    9 - generated.  (if 1, the annotation was automatically interpolated.)
    10 - category.  ( from 0 to 5, matching CATEGORIES)
    """
    with open(sdd_ann_file) as f:
        line = f.readline()
        ann = []
        while line:
            for i in range(len(CATEGORIES)):
                line = line.replace(f"\"{CATEGORIES[i]}\"",str(i+1))
            line = line.replace(" ",",")
            ann.append(eval(f"[{line}]"))
            line = f.readline()
    return ann

def generate_coco_annotation(src,des):
    """
    generate coco annotations
    @param src - The source path of the sdd data
    @param des - The output path of the coco data, frame is in place
    """
    for scene in SCENES:
        video_cnt = 0
        for vid in sorted(os.listdir(f"{OUT_PATH}/videos/{scene}/")):
            # skip annotations folder
            if vid == "annotations":
                continue
            video_cnt += 1
            for split in SPLITS:
                des_dir = f"{des}/videos/{scene}/{vid}/annotations/"
                if not os.path.exists(des_dir):
                    os.makedirs(des_dir)
                out_path = f"{des_dir}/{split}.json"
                out = {'images':[],
                       'annotations':[],
                       'categories':[{'id':1,'name':CATEGORIES[0]},
                                     {'id':2,'name':CATEGORIES[1]},
                                     {'id':3,'name':CATEGORIES[2]},
                                     {'id':4,'name':CATEGORIES[3]},
                                     {'id':5,'name':CATEGORIES[4]},
                                     {'id':6,'name':CATEGORIES[5]}],
                       'videos':[]}
                ann_cnt = 0
                # 1 - create image description
                out['videos'].append({
                    'id': video_cnt,
                    'file_name': vid})
                vid_path = f"{OUT_PATH}/videos/{scene}/{vid}/"
                images = os.listdir(vid_path)
                num_images = len([image for image in images if 'jpg' in image])
                image_range = [0, num_images - 1]
                for i in range(num_images):
                    image_info = {'file_name': f'frame{i}.jpg',
                                  'id': i,
                                  'frame_id': i,
                                  'prev_image_id': i-1,
                                  'next_image_id': i+1,
                                  'video_id': video_cnt}
                    out['images'].append(image_info)
                # test will skip the annotations
                if split != 'test':
                    # 2 - create the annotations
                    # 2.1 load SDD annotations.
                    sdd_anns = load_sdd_annotation(f"{src}/annotations/{scene}/{vid}/annotations.txt")
                    ann_cnt = 1
                    # 2.2 transform SDD annotations.
                    for sdd_ann in sdd_anns:
                        if sdd_ann[6]==1:
                            continue
                        ann = {'id': ann_cnt,
                               'category_id': sdd_ann[9],
                               'image_id': sdd_ann[5],
                               'track_id': sdd_ann[0],
                               'bbox': [sdd_ann[1],sdd_ann[2],sdd_ann[3]-sdd_ann[1], sdd_ann[4]-sdd_ann[2]],
                               'conf': 1.0} # what is conf?
                        out['annotations'].append(ann)
                # 3 - write output 
                json.dump(out,open(out_path, 'w'))

if __name__=='__main__':
    # This has been done.
    # video_to_frames(SDD_PATH,OUT_PATH)
    generate_coco_annotation(SDD_PATH,OUT_PATH)
