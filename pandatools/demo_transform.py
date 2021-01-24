import os
import json
import glob
import random
from panda_utils import DetRes2GT,generate_coco_anno,generate_res_from_gt,generate_coco_anno_vehicles,generate_coco_anno_persons,persons_challenge_GT2coco,result2panda
print("----------------res2panda")
# detrespath="/root/data/gvision/dataset/split/person_s0.5_t0.9_14_test/image_annos/coco_challenge_results.json"
# outgtpath="/root/data/gvision/dataset/split/person_s0.5_t0.9_14_test/image_annos/panda_results11.json"
# gtannopath="/root/data/gvision/dataset/split/person_s0.5_t0.9_14_test/image_annos/person_s0.5_t0.9_14_split_test.json"
# result2panda(detrespath, outgtpath, gtannopath)
"""
---------------------------------------------panda2coco"
"""
print("------------------------TRAIN_pv")
print("---------------------------------------------persons")
basepath="/root/data/gvision/dataset/raw_data"
personsrcfile="/root/data/gvision/dataset/raw_data/image_annos/person_bbox_train.json"
tgtfile="/root/data/gvision/dataset/train_center/image_annos/split_p.json"
scale=1
# generate_coco_anno_persons(basepath,personsrcfile, tgtfile,scale)
print("---------------------------------------------vehicles")
vehiclesrcfile="/root/data/gvision/dataset/raw_data/image_annos/vehicle_bbox_train.json"
tgtfile="/root/data/gvision/dataset/raw_data/image_annos/coco_vehicles_bbox_train.json"
generate_coco_anno_vehicles(basepath,vehiclesrcfile,tgtfile,scale)

print("---------------------------------------------persons&vehicles")
tgtfile="/root/data/gvision/dataset/raw_data/image_annos/coco_bbox_train.json"
# generate_coco_anno(basepath,personsrcfile,vehiclesrcfile,tgtfile,scale)

# "---------------------------------------------persons----------------challenge2panda2coco"
# personsrcfile="/root/data/gvision/dataset/split/person_s0.5_t0.9_14_test/image_annos/panda_results.json"
# tgtfile="/root/data/gvision/dataset/split/person_s0.5_t0.9_14_test/image_annos/coco_results.json"
# persons_challenge_GT2coco(personsrcfile, tgtfile)

# import json 
# with open("/root/data/gvision/dataset/split/person_s0.5_t0.9_14_test/image_annos/coco_challenge_results.json", 'r') as load_f:
#     seqinfo_dict = json.load(load_f)
# print(len(seqinfo_dict))


