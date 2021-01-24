import json
import cv2
import os
load_path="/root/data/gvision/CrowdDet-master/data/CrowdHuman/annotation_train_hbox.json"
output_path="/root/data/gvision/CrowdDet-master/data/CrowdHuman/annotation_train_hbox_hw.json"
with open(load_path,'r') as load_f:
    dataset_dicts = json.load(load_f)
def ceil_hw(x,ceil):
    if x>ceil:
        print("x>ceil")
        x=ceil
    else:
        x=x
    return x
def down_xy(x):
    if x<0:
        print("x<0")
        x=0
    else:
        x=x
    return x
# for i in dataset_dicts["annotations"]:
#     images_selected=[x for x in dataset_dicts["images"] if x["image_id"]==i["id"]] 
#     images_dict=images_selected[0]
#     height,width=images_dict["height"],images_dict["width"]
#     x, y, w, h =i["bbox"]
#     xmax,ymax=x+w,y+h
#     w=ceil_hw(xmax,width)-x
#     h=ceil_hw(ymax,height)-y
#     i["bbox"]=[x,y,w,h]
# with open(output_path, 'w') as load_f:
#     dataset_dicts= json.dumps(dataset_dicts,indent=2)
#     load_f.write(dataset_dicts)


for images_dict in dataset_dicts["images"]:
    print(images_dict["id"])
    height,width=images_dict["height"],images_dict["width"]
    for i in dataset_dicts["annotations"]:
        if i["image_id"]==images_dict["id"]:
            x, y, w, h =i["bbox"]
            xmax,ymax=x+w,y+h
            w=ceil_hw(xmax,width)-x
            h=ceil_hw(ymax,height)-y
            x=down_xy(x)
            y=down_xy(y)
            i["bbox"]=[x,y,w,h]
with open(output_path, 'w') as load_f:
    dataset_dicts= json.dumps(dataset_dicts,indent=2)
    load_f.write(dataset_dicts)

