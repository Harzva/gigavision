from __future__ import division
import os
import json
from PIL import Image
import xml.dom.minidom
from detectron2.structures import BoxMode
from detectron2.data.datasets import MyEncoder
import numpy as np
import cv2
import random
xmlfile = "/root/data/gvision/panda_tools/15_nanshan_park.xml"
ProcessedPath = '/root/data/gvision/dataset/train/bg'
if not os.path.exists(ProcessedPath):
    os.makedirs(ProcessedPath)
DomTree = xml.dom.minidom.parse(xmlfile)
annotation = DomTree.documentElement
filenamelist = annotation.getElementsByTagName('filename')
filenamelist=filenamelist[0].childNodes[0].data
print(filenamelist)
filenamefolder= annotation.getElementsByTagName('folder')
filenamefolder=filenamefolder[0].childNodes[0].data
print(filenamefolder)
filename = os.path.join("/root/data/gvision/dataset/raw_data/image_test",filenamefolder,filenamelist)
raw_img = cv2.imread(filename )
objectlist = annotation.getElementsByTagName('object')
set_w=1536
set_h=1536
dataset_dicts = []
for idx,objects in enumerate(objectlist):
    objs=[]
    record = {}
    namelist = objects.getElementsByTagName('name')
    # print('namelist:',namelist)
    name_i=0
    name_end=[]
    for objectname in namelist:
        name_end.append(namelist[name_i].childNodes[0].data)
        name_i+=1
        # print(name_end)
    bndbox = objects.getElementsByTagName('bndbox')
    box_i=0
    box_end=[]
    print(bndbox)
    for box in bndbox:
        x1_list = box.getElementsByTagName('xmin')
        x1 = int(x1_list[0].childNodes[0].data)
        y1_list = box.getElementsByTagName('ymin')
        y1 = int(y1_list[0].childNodes[0].data)
        x2_list = box.getElementsByTagName('xmax')
        x2 = int(x2_list[0].childNodes[0].data)
        y2_list = box.getElementsByTagName('ymax')
        y2 = int(y2_list[0].childNodes[0].data)
        w = x2 - x1
        h = y2 - y1
        print(os.path.join(ProcessedPath,filenamefolder+"_"+filenamelist[:-4]+f"_{x1}_{y1}_{w}_{h}.jpg"))
        record["file_name"] =os.path.join(ProcessedPath,filenamefolder+"_"+filenamelist[:-4]+f"_{x1}_{y1}_{w}_{h}.jpg")
        print(idx)
        record["image_id"] = idx
        record["height"] =h
        record["width"] =w
        obj = {
            "bbox": [],
            "bbox_mode":1,
            "segmentation": [[]],
            "category_id": 0,
            "iscrowd": 0
            }
        objs.append(obj)
        cv2.imwrite(os.path.join(ProcessedPath,filenamefolder+"_"+filenamelist[:-4]+f"_{x1}_{y1}_{w}_{h}.jpg"),raw_img[y1:y2,x1:x2,:] )
    record["annotations"] =objs
    dataset_dicts.append(record)
jsonString = json.dumps(dataset_dicts, cls=MyEncoder,indent=2)
with open(os.path.join(ProcessedPath,"bg.json"), "w") as f:
    f.write(jsonString)
# def _2json()


 
def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
        jsonString = json.dumps(dataset_dicts, cls=MyEncoder,indent=2)
        with open("/root/data/gvision/dataset/raw_data/ballon/val/cocovia_region_data.json", "w") as f:
            f.write(jsonString)
    return dataset_dicts
# get_balloon_dicts("/root/data/gvision/dataset/raw_data/ballon/val")