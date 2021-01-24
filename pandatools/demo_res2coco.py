import json
import cv2
detrespath = "/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_annos/person_bbox_test_141517_split.json"
det_P_result_path="/root/data/gvision/dataset/d2_output/my_pv_mask/my_predict/pre_result_all.json"
tgtfile = "/root/data/gvision/dataset/d2_output/my_pv_mask/my_predict/coco_pre_result_all.json"

with open(detrespath, 'r') as load_f:
    person_anno_dict = json.load(load_f)
with open(det_P_result_path, 'r') as load_f:
    res_anno_dict = json.load(load_f)
# with open(det_V_result_path, 'r') as load_f:
#     vehicle_anno_dict = json.load(load_f)
attrDict = dict()
attrDict["categories"] = [
        {"supercategory": "none", "id": 1, "name": 'visible body'},
        {"supercategory": "none", "id": 2, "name": 'full body'},
        {"supercategory": "none", "id": 3, "name": 'head'},
        {"supercategory": "none", "id": 4, "name": 'vehicle'}
    ]

images = list()
annotations=list()

###########images#########
for (imagename, imagedict) in person_anno_dict.items():
    image = dict()
    image['file_name'] = imagename
    # height,width=cv2.imread(os.path.join("/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_test",imagename)).shape[0:2]
    # image['height'] =height,
    # image['width'] = width
    image['height'] = imagedict['image size']['height']
    image['width'] = imagedict['image size']['width']
    image['id'] = imagedict['image id']
    images.append(image)
####person————annonatation
objid = 1
for objdict in res_anno_dict:
    cate = objdict['category_id']
    annotation = dict()
    annotation["image_id"] = objdict["image_id"]
    annotation["iscrowd"] = 0
    annotation["bbox"] = objdict["bbox"]
    x, y, w, h=objdict["bbox"]
    annotation["area"] = float(w * h)
    annotation["category_id"] = cate
    annotation["id"] = objid
    objid += 1
    annotation["segmentation"] = [[x, y, x, (y + h), (x + w), (y + h), (x + w), y]]
    annotations.append(annotation)


# for objdict in vehicle_anno_dict:
#     cate = objdict['category_id']
#     if cate == 4:
#         annotation = dict()
#         annotation["image_id"] = imgid
#         annotation["iscrowd"] = 0
#         annotation["bbox"] = objdict["bbox"]
#         x, y, w, h = objdict["bbox"]
#         annotation["area"] = float(w * h)
#         annotation["category_id"] = cate
#         annotation["id"] = objid
#         objid += 1
#
#         annotation["segmentation"] = [[x, y, x, (y + h), (x + w), (y + h), (x + w), y]]
#         annotations.append(annotation)


attrDict["images"] = images
attrDict["annotations"] = annotations
attrDict["type"] = "instances"

# print attrDict
jsonString = json.dumps(attrDict, indent=2)
with open(tgtfile, "w") as f:
    f.write(jsonString)