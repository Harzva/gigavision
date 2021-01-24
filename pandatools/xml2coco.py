# -*- coding: utf-8 -*-
# @Time    : 2019/8/27 10：48
# @Author  :Rock
# @File    : voc2coco.py
# just for object detection
import xml.etree.ElementTree as ET
import os
import cv2
import numpy
import random
import json
ass= [
    {
      "supercategory": "none",
      "id": 1,
      "name": "small car"
    },
    {
      "supercategory": "none",
      "id": 2,
      "name": "midsize car"
    },
    {
      "supercategory": "none",
      "id": 3,
      "name": "large car"
    },
    {
      "supercategory": "none",
      "id": 4,
      "name": "bicycle"
    },
    {
      "supercategory": "none",
      "id": 5,
      "name": "baby carriage"
    },
    {
      "supercategory": "none",
      "id": 6,
      "name": "motorcycle"
    },
    {
      "supercategory": "none",
      "id": 7,
      "name": "tricycle"
    },
    {
      "supercategory": "none",
      "id": 8,
      "name": "electric car"
    }
  ]
#   ['bicycle', 'car', 'motorcycle', 'bus']=[4,1,6,3]

# -*- coding: utf-8 -*-
# @Time    : 2019/8/27 10：48
# @Author  :Rock
# @File    : voc2coco.py
# just for object detection

coco = dict()
coco['categories'] = []
coco['images'] = []
coco['annotations'] = []
coco['type'] = 'instances'


category_set = dict()
image_set = set()

category_item_id = 0
image_id = 0
annotation_id = 0


def addCatItem(name):
    global category_item_id
    category_item = dict()
    category_item['supercategory'] = 'none'
    category_item_id += 1
    category_item['id'] = category_item_id
    category_item['name'] = name
    coco['categories'].append(category_item)
    category_set[name] = category_item_id
    return category_item_id


def addImgItem(file_name, size):
    global image_id
    if file_name is None:
        raise Exception('Could not find filename tag in xml file.')
    if size['width'] is None:
        raise Exception('Could not find width tag in xml file.')
    if size['height'] is None:
        raise Exception('Could not find height tag in xml file.')
    img_id = "%04d" % image_id
    image_id += 1
    image_item = dict()
    # image_item['id'] = int(img_id)
    image_item['file_name'] = file_name
    image_item['width'] = size['width']
    image_item['height'] = size['height']
    image_item['id'] = image_id
    coco['images'].append(image_item)
    image_set.add(file_name)
    return image_id


def addAnnoItem(object_name, image_id, category_id, bbox):
    global annotation_id
    annotation_item = dict()
    annotation_item['segmentation'] = []
    seg = []
    # bbox[] is x,y,w,h
    # left_top
    seg.append(bbox[0])
    seg.append(bbox[1])
    # left_bottom
    seg.append(bbox[0])
    seg.append(bbox[1] + bbox[3])
    # right_bottom
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1] + bbox[3])
    # right_top
    seg.append(bbox[0] + bbox[2])
    seg.append(bbox[1])

    annotation_item['segmentation'].append(seg)

    annotation_item['area'] = bbox[2] * bbox[3]
    annotation_item['iscrowd'] = 0
    annotation_item['ignore'] = 0
    annotation_item['image_id'] = image_id
    annotation_item['bbox'] = bbox
    annotation_item['category_id'] = category_id
    annotation_id += 1
    annotation_item['id'] = annotation_id
    coco['annotations'].append(annotation_item)


def parseXmlFiles(xml_path):
    for f in os.listdir(xml_path):
        if not f.endswith('.xml'):
            continue

        bndbox = dict()
        size = dict()
        current_image_id = None
        current_category_id = None
        file_name = None
        size['width'] = None
        size['height'] = None
        size['depth'] = None

        xml_file = os.path.join(xml_path, f)
        # print(xml_file)

        tree = ET.parse(xml_file)
        root = tree.getroot()
        if root.tag != 'annotation':
            raise Exception('pascal voc xml root element should be annotation, rather than {}'.format(root.tag))

        # elem is <folder>, <filename>, <size>, <object>
        for elem in root:
            current_parent = elem.tag
            current_sub = None
            object_name = None

            if elem.tag == 'folder':
                folder_name=elem.text
                # continue

            if elem.tag == 'filename':
                file_name = elem.text
                if file_name in category_set:
                    raise Exception('file_name duplicated')

            # add img item only after parse <size> tag
            elif current_image_id is None and file_name is not None and size['width'] is not None:
                print(file_name)
                if file_name not in image_set and folder_name=="JPEGImages":
                    print(folder_name,file_name)
                    file_name=folder_name+"/"+file_name+".jpg"
                    current_image_id = addImgItem(file_name, size)
                    # print('add image with {} and {}'.format(file_name, size))
                else:
                    raise Exception('duplicated image: {}'.format(file_name))
                    # subelem is <width>, <height>, <depth>, <name>, <bndbox>
            for subelem in elem:
                bndbox['xmin'] = None
                bndbox['xmax'] = None
                bndbox['ymin'] = None
                bndbox['ymax'] = None

                current_sub = subelem.tag
                if current_parent == 'object' and subelem.tag == 'name':
                    object_name = subelem.text
                    if object_name not in category_set:
                        current_category_id = addCatItem(object_name)
                    else:
                        current_category_id = category_set[object_name]

                elif current_parent == 'size':
                    if size[subelem.tag] is not None:
                        raise Exception('xml structure broken at size tag.')
                    size[subelem.tag] = int(subelem.text)

                # option is <xmin>, <ymin>, <xmax>, <ymax>, when subelem is <bndbox>
                for option in subelem:
                    if current_sub == 'bndbox':
                        if bndbox[option.tag] is not None:
                            raise Exception('xml structure corrupted at bndbox tag.')
                        bndbox[option.tag] = int(option.text)

                # only after parse the <object> tag
                if bndbox['xmin'] is not None:
                    if object_name is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_image_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    if current_category_id is None:
                        raise Exception('xml structure broken at bndbox tag')
                    bbox = []
                    # x
                    bbox.append(bndbox['xmin'])
                    # y
                    bbox.append(bndbox['ymin'])
                    # w
                    bbox.append(bndbox['xmax'] - bndbox['xmin'])
                    # h
                    bbox.append(bndbox['ymax'] - bndbox['ymin'])
                    # print('add annotation with {},{},{},{}'.format(object_name, current_image_id, current_category_id,
                    #                                                bbox))
                    addAnnoItem(object_name, current_image_id, current_category_id, bbox)

def coco2my_coco(respath,coco_json):

    """
    
    {"categories": [{"supercategory": "none","id": 1,
      "name": "bicycle"
    },
    {
      "supercategory": "none",
      "id": 2,
      "name": "bus"
    },
    {
      "supercategory": "none",
      "id": 3,
      "name": "car"
    },
    {
      "supercategory": "none",
      "id": 4,
      "name": "motorcycle"
    }
     [{'supercategory': 'none', 'id': 1, 'name': 'small car'}, {'supercategory': 'none', 'id': 2, 'name': 'midsize car'}, {'supercategory': 'none', 'id': 3, 'name': 'large car'}, {'supercategory': 'none', 'id': 4, 'name': 'bicycle'}, {'supercategory': 'none', 'id': 5, 'name': 'baby carriage'}, {'supercategory': 'none', 'id': 6, 'name': 'motorcycle'}, {'supercategory': 'none', 'id': 7, 'name': 'tricycle'}, {'supercategory': 'none', 'id': 8, 'name': 'electric car'}]
    """
    with open(respath, 'r') as load_f:
        coco_dicts= json.load(load_f)
        print("orgin cat",coco_dicts['categories'])
        # coco_dicts['categories']=ass
    coco_dicts['categories']=[{'id': 4, 'name': 'car', 'supercategory': 'None'}]
    for images_dict in  coco_dicts["annotations"]:
        images_dict['category_id']=4
        # if images_dict['category_id']==1:
        #     images_dict['category_id']=4

        # elif images_dict['category_id']==2:
        #     images_dict['category_id']=3

        # elif images_dict['category_id']==3:
        #     images_dict['category_id']=1

        # elif images_dict['category_id']==4:
        #     images_dict['category_id']=2

        # elif images_dict['category_id']==5:
        #     images_dict['category_id']=6
        # else:
        #     print("erro")
    json.dump(coco_dicts, open(coco_json, 'w'),indent=2)
def _sscoco(voc,cat):
    a=[]
    # print(cat)
    for i in voc:
        for add in cat:
            if i==add["id"]:
                # print(i)
                # print(add["name"])
                a.append(add["name"])


        # if i==4:
        #     a.append("bicycle")

        # if i==3:
        #     a.append("bus")

        # if i==1:
        #     a.append("small car")

        # if i==3:
        #     a.append("motorcycle")
    return a 
def zz():
    json_file=r'/root/data/gvision/dataset/VOC2007car/my_VOC2007car.json'
    tgtfile='/root/data/gvision/dataset/VOC2007car/my_end_VOC2007car.json'
    attrDict = dict()
    attrDict["categories"] = [
            {"supercategory": "none", "id": 4, "name": 'car'}
        ]

    images = list()
    annotations=list()
    with open(json_file,'r') as load_f:
        coco_dicts= json.load(load_f)
    temp_list=[]
    for j,images_dict in  enumerate(coco_dicts["images"]):
        if int(images_dict["file_name"][-7:-4])<975:
            image = dict()
            image['file_name'] = images_dict["file_name"]
            # height,width=cv2.imread(os.path.join("/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_test",imagename)).shape[0:2]
            # image['height'] =height,
            # image['width'] = width
            image['height'] = images_dict['height']
            image['width'] = images_dict['width']
            image['id'] = images_dict['id']
            temp_list.append(image['id'] )
            images.append(image)
    objid = 1
    for objdict in coco_dicts["annotations"]:
        if objdict["image_id"] in temp_list:
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
    attrDict["images"] = images
    attrDict["annotations"] = annotations
    attrDict["type"] = "instances"
    # print attrDict
    jsonString = json.dumps(attrDict, indent=2)
    with open(tgtfile, "w") as f:
        f.write(jsonString)

def visual(parent_path,json_file ):
    with open(json_file,'r') as load_f:
        coco_dicts= json.load(load_f)
    for j,images_dict in  enumerate(coco_dicts["images"]):
        file_name=images_dict["file_name"]
        print(file_name)
        # print("{}\t{}-------------------{}".format(file_name,j,10))
        image_id=images_dict["id"]
        img=cv2.imread(os.path.join(parent_path,file_name))
        result_list=[x["bbox"] for x in coco_dicts["annotations"] if x["image_id"]==images_dict["id"]]
        result_cat=[x["category_id"] for x in coco_dicts["annotations"] if x["image_id"]==images_dict["id"]]
        names=_sscoco(result_cat,coco_dicts["categories"])
        assert len(result_list)==len(result_cat) and len(names)==len(result_cat),"no len"
        for result_dict,cat,name in  zip(result_list,result_cat,names):
            # print(result_dict)
            xmin, ymin, w , h = result_dict
            xmax,ymax=xmin+w,ymin+h
            xmin, ymin,xmax,ymax=int(xmin),int(ymin),int(xmax),int(ymax)
            img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (138,255,0), 2,lineType=8)
            cv2.putText(img, '{}|  {}'.format(cat,name), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 1, (255,46,46), 1)
        # print(os.path.join("/root/data/gvision/dataset/coco/vehicle","truckdraw", str(image_id)+ '.jpg'))
        # cv2.imwrite(os.path.join("/root/data/gvision/dataset/coco/vehicle","truckdraw", str(image_id) + '.jpg'),img)
        # print(type(img))
        # print(img)
        assert type(img)==numpy.ndarray,"no len"
        print(os.path.join("/root/data/gvision/dataset/VOC2007car/visual", str(image_id)+ 'cocomy_else.jpg'))
        cv2.imwrite(os.path.join("/root/data/gvision/dataset/VOC2007car/visual", str(image_id) + 'cocomy_else.jpg'),img)
if __name__ == '__main__':
	#修改这里的两个地址，一个是xml文件的父目录；一个是生成的json文件的绝对路径
    xml_path = r'/root/data/gvision/dataset/VOC2007car/Annotations/'
    json_file = r'/root/data/gvision/dataset/VOC2007car/VOC2007car.json'
    parseXmlFiles(xml_path)
    json.dump(coco, open(json_file, 'w'),indent=2)

    
    coco2my_coco(json_file,coco_json=r'/root/data/gvision/dataset/VOC2007car/my_VOC2007car.json')

    # visual(parent_path = '/root/data/gvision/dataset/UAC/UAC_test/test',
    # json_file = '/root/data/gvision/dataset/UAC/UAC_test/test/my_annotations.json')
    # visual(parent_path = '/root/data/gvision/dataset/train/ruby_output/vehicle',
    # json_file = '/root/data/gvision/dataset/train/ruby_output/fw_split_vehicle_else.json')
    # zz()
    visual(parent_path = '/root/data/gvision/dataset/VOC2007car',
    json_file = r'/root/data/gvision/dataset/VOC2007car/my_end_VOC2007car.json')
	

    
    # visual(parent_path = '/root/data/gvision/dataset/coco/train2017',
    # json_file = '/root/data/gvision/dataset/coco/annotations/instances_train2017.json')


