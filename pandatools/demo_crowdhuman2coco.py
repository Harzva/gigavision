import json
import os
from PIL import Image
def load_file(fpath):#fpath是具体的文件 ，作用：#str to list
    assert os.path.exists(fpath)  #assert() raise-if-not
    with open(fpath,'r') as fid:
        lines = fid.readlines()
    records = [json.loads(line.strip('\n')) for line in lines] #str to list
    return records
odgt_path="/root/data/gvision/CrowdDet-master/data/CrowdHuman/annotation_train.odgt"
# bbox_types=["vbox","fbox","hbox"]
bbox_types=["hbox"]
cat_lists=[{"supercategory": "none", "id": 1, "name": 'visible body'},
    {"supercategory": "none", "id": 2, "name": 'full body'},
    {"supercategory": "none", "id": 3, "name": 'head'}]
json_paths=["/root/data/gvision/CrowdDet-master/data/CrowdHuman/annotation_train_vbox.json",
    "/root/data/gvision/CrowdDet-master/data/CrowdHuman/annotation_train_fbox.json",
    "/root/data/gvision/CrowdDet-master/data/CrowdHuman/annotation_train_hbox.json"]
json_paths=[
    "/root/data/gvision/CrowdDet-master/data/CrowdHuman/annotation_train_hbox.json"]
# cat_list=[{"supercategory": "none", "id": 1, "name": 'visible body'}]
# cat_list=[{"supercategory": "none", "id": 2, "name": 'full body'}]
# cat_list=[{"supercategory": "none", "id": 3, "name": 'head'}]
def ceil_hw(x,ceil):
    if x>ceil:
        print("x>ceil",x,ceil)
        x=ceil
    else:
        x=x
    return x
def down_xy(x):
    if x<0:
        print("x<0",x)
        x=0
    else:
        x=x
    return x
def crowdhuman2coco(odgt_path,json_path,bbox_type,cat_list):  # 一个输入文件路径，一个输出文件路径
    print(f"json_path: {json_path}")
    records = load_file(odgt_path)  # 提取odgt文件数据
    # cat_list=[{"supercategory": "none", "id": 1, "name": 'visible body'},
    #     {"supercategory": "none", "id": 2, "name": 'full body'},
    #     {"supercategory": "none", "id": 3, "name": 'head'}]
    # 预处理
    json_dict ={"categories":[cat_list],"images": [], "annotations": [],"type":"instances"} # 定义一个字典，coco数据集标注格式
    START_B_BOX_ID = 1  # 设定框的起始ID
    image_id = 1  # 每个image的ID唯一，自己设定start，每次++
    bbox_id = START_B_BOX_ID
    image = {}  # 定义一个字典，记录image
    annotation = {}  # 记录annotation
    categories = {}  # 进行类别记录
    record_list = len(records)  # 获得record的长度，循环遍历所有数据。
    print(record_list)
    # 一行一行的处理。
    for i in range(record_list):
        file_name = records[i]['ID'] + '.jpg'  # 这里是字符串格式  eg.273278,600e5000db6370fb
        # image_id = int(records[i]['ID'].split(",")[0]) 这样会导致id唯一，要自己设定
        im = Image.open("/root/data/gvision/CrowdDet-master/data/CrowdHuman/Images/" + file_name)
        # 根据文件名，获取图片，这样可以获取到图片的宽高等信息。因为再odgt数据集里，没有宽高的字段信息。
        image = {'file_name': file_name, 'height': im.size[1], 'width': im.size[0],
                 'id': image_id}  # im.size[0]，im.size[1]分别是宽高
        json_dict['images'].append(image)  # 这一步完成一行数据到字典images的转换。

        gt_box = records[i]['gtboxes']
        gt_box_len = len(gt_box)  # 每一个字典gtboxes里，也有好几个记录，分别提取记录。
        for j in range(gt_box_len):
            category = gt_box[j]['tag']
            bbox = gt_box[j][bbox_type]  # 获得全身框
    
            x, y,w,h=bbox

            xmax,ymax=x+w,y+h
            if xmax<0 or ymax<0:
                print("max<0**************",xmax,ymax)
                print(bbox)
            x=down_xy(x)
            y=down_xy(y)
            w=ceil_hw(xmax,im.size[0])-x
            h=ceil_hw(ymax,im.size[1])-y

            # if x<0:
            #     print("--------x<0")            
            # if y<0:
            #     print("---------------y<0")
            # if w<0:
            #     print("------------w<0",xmax,im.size[0],x)      
            # if h<0:
            #     print("----------------h<0",ymax,im.size[1],y)  


            # 对ignore进行处理，ignore有时在key：extra里，有时在key：head_attr里。属于互斥的。
            ignore = 0  # 下面key中都没有ignore时，就设为0，据观察，都存在，只是存在哪个字典里，需要判断一下
            if "ignore" in gt_box[j]['head_attr']:
                ignore = gt_box[j]['head_attr']['ignore']
            if "ignore" in gt_box[j]['extra']:
                ignore = gt_box[j]['extra']['ignore']
            # 对字典annotation进行设值。
            # annotation = {'area': bbox[2] * bbox[3], 'iscrowd': ignore, 'image_id':  # 添加hbox、vbox字段。[x, y, x, (y + h), (x + w), (y + h), (x + w), y]
            #     image_id, 'bbox': bbox,'category_id':cat_list["id"], 'id': bbox_id, 'ignore': ignore, 'segmentation': [[x, y, x, (y + h), (x + w), (y + h), (x + w), y]]}
            annotation = {'area': bbox[2] * bbox[3], 'iscrowd': ignore, 'image_id':  # 添加hbox、vbox字段。[x, y, x, (y + h), (x + w), (y + h), (x + w), y]
                image_id, 'bbox': [x,y,w,h],'category_id':cat_list["id"], 'id': bbox_id,'segmentation': [[x, y, x, (y + h), (x + w), (y + h), (x + w), y]]}
            # area的值，暂且就是bbox的宽高相乘了，观察里面的数据，发现bbox[2]小、bbox[3]很大，刚好一个全身框的宽很小，高就很大。（猜测），不是的话，再自行修改
            # segmentation怎么处理？博主自己也不知道，找不到对应的数据，这里就暂且不处理。
            # hbox、vbox、ignore是添加上去的，以防有需要。
            json_dict['annotations'].append(annotation)

            bbox_id += 1  # 框ID ++
        image_id += 1 
    # 因为没有用到（不访问），就不需要给他们空间，也不需要去处理，字典是按key访问的，如果自己需要就自己添加上去就行
    json_fp = open(json_path, 'w')
    json_str = json.dumps(json_dict,indent=2)  # 写json文件。
    json_fp.write(json_str)
    json_fp.close()
for bbox_type,cat_list,json_path in zip(bbox_types,cat_lists,json_paths):
    crowdhuman2coco(odgt_path,json_path,bbox_type,cat_list)
