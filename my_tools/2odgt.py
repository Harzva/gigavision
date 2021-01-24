import json
from tqdm import tqdm
import random

category_id = 1  # 1:visible body 2:full body 3:head 4:vehicle
json_path = "/root/data/gvision/dataset/train/ruby_output/dyy_split_visiblebody.json"
odgt_path = "/root/data/gvision/CrowdDet-master/data/CrowdHuman/annotation_tvp_fusion.odgt"
odgt_train_append_path = "/root/data/gvision/CrowdDet-master/data/CrowdHuman/annotation_train.odgt"
odgt_test_append_path = "/root/data/gvision/CrowdDet-master/data/CrowdHuman/annotation_val.odgt"
def coco2crowdhuman(odgt_path,json_path,odgt_train_append_path,odgt_test_append_path):
    with open(json_path, 'r') as load_f:
        json_dict = json.load(load_f)
    images = json_dict["images"]
    annos = json_dict["annotations"]
    odgt = list()
    pbar = tqdm(total=len(images), ncols=50)
    for image in images:
        odgt_one = dict()
        odgt_one["ID"] = image["file_name"][0:-4]
        image_id = image["id"]
        gtboxes = list()
        box_id = 0
        for anno in annos:
            if anno["image_id"] == image_id and anno["category_id"] == category_id:
                gtbox = dict()

                gtbox["tag"] = "person"
                gtbox["hbox"] = [0, 0, 0, 0]
                gtbox["head_attr"] = {"ignore": 0, "occ": 1, "unsure": 0}

                # visible body
                gtbox["fbox"] = [0, 0, 0, 0]
                gtbox["vbox"] = anno["bbox"]

                # full body
                # gtbox["fbox"] = anno["bbox"]
                # gtbox["vbox"] = [0, 0, 0, 0]

                extra = dict()
                extra["box_id"] = box_id
                box_id += 1
                extra["occ"] = 1
                gtbox["extra"] = extra

                gtboxes.append(gtbox)

        odgt_one["gtboxes"] = gtboxes
        odgt.append(odgt_one)
        pbar.update(1)
    pbar.close()
    odgtvp=odgt
    odgttp=odgt

    with open(odgt_train_append_path, 'r') as load_f:
        for line in load_f.readlines():
            odgt.append(line[0:-1])
            odgttp.append(line[0:-1])

    with open(odgt_test_append_path, 'r') as load_f:
        for line in load_f.readlines():
            odgt.append(line[0:-1])
            odgtvp.append(line[0:-1])

    print(odgt[1])
    random.shuffle(odgt)
    print(odgt[1])
    random.shuffle(odgtvp)
    random.shuffle(odgttp)



    with open(odgt_path, "w") as f:
        for line in odgt:
            f.write(str(line) + '\n')
    
    with open("/root/data/gvision/CrowdDet-master/data/CrowdHuman/annotation_vp_fusion.odgt", "w") as f:
        for line in odgtvp:
            f.write(str(line) + '\n')
    
    with open("/root/data/gvision/CrowdDet-master/data/CrowdHuman/annotation_tp_fusion.odgt", "w") as f:
        for line in odgttp:
            f.write(str(line) + '\n')

coco2crowdhuman(odgt_path,json_path,odgt_train_append_path,odgt_test_append_path)