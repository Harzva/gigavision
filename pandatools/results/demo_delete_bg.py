import json

input_dir = "D:/giga/Professional/image_annos/person_bbox_train.json"
output_dir = "C:/Users/Du/detectron2/demo/json/person_bbox_train_dbg.json"

scene_name = ["01_University_Canteen","02_Xili_Crossroad","03_Train_Station Square",
              "04_Grant_Hall","05_University_Gate","06_University_Campus","07_East_Gate",
              "08_Dongmen_Street","09_Electronic_Market","10_Ceremony","11_Shenzhen_Library",
              "12_Basketball_Court","13_University_Playground",]

with open(input_dir,'r') as load_f:
    load_dict = json.load(load_f)

def kuang2(min_y,tl_y, image_height):
    y0 = int(min_y*image_height)
    if y0 > image_height:
        y0 = image_height
    if y0 < tl_y:
        tl_y = y0
    return tl_y

for k in range(len(scene_name)):
    br_x = 0
    br_y = 0
    flag = 1
    new_dict = {}
    for key,value in load_dict.items():
        if scene_name[k] in key:
            new_dict[key] = value
            if flag ==1:
                image_width = value["image size"]["width"]
                image_height = value["image size"]["height"]
                tl_x = image_width
                tl_y = image_height
                flag = 0
            for i in range(len(value["objects list"])):
                objectList = value["objects list"][i]
                if objectList["category"] == "person":
                    min_y = objectList["rects"]["full body"]["tl"]["y"]
                    tl_y = kuang2(min_y,tl_y, image_height)

    print(tl_y)

    for key,value in new_dict.items():
        value["image size"]["height"] = value["image size"]["height"] - tl_y
        for i in range(len(value["objects list"])):
            if value["objects list"][i]["category"] == "person":
                value["objects list"][i]["rects"]["full body"]["tl"]["y"] = value["objects list"][i]["rects"]["full body"]["tl"]["y"] - tl_y/image_height
                value["objects list"][i]["rects"]["full body"]["br"]["y"] = value["objects list"][i]["rects"]["full body"]["br"]["y"] - tl_y/image_height
        load_dict[key] = value

with open(output_dir,"w") as f:
    json.dump(load_dict, f)


import json

input_dir = "D:/giga/Professional/image_annos/coco_pv_train_hwnoi.json"
output_dir = "C:/Users/Du/detectron2/demo/json/coco_person&vehicle_delBg.json"

del_h = [4473,4233,3789,6033,7267,3614,3535,1996,4434,3675,6132,3011,3418]
file_name = [[]*3]      # file_name   id    tl_y

with open(input_dir,'r') as load_f:
    load_dict = json.load(load_f)

categories = load_dict["categories"]
images = load_dict["images"]
annotations = load_dict["annotations"]
type = load_dict["type"]

images_new = list()
for i in range(len(images)):
    img = images[i]
    id = img["id"]
    count = int((id-1)/30)
    # print(count)
    img["height"] -= del_h[count]
    images_new.append(img)

annotations_new = list()
for j in range(len(annotations)):
    anno = annotations[j]
    image_id = anno["image_id"]
    count = int((image_id-1)/30)
    anno["bbox"][1] -= del_h[count]
    annotations_new.append(anno)

attrDict = {}
attrDict["categories"] = categories
attrDict["images"] = images_new
attrDict["annotations"] = annotations_new
attrDict["type"] = "instances"

# print attrDict
jsonString = json.dumps(attrDict, indent=2)
with open(output_dir, "w") as f:
    f.write(jsonString)