import json
import numpy as np
import cv2
import os
import random
"""14_OCT_Habour_IMG_14_01___0.97897__10508__3868.jpg": {
    "image id": 9,
    "image size": {
      "height": 1010,
      "width": 2048
    }
  }
}"""
results_path="/root/data/gvision/my_merge/finalsubmission/final2/fullbody.json"
annos_path="/root/data/gvision/dataset/raw_data/image_annos/person_bbox_test.json"
image_folder_test="/root/data/gvision/dataset/raw_data/image_test"
with open(annos_path,'r') as load_f:
    test_dicts= json.load(load_f)
with open(results_path,'r') as load_f:
    results_dicts= json.load(load_f)
# def say(file_name,)
#     cropImg = image[int(302-150):int(302+150), int(278-150):int(278+150)]
# def say(iss):
#     file_name,dict_value=iss[0],iss[1]
#     print(file_name)
#     img=cv2.imread(os.path.join(test_image_path,file_name))
#     pre_output =predictor(img)
#     num_instance=0
#     cid=[0,0,0,0]
#     coco_list_result=[]
#     pre_instances=pre_output['instances']
#     if "instances" in pre_output and len(pre_instances)!=0:
#         coco_list_result,num_instance,cid=instances_to_coco_json(pre_instances.to(torch.device("cpu")),dict_value["image id"])
#     return coco_list_result,file_name,img,num_instance,cid,pre_instances 
# executor = ThreadPoolExecutor(max_workers=80)
# func_var = [[file_name,dict_value] for file_name,dict_value in dataset_test_dicts.items()]
# for coco_list_result,file_name,img,num_instance,cid,pre_instances in executor.map(say,func_var):
ext=3
imgfilters=["15_27","14_02","16_01","17_23","18_01"]
tempannos={}
if imgfilters:
    for imgfilter in imgfilters:
        tempannos.update({i:j for i,j in test_dicts.items() if imgfilter in i })
    test_dicts=tempannos
for file_name,images_dict in  test_dicts.items():
    print(os.path.join(image_folder_test,file_name))
    img=cv2.imread(os.path.join(image_folder_test,file_name))
    for result_dict in  results_dicts:
        if result_dict["image_id"]==images_dict["image id"]:
            xmin, ymin,w,h=int(result_dict["bbox"][0]),int(result_dict["bbox"][1]),int(result_dict["bbox"][2]),int(result_dict["bbox"][3])
            xmax,ymax=xmin+w,ymin+h
            left,up,right,down=xmin-ext,ymin-ext,xmax+ext,ymax+ext
            cropImg =img[up:down,left:right]
            print(f'/root/data/gvision/dataset/crop/{file_name[:-4].replace("/","_")}_{left}_{up}_{right-left}_{down-up}_{result_dict["score"]}_2.jpg')
            cv2.imwrite(f'/root/data/gvision/dataset/crop/test1/{file_name[:-4].replace("/","_")}_{left}_{up}_{right-left}_{down-up}_{result_dict["score"]}_2.jpg', cropImg)
