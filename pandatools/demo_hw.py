
import json
import cv2
import numpy as np
import os
# basepath="/root/data/gvision/dataset/train_all_annos/s0.3_t0.7_all"
# load_path="/root/data/gvision/dataset/output/my_pv_train/my_inference/coco_pv_inference_results.json"
# load_path_coco="/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_annos/person_bbox_test_141517_split.json"
"""  "14_OCT_Habour_IMG_14_01___0.5__1408__3072.jpg": {
    "image size": {
      "height": 2049,
      "width": 1025
    },
    "image id": 18

        {
      "file_name": "14_OCT_Habour_IMG_14_01___0.5__704__1024.jpg",
      "height": 2049,
      "width": 1025,
      "id": 9
    },
    
    """
# aaas=os.listdir("/root/data/rubzz/ruby/ruby_output3/split_train_person_panda_fafaxue_3category/img")
# for i in aaas:
#   print(os.path.join("/root/data/rubzz/ruby/ruby_output3/split_train_person_panda_fafaxue_3category/img",i))
#   im=cv2.imread(os.path.join("/root/data/rubzz/ruby/ruby_output3/split_train_person_panda_fafaxue_3category/img",i))
#   print(im.shape)
# load_path="/root/data/rubzz/ruby/ruby_output2/train_person_unsure_cell/train_person_unsure_cell_3category.json"
# print(im.shape)
# with open(load_path,'r') as load_f:
#     dataset_dicts = json.load(load_f)
# # print(dataset_dicts[0:100])
# with open(load_path_coco,'r') as load_path_coco:
#     coco_dataset_dicts = json.load(load_path_coco)
# for coco_images_dict in coco_dataset_dicts["images"]:
#     print(coco_images_dict["id"])
#     for images_dict in dataset_dicts["images"]:
#         if coco_images_dict["id"]==images_dict["id"]:
#             h,w=images_dict["height"],images_dict["width"]
#             coco_images_dict["height"]=h
#             coco_images_dict["width"]=w
# with open(output_path, 'w') as load_f:
#     COCO_dataset_dicts= json.dumps(coco_dataset_dicts,indent=2)
#     load_f.write(COCO_dataset_dicts)


# with open("/root/data/gvision/dataset/train_all_annos/s0.3_t0.7_all/image_annos/coco_vehicle_train_hwnoi.json",'r') as load_f:
#     dataset_dicts = json.load(load_f)
# print(len(dataset_dicts["annotations"]))
# # print(dataset_dicts)#1,2
# print("type",type(dataset_dicts))
"""
450558 coco_person_train_hwnoi.json
483276 coco_pv_train_bbox_hwnoi.json coco_pv_train_hwnoi.json
32718 coco_vehicle_train_bbox_hwnoi.json coco_vehicle_train_hwnoi
"""
def coco_hw(load_path_coco,save_path):
    with open(load_path_coco,'r') as load_path_coco:
        coco_dataset_dicts = json.load(load_path_coco)
    f=open(save_path,'w') 
    for images_dict in coco_dataset_dicts["images"]:
        imagename=images_dict["file_name"]
        print(imagename)
        height,width=cv2.imread(os.path.join("/root/data/rubzz/ruby/ruby_output3/split_train_person_panda_fafaxue_3category/img",imagename)).shape[0:2]
        images_dict['height'] =height
        images_dict['width'] = width
    f.write(json.dumps(coco_dataset_dicts,indent=2))

coco_hw(load_path_coco="/root/data/rubzz/ruby/ruby_output3/split_train_person_panda_fafaxue_3category/split_train_person_panda_fafaxue_3category.json",
        save_path="/root/data/rubzz/ruby/ruby_output3/split_train_person_panda_fafaxue_3category/split_train_person_panda_fafaxue_3category_hw.json")
# class MyEncoder(json.JSONEncoder):
#     def default(self, obj):
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         else:
#             return super(MyEncoder, self).default(obj)
# load_path_coco="/root/data/gvision/dataset/d2_output/my_pv_mask/metrics.json"
# # target="/root/data/gvision/dataset/d2_output/my_pv_mask/my_predict/predict_all_0500.json"
# with open(load_path_coco,'r') as load_path_coco:
#     result_list= json.load(load_path_coco)
# print(result_list)

# f=open(target,'w') 
# f.write(json.dumps(result_list[0:500],cls=MyEncoder))

# a=[]
# for result_dict in result_list:
#     result_dict.pop('segmentation')
#     a.append(result_dict)
# f=open(target,'w') 

# f.write(json.dumps(a,cls=MyEncoder))
# a=np.load("/root/data/gvision/dataset/d2_output/my_pv_mask/model_final_indexedresults.npy",allow_pickle=True)
# print(len(a))
# print(os.path.getsize("/root/data/gvision/dataset/d2_output/my_pv_mask/model_final_indexedresults.npy"))


# load_path_coco="/root/data/gvision/dataset/d2_output/my_pv_center_mask/metrics_18499.json"
# import json
# data = []
# a=[0,0]
# f=open(load_path_coco, 'r', encoding="utf-8")
# # 读取所有行 每行会是一个字符串
# loss=10
# for line,j in enumerate(f.readlines()):
#     j = json.loads(j)
#     if j["total_loss"]<loss:
#         loss=j["total_loss"]
#         a[0]=line+1
# #         print(line)
# # print(loss)
# a[1]=loss
# print(a)

# img=cv2.imread("/root/data/gvision/panda_tools/panda-imgae-test.png")
# img18=img[0:238,0:423,:]   
# img14=img[0:238,423:423*2,:]   
# img17=img[0:238,423*2:423*3,:]
# print(img14.shape,img14.shape,img14.shape)
# cv2.imwrite("/root/data/gvision/panda_tools/test18.png",img18)
# cv2.imwrite("/root/data/gvision/panda_tools/test14.png",img14)
# cv2.imwrite("/root/data/gvision/panda_tools/test17.png",img17)
# img18=cv2.resize(img18,(423*50,238*50),interpolation=cv2.INTER_CUBIC)
# img14=cv2.resize(img14,(423*50,238*50),interpolation=cv2.INTER_CUBIC)
# img17=cv2.resize(img17,(423*50,238*50),interpolation=cv2.INTER_CUBIC)
# cv2.imwrite("/root/data/gvision/panda_tools/test_18.png",img18)
# cv2.imwrite("/root/data/gvision/panda_tools/test_14.png",img14,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])
# cv2.imwrite("/root/data/gvision/panda_tools/test_17.png",img17,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])
# import numpy as np
# a=[[1,2,3,4],[1,2,3,4],[1,2,3,4]]
# b=[1]
# c=[b,b,b,b]
# [old+new for old,new in zip(a,c)]
# print([old+new for old,new in zip(a,c)])
# print([1176.27, 637.9455, 1412.9817, 1139.9287] +[0.7856537])