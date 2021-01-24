import cv2
import json
import os
import numpy as np


"""
imgfilters=["14_OCT_Habour","15_Nanshani_Park","16_Primary_School","17_New_Zhongguan","18_Xili_Street"]
"""
def del_mean(npyname='/root/data/gvision/dataset/raw_data/image_annos/mean_14.npy',
            imgfilters=["14_OCT_Habour"]):
    basename="/root/data/gvision/dataset/raw_data"
    test_json_path="/root/data/gvision/dataset/raw_data/image_annos/vehicle_bbox_test.json"
    test_dicts=json.load(open(test_json_path,"r"))

    for imgfilter in imgfilters:
        print(f"scene: {imgfilter} picture sum")
        height=list(test_dicts.values())[1]["image size"]["height"]
        width=list(test_dicts.values())[1]["image size"]["width"]
        imgs=np.zeros((height,width,3))
        if not os.path.exists(npyname):
            print("npz not exists --> save")
            for j,(file_name,image_dict) in enumerate(test_dicts.items()):
                if imgfilter in file_name:
                    print(file_name)
                    img=cv2.imread(os.path.join(basename,"image_test",file_name))
                    imgs=imgs+img
            imgs=np.save(npyname,imgs)#no exists --> save and not load
            print(imgs)
        else:
            print("npz exists and drict load")
            imgs=np.load(npyname,allow_pickle=True)# save and load
            print(type(imgs))
            print(imgs.shape)
            print(imgs[0:100,0:100,1])
              
        print("del mean-------------start")
        for j,(file_name,image_dict) in enumerate(test_dicts.items()):
            if imgfilter in file_name:
                img=cv2.imread(os.path.join(basename,"image_test",file_name))
                img=img-imgs/30
                img=cv2.resize(img,(int(img.shape[1]*0.1),int(img.shape[0]*0.1)))
                print(os.path.join(basename,"image_test_del_mean"))
                os.makedirs(os.path.join(basename,"image_test_del_mean"),exist_ok=True)
                cv2.imwrite(os.path.join(basename,"image_test_del_mean","del_mean_{}".format(file_name.replace("/","_"))),img,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])
if __name__ == "__main__":
    del_mean()
            

