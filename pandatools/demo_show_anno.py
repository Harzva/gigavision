import os
from ImgSplit import ImgSplit
from PANDA import PANDA_IMAGE, PANDA_VIDEO
from ImgSplit import ImgSplit
from ResultMerge import DetResMerge
from panda_utils import DetRes2GT
basepath=r'/root/data/gvision/dataset/predict/s0.5_t0.8_141517'#主目录
detrespath="/root/data/gvision/dataset/predict/s0.5_t0.8_141517/resJSONS/merge_result.json"
outgtpath="/root/data/gvision/dataset/predict/s0.5_t0.8_141517/resJSONS/meresult2GT.json"
gtannopath="/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_annos/person_bbox_test_14_15_17.json"
# DetRes2GT(detrespath, outgtpath, gtannopath)

# os.chdir("/root/data/gvision/PANDA-Toolkit-master") 

# # annofile=r"vehicle_bbox_train.json"#文件  annofile=r"\vehicle_bbox_train.json" opth则 annopath E:\vehicle_bbox_train.json
# annofile=r"vtest.json"
# outpath =os.path.join(basepath,"split")
# print(outpath)
# if os.path.exists(outpath) !=True:
#     os.mkdir(outpath)
# outannofile =r"vehicle_bbox_train_split.json"
# split = ImgSplit(basepath,annofile, annomode, outpath, outannofile)
# split.splitdata(0.5)

# person_anno_file =r"person_s0.5_t0.9_train_02_split.json"
# basepath=r'/root/data/gvision/all-obj-split/center'#主目录
# person_anno_file =r"pv_split_obj.json"
# vehicle_anno_file = 'pv_split_obj.json'
# annomode='headbbox'
# annomode='person'
# example = PANDA_IMAGE(basepath, person_anno_file, annomode,addmode="image_train",savepath="my_annos_in_image_p" )
# 'person', 'vehicle', 'person&vehicle', 'headbbox' or 'headpoint'
annomode='crowd'
person_anno_file =r"person_bbox_train.json"
basepath=r'/root/data/gvision/dataset/raw_data'#主目录
vehicle_anno_file ='vehicle_bbox_train.json'

example = PANDA_IMAGE(basepath, person_anno_file, annomode=annomode,addmode="image_train",savepath="annos_in_image_crowd",showcate=False)
# # # '''1. show images'''vehicle
# # # example.showImgs(range=2)

# # '''2. show annotations'''"01_University_Canteen/IMG_01_02.jpg""10_Ceremony/IMG_10_21.jpg"
example.showAnns(imgrequest=None, range=1,imgfilters=["10_Ceremony/IMG_10_21.jpg"], shuffle=True,saveimg=True)

