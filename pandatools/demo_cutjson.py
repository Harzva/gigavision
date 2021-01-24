import os
from panda_utils import DetRes2GT,generate_coco_anno,generate_res_from_gt,generate_coco_anno_vehicles,generate_coco_anno_persons,persons_challenge_GT2coco,result2panda
from ImgSplit import ImgSplit,ImgSplit_withoutsave,ImgSplit_test#imgsplit withoutsave
scene_name = ["01_University_Canteen","02_Xili_Crossroad","03_Train_Station Square",
              "04_Grant_Hall","05_University_Gate","06_University_Campus","07_East_Gate",
              "08_Dongmen_Street","09_Electronic_Market","10_Ceremony","11_Shenzhen_Library",
              "12_Basketball_Court","13_University_Playground"]

# basepath=r'/root/data/gvision/dataset/raw_data'#主目录
# annomode="vehicle"#模式
# addmode="image_train"
# annofile=r"vehicle_bbox_train.json"#文件  annofile=r"\vehicle_bbox_train.json" opth则 annopath E:\vehicle_bbox_train.json
# annofile=r"vtest.json"
# print(addmode,annomode)
# outpath =r"/root/data/gvision/dataset/train_all_annos/s0.3_t0.3_all_v"
# print(outpath)
# if os.path.exists(outpath) !=True:
#     os.mkdir(outpath)
# outannofile =r"vehicle_bbox_train.json"
# split = ImgSplit(basepath,annofile, annomode, outpath, outannofile,addmode,saveimage=True,gap_w=100,gap_h=512,subwidth=2048,subheight=2048,thresh=0.2)
# split.splitdata(0.3,imgfilters=["02_Xili_Crossroad"])
# split.splitdata(0.3,imgfilters=[])

# personsrcfile="/root/data/gvision/dataset/train_all_annos/s0.3_t0.3_all_v/image_annos/person_bbox_train.json"
# tgtfile="/root/data/gvision/dataset/train_all_annos/s0.3_t0.3_all_v/image_annos/COCO_person_train_hwnoi.json"
scale=1
# generate_coco_anno_persons(basepath,personsrcfile, tgtfile,scale)
print("---------------------------------------------vehicles")
# vehiclesrcfile=os.path.join(outpath, 'image_annos',outannofile )
# tgtfile=os.path.join(outpath, 'image_annos',"coco_vehicle_train_hwnoi.json")
# generate_coco_anno_vehicles(outpath,vehiclesrcfile,tgtfile,scale)

print("---------------------------------------------persons&vehicles")
# tgtfile="/root/data/gvision/dataset/train_all_annos/s0.3_t0.3_all_v//image_annos/coco_pv_train_hwnoi.json"


# generate_coco_anno(basepath,personsrcfile,vehiclesrcfile,tgtfile,scale)
# basepath=r'/root/data/gvision/dataset/'#主目录
# annomode="person"#模式
# annofile=r"person_bbox_train.json"#文件  annofile=r"\vehicle_bbox_train.json" opth则 annopath E:\vehicle_bbox_train.json
# # annofile=r"vtest.json"
# outpath =r"/root/data/gvision/dataset/image_train_person_split"
# print(outpath)
# if os.path.exists(outpath) !=True:
#     os.mkdir(outpath)
# outannofile =r"person_bbox_train_split.json"
# split = ImgSplit(basepath,annofile, annomode, outpath, outannofile)
# split.splitdata(0.5)
# annomode="vehicle"#模式
# annofile=r"vehicle_bbox_train.json"#文件  annofile=r"\vehicle_bbox_train.json" opth则 annopath E:\vehicle_bbox_train.json
# # annofile=r"vtest.json"
# outpath =r"/root/data/gvision/dataset/image_train_vehicle_split"
# print(outpath)
# if os.path.exists(outpath) !=True:
#     os.mkdir(outpath)
# outannofile =r"vehicle_bbox_train_split.json.json"
# split = ImgSplit(basepath,annofile, annomode, outpath, outannofile)
# split.splitdata(0.5)

"""
test_cut
"""
basepath=r'/root/data/gvision/dataset/raw_data'#主目录
annomode="person"#模式
annofile=r"person_bbox_test.json"#文件  annofile=r"\vehicle_bbox_train.json" opth则 annopath E:\vehicle_bbox_train.json
# annofile=r"vtest.json"
outpath =r"/root/data/gvision/dataset/predict/16_01"
addmode="image_test"
print(outpath)
if os.path.exists(outpath) !=True:
    os.mkdir(outpath)
outannofile =r"s0.5_16_01_split_test.json"
split = ImgSplit_test(basepath,annofile, annomode,outpath,outannofile,addmode)
# split.splitdata(0.5,imgfilters=["14_OCT_Habour/IMG_14_01.jpg"])
split.splitdata(0.5,imgfilters=["16_Primary_School/IMG_16_01.jpg"])
"""
person_s0.5_t0.9_02
"""
# basepath=r'/root/data/gvision/dataset/scene_02'#主目录
# annomode="person"#模式
# annofile=r"person_bbox_train_02.json"#文件  annofile=r"\vehicle_bbox_train.json" opth则 annopath E:\vehicle_bbox_train.json
# # annofile=r"vtest.json"
# outpath =r"/root/data/gvision/dataset/split/person_s0.5_t0.9_02"
# print(outpath)
# if os.path.exists(outpath) !=True:
#     os.mkdir(outpath)
# outannofile =r"person_s0.5_t0.9_train_02_split.json"
# split = ImgSplit(basepath,annofile, annomode, outpath, outannofile)
# split.splitdata(0.5)
"""
person_s1_t0.3_02
"""
# basepath=r'/root/data/gvision/dataset/scene_02'#主目录
# annomode="person"#模式
# annofile=r"person_bbox_train_02.json"#文件  annofile=r"\vehicle_bbox_train.json" opth则 annopath E:\vehicle_bbox_train.json
# # annofile=r"vtest.json"
# outpath =r"/root/data/gvision/dataset/split/person_s1_t0.3_02"
# print(outpath)
# if os.path.exists(outpath) !=True:
#     os.mkdir(outpath)
# outannofile =r"person_bbox_train_02_split.json"
# split = ImgSplit(basepath,annofile, annomode, outpath, outannofile)
# split.splitdata(1)

# import json
# lnum=0
# fr=open('/root/data/gvision/dataset/image_annos/person_bbox_train.json', 'r')
# fw=open('/root/data/gvision/dataset/scene_02/person_bbox_train_02.json', 'w')
# # data=json.load(fr)
# # print(type(data))
# for line in fr:
#     lnum += 1
# #69150
# # 295149

#     if (lnum >= 69150) and (lnum <= 295149):
# #         print(line)
#         fw.write(line)

# fw.close()

# fr.close()
