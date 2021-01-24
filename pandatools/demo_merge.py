import os
from PANDA import PANDA_IMAGE, PANDA_VIDEO
import panda_utils as util
from ImgSplit import ImgSplit
from ResultMerge import DetResMerge

# print(os.getcwd())
# os.chdir("/root/data/gvision/detectron2-master") 
# basepath='/root/data/gvision/dataset/split/person_s0.5_t0.7_01_02'
# # resfile="split_result.json"#检测结果未融合
# resfile=os.path.join(basepath, 'image_annos', "person_bbox_train_split_02.json")#"split_result.json"#未融合的检测结果
# splitannofile=os.path.join(basepath, 'image_annos', "person_bbox_train_split_02.json")##切图后的GT
# srcannofile="/root/data/gvision/dataset/image_annos/person_bbox_train.json"#未切图的GT，最后是test

# outpath="/root/data/gvision/dataset/split/person_s0.5_t0.7_01_02/resJSONS"
# outfile="merge_result.json"#融合的检测结果
# merge = DetResMerge(basepath, resfile, splitannofile, srcannofile, outpath, outfile)
# merge.mergeResults()
# # merge.mergeResults(is_nms=False)

resfile="/root/data/gvision/dataset/rs_output/my_p/results.json.bbox.json"#检测结果未融合
# resfile=os.path.join(basepath, 'image_annos', "/root/data/gvision/detectron2-master/output/train_coco_pre/coco_pre_result.json")#"split_result.json"#未融合的检测结果
splitannofile="/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_annos/person_bbox_test_141517_split.json"##切图后的GT
srcannofile="/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_annos/person_bbox_test_141517.json"#未切图的GT，最后是test

outpath="/root/data/gvision/dataset/rs_output"
outfile="merge_result.json"#融合的检测结果
merge = DetResMerge(resfile, splitannofile, srcannofile, outpath, outfile)
# merge.mergeResults()
merge.mergeResults(is_nms=True)
print("-----------------end")