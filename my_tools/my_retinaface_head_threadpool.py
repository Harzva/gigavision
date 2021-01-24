from Res2coco import res2coco 
from Post_processing import result_analysis
import ResultMerge
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
import tqdm,time,torch,logging,json,csv,os,random,cv2
from detectron2.data.datasets import register_dataset,MyEncoder,register_coco_instances
from detectron2.utils.visualizer import Visualizer
import mmcv
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
from collections import OrderedDict
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader,MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch,DefaultPredictor
from detectron2.evaluation import (
    COCOEvaluator,
    verify_results,
    DatasetEvaluators,
    DatasetEvaluator,
)
import os

from tqdm import tqdm 

from detectron2.checkpoint import DetectionCheckpointer

# from detectron2.config import get_cfg
from detectron2.data import build_detection_test_loader, build_detection_train_loader
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, launch

from aifwdet.config import get_cfg
from aifwdet.data.datasets.widerface import register_widerface
from aifwdet.data.widerface_dataset_mapper import WiderFace_DatasetMapper

# from aifwdet.engine.apex_trainer import ApexTrainer
from aifwdet.evaluation.evaluator import WiderFaceEvaluator
from detectron2.evaluation import (COCOEvaluator,
                                   COCOPanopticEvaluator, DatasetEvaluators,
                                   LVISEvaluator, PascalVOCDetectionEvaluator,
                                   SemSegEvaluator, verify_results)
from detectron2.modeling import GeneralizedRCNNWithTTA
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
import mmcv
from detectron2.structures import BoxMode
from detectron2.utils.visualizer import ColorMode
from collections import OrderedDict
import detectron2.utils.comm as comm
from detectron2.checkpoint import DetectionCheckpointer
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader,MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultTrainer, default_argument_parser, default_setup, hooks, launch,DefaultPredictor
from detectron2.evaluation import (
    COCOEvaluator,
    verify_results,
    DatasetEvaluators,
    DatasetEvaluator,
)
from detectron2.data.datasets import register_dataset,MyEncoder,register_coco_instances

def instances_to_coco_json(instances, img_id):
    """
    Dump an "Instances" object to a COCO-format json that's used for evaluation.

    Args:
        instances (Instances):
        img_id (int): the image id

    Returns:
        list[dict]: list of json annotations in COCO format.
    """
    num_instance = len(instances)
    if num_instance == 0:
        print("no pre")
        return []

    boxes = instances.pred_boxes.tensor.numpy()
    boxes = BoxMode.convert(boxes, BoxMode.XYXY_ABS, BoxMode.XYWH_ABS)
    boxes = boxes.tolist()
    scores = instances.scores.tolist()
    classes = instances.pred_classes.tolist()
    results = []
    id_1,id_2,id_3,id_4=0,0,0,0
    for k in range(num_instance):
        if classes[k]==0:
            id_1+=1
        if classes[k]==1:
            id_2+=1
        if classes[k]==2:
            id_3+=1    
        if classes[k]==3:
            id_4+=1      
        result = {"image_id": img_id,"category_id": classes[k]+1,"bbox": boxes[k],"score": scores[k],}
        results.append(result)
    cid=[id_1,id_2,id_3,id_4]
    return results,num_instance,cid
"""
Create configs and perform basic setups.
"""
cfg = get_cfg()
cfg.merge_from_file("/root/data/gvision/detectron2-master/projects/Retinaface/configs/facetron/retinaface_r_50_3x.yaml")
cfg.SOLVER.BASE_LR = 0.00002 # pick a good LR
cfg.SOLVER.IMS_PER_BATCH = 2*2# batch_size=2*5; iters_in_one_epoch = dataset_imgs/batch_size 22302
ITERS_IN_ONE_EPOCH = int(8188/cfg.SOLVER.IMS_PER_BATCH )
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False
cfg.TEST.DETECTIONS_PER_IMAGE=50
# ITERS_IN_ONE_EPOCH = int(61/cfg.SOLVER.IMS_PER_BATCH )
# 保存模型文件的命名数据减1# Save a checkpoint after every this number of iterations
cfg.SOLVER.CHECKPOINT_PERIOD =ITERS_IN_ONE_EPOCH
cfg.SOLVER.MAX_ITER =ITERS_IN_ONE_EPOCH *20 # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.OUTPUT_DIR="/root/data/gvision/detectron2-master/workdir/output/my_head_retinaface_ms_panda"
cfg.DATASETS.TRAIN=("pandahead",) 
# cfg.DATASETS.TRAIN=("crowdh_all_train","crowdh_all_val") 
cfg.DATASETS.TEST=("pv_test",) 
cfg.MODEL.WEIGHTS=os.path.join(cfg.OUTPUT_DIR,"model_0028043.pth")
cfg.MODEL.ROI_HEADS.NUM_CLASSES =1
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES =1
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.5
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST=0.3
cfg.MODEL.RETINANET.SCORE_THRESH_TEST=0.1
cfg.MODEL.RETINANET.NMS_THRESH_TEST=0.3
cfg.MODEL.RETINANET.NUM_CLASSES =4
os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)
with open(os.path.join(cfg.OUTPUT_DIR,"config_test.yaml"),'w') as f: 
    f.write("{}".format(cfg))  
with open(os.path.join(cfg.OUTPUT_DIR,"config_train.yaml"),'w') as f: 
    f.write("{}".format(cfg))  
# cfg.freeze()
def train(train_flag,resume_load=False):
    # trainer= Trainer(cfg)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume_load)
    if train_flag:
        trainer.train()

def predict_my(visual=True,model_weight="model_final.pth"):
    cfg.MODEL.WEIGHTS=os.path.join(cfg.OUTPUT_DIR,model_weight)
    predictor = DefaultPredictor(cfg)
    cfg.TEST.DETECTIONS_PER_IMAGE=500
    # test_json= "/root/data/rubzz/ruby/ruby_output/test/person/split_test_method2_person.json"
    # test_image_path="/root/data/rubzz/ruby/ruby_output/test/person/split_test_method2_person"
    # test_json="/root/data/gvision/dataset/predict/person/test_person.json"
    # test_image_path="/root/data/gvision/dataset/predict/person/img"
    test_json="/root/data/rubzz/ruby/ruby_output/test/person/img_testone/test_person_onetest.json"
    test_image_path= '/root/data/rubzz/ruby/ruby_output/test/person/img_testone'
    dataset_test_dicts = json.load(open(test_json,"r"))
    # MetadataCatalog.get("pandahead").set(thing_classes=["head"], thing_dataset_id_to_contiguous_id={1: 0})
    train_dicts_metadata = MetadataCatalog.get("pandahead")
    print("metadata----------------",train_dicts_metadata)
    print("predict-------------------start")
    "thing_classes=['visible body', 'full body', 'head', 'vehicle'], thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3}"
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, "my_predict"),exist_ok=True)
    # for j,(file_name,dict_value) in  enumerate(random.sample(dataset_test_dicts.items(),5)):
    
    # pbar = tqdm(total=len(dataset_test_dicts), ncols=50)
    import threadpool  
    func_var = [([file_name,dict_value ], None)for file_name,dict_value in dataset_test_dicts.items()]
    
    def sayhello(file_name,dict_value):
        coco_list_results=[]
        print("{}------------------{}\t{}".format(os.path.join(test_image_path,file_name),model_weight[6:-4],len(dataset_test_dicts.keys())),flush=True)
        img=cv2.imread(os.path.join(test_image_path,file_name))
        pre_output =predictor(img)
        num_instance=0
        cid=[0,0,0,0]
        pre_instances=pre_output['instances']
        if "instances" in pre_output and len(pre_instances)!=0:
            coco_list_result,num_instance,cid=instances_to_coco_json(pre_instances.to(torch.device("cpu")),dict_value["image id"])
            coco_list_results=coco_list_results+coco_list_result
        srcfile, paras = file_name.split('___')
        srcfile =srcfile.replace('_IMG', '/IMG') 
        image_id=srcfile[-2:]
        scale, left, up = paras.replace('.jpg', '').split('__')
        if visual and (up=="16480" or up=="12360"):
            print("visual-------------------------",file_name)
            cv2.putText(img, f"len:{num_instance} c1:{cid[0]} c2:{cid[1]} c3:{cid[2]} c4:{cid[3]}", (15,80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (138,0,255), 4)
            v = Visualizer(img[:, :, ::-1],metadata=train_dicts_metadata, scale=1)# ColorMode.SEGMENTATION or ColorMode.IMAGE_BW) 
            v = v.draw_instance_predictions(pre_instances.to("cpu"))#draw xyxy
            os.makedirs(os.path.join(cfg.OUTPUT_DIR,"d2_predict_split_visual_17_01"),exist_ok=True)
            cv2.imwrite(os.path.join(cfg.OUTPUT_DIR,"d2_predict_split_visual_17_01","visual{}_{}".format(model_weight[6:-4],file_name)),v.get_image()[:, :, ::-1])
    #     pbar.update(1)
    # pbar.close()
    
    pool = threadpool.ThreadPool(500) 
    requests = threadpool.makeRequests(sayhello,func_var) 
    [pool.putRequest(req) for req in requests] 
    pool.wait() 

    
    # for file_name,dict_value in dataset_test_dicts.items():
    #     # print("{}\t{}------------------{}\t{}".format(os.path.join(test_image_path,file_name),j,model_weight[6:-4],len(dataset_test_dicts.keys())),flush=True)
    #     img=cv2.imread(os.path.join(test_image_path,file_name))
    #     pre_output =predictor(img)
    #     num_instance=0
    #     cid=[0,0,0,0]
    #     pre_instances=pre_output['instances']
    #     if "instances" in pre_output and len(pre_instances)!=0:
    #         coco_list_result,num_instance,cid=instances_to_coco_json(pre_instances.to(torch.device("cpu")),dict_value["image id"])
    #         coco_list_results=coco_list_results+coco_list_result
    #     srcfile, paras = file_name.split('___')
    #     srcfile =srcfile.replace('_IMG', '/IMG') 
    #     image_id=srcfile[-2:]
    #     scale, left, up = paras.replace('.jpg', '').split('__')
    #     if visual and (up=="16480" or up=="12360"):
    #         print("visual-------------------------",file_name)
    #         cv2.putText(img, f"len:{num_instance} c1:{cid[0]} c2:{cid[1]} c3:{cid[2]} c4:{cid[3]}", (15,80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (138,0,255), 4)
    #         v = Visualizer(img[:, :, ::-1],metadata=train_dicts_metadata, scale=1)# ColorMode.SEGMENTATION or ColorMode.IMAGE_BW) 
    #         v = v.draw_instance_predictions(pre_instances.to("cpu"))#draw xyxy
    #         os.makedirs(os.path.join(cfg.OUTPUT_DIR,"d2_predict_split_visual_17_01"),exist_ok=True)
    #         cv2.imwrite(os.path.join(cfg.OUTPUT_DIR,"d2_predict_split_visual_17_01","visual{}_{}".format(model_weight[6:-4],file_name)),v.get_image()[:, :, ::-1])
    tempc=os.path.join(cfg.OUTPUT_DIR, "my_predict",f"{model_weight[6:-4]}_nms{cfg.MODEL.RETINANET.NMS_THRESH_TEST}_fs{cfg.MODEL.RETINANET.SCORE_THRESH_TEST}_17_01.json")
    print(tempc)
    f1=open(tempc,'w') 
    f1.write(json.dumps(coco_list_results,cls=MyEncoder))
    print("predict----------------end")
def merge():
    print("--------->>>>>>>>>merge-------------start")
    merge =ResultMerge.DetResMerge(resfile=os.path.join(cfg.OUTPUT_DIR, "my_predict","0017574_predict_all.json"), 
                            splitannofile="/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_annos/person_bbox_test_141517_split.json" ,
                            srcannofile="/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_annos/person_bbox_test_141517.json",
                            outpath=cfg.OUTPUT_DIR,
                            )
    nms_thresh_list=[0.9,0.8]        
    for i in nms_thresh_list:
        merge.mergeResults(is_nms=True,nms_thresh=i,outfile=f"my_predict/nms{i}_ms_0017574_merge_predict_all.json")
    # merge =ResultMerge.DetResMerge(resfile="/root/data/gvision/dataset/d2_output/my_pv_mask/my_predict/test.json", 
    #                         splitannofile="/root/data/gvision/dataset/predict/17/image_annos/s0.5_17_split_test.json" ,
    #                         srcannofile="/root/data/gvision/dataset/raw_data/image_annos/17.json",
    #                         outpath=cfg.OUTPUT_DIR,
    #                         mode="xywh",
    #                         outfile="my_predict/17_pred_merge_xyxy.json")
    # merge.mergeResults(is_nms=True, nms_thresh=0.5)
    # print("merge-------------end")

def main():
    "register data"
    register_dataset()
    "checkout raw dataset annotation"
    # checkout_dataset_annotation()
    "train"
    train(train_flag=False,resume_load=False)
    # single_predict()
    "check pre annotation"
    # checkout_pre_annotation(visual_num=10)
    "predict and merge_nms" 
    # # single_predict()
    for i in ["model_0032717.pth"]:
        predict_my(model_weight=i,visual=True)
    # # "merge"
    # merge()
    # "-------------"
    "checkout merge predict result annotation "
    # checkout_pre_annotation()
    # checkout_dataset_annotation((save_path=os.path.join(cfg.OUTPUT_DIR,"my_merge_visual"),subdata_name="pv_raw_test")
    "data analysis ,coco format save,output visual,megrgeresult visual--"
    # result_analysis()


    """Evaluate object proposal, instance detection/segmentation, keypoint detectionoutputs using COCO's metrics and APIs."""
    # cfg.MODEL.WEIGHTS=os.path.join(cfg.OUTPUT_DIR,"model_final.pth")
    # cfg.MODEL.WEIGHTS=os.path.join(cfg.OUTPUT_DIR,"model_final.pth")
    # evaluator = COCOEvaluator("pv_test",cfg, True, output_dir=os.path.join(cfg.OUTPUT_DIR,"my_pv_mask_test_inference1"))
    # trainer.test(cfg,trainer.model,evaluator)#与cfg.DATASETS.TEST=("pv_VAL",) 有关
    # evaluator = DatasetEvaluators("pv_test",cfg, True, output_dir=os.path.join(cfg.OUTPUT_DIR,"my_inference"),result_name="pv_inference_results_11.json",pth_name="instances_predictions.pth")
    # val_loader = build_detection_test_loader(cfg,"test")#与cfg.DATASETS.TEST=("pv_VAL",) 无关
    # inference_on_dataset(trainer.model, val_loader, evaluator=None)
if __name__ == "__main__":
    main()




