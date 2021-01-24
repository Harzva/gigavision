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
"""
Create configs and perform basic setups.
"""
cfg = get_cfg()
cfg.merge_from_file("/root/data/gvision/detectron2-master/configs/COCO-InstanceSegmentation/my_mask_rcnn_R_50_FPN_3x.yaml")
cfg.SOLVER.BASE_LR = 0.00002 # pick a good LR
cfg.SOLVER.IMS_PER_BATCH = 2*2# batch_size=2*5; iters_in_one_epoch = dataset_imgs/batch_size 22302
ITERS_IN_ONE_EPOCH = int(8188/cfg.SOLVER.IMS_PER_BATCH )

cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False
cfg.TEST.DETECTIONS_PER_IMAGE=200
# ITERS_IN_ONE_EPOCH = int(61/cfg.SOLVER.IMS_PER_BATCH )
# 保存模型文件的命名数据减1# Save a checkpoint after every this number of iterations
cfg.SOLVER.CHECKPOINT_PERIOD =ITERS_IN_ONE_EPOCH
cfg.SOLVER.MAX_ITER =ITERS_IN_ONE_EPOCH *20 # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.OUTPUT_DIR="/root/data/gvision/detectron2-master/workdir/output/my_vehicle_ms_mask"
cfg.DATASETS.TRAIN=("pandavehicle_07","pandavehicle") 
# cfg.DATASETS.TRAIN=("crowdh_all_train","crowdh_all_val") 
cfg.DATASETS.TEST=("pv_test",) 
cfg.MODEL.WEIGHTS=os.path.join(cfg.OUTPUT_DIR,"model_0004093.pth")
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST=0.2
cfg.MODEL.ROI_HEADS.NMS_THRESH_TEST=0.3
cfg.MODEL.ROI_HEADS.NUM_CLASSES =8
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES =8
os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)
with open(os.path.join(cfg.OUTPUT_DIR,"config_train.yaml"),'w') as f: 
    f.write("{}".format(cfg))  
# cfg.freeze()
def train(train_flag,resume_load=False):
    # trainer= Trainer(cfg)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume_load)
    if train_flag:
        trainer.train()
    # return trainer
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
def predict_my(visual=True,model_weight="model_final.pth"):
    cfg.MODEL.WEIGHTS=os.path.join(cfg.OUTPUT_DIR,model_weight)
    predictor = DefaultPredictor(cfg)
    # test_json="/root/data/gvision/dataset/predict/17/image_annos/s0.5_17_split_test.json"
    # test_image_path="/root/data/gvision/dataset/predict/17/image_test"
    test_json="/root/data/gvision/dataset/train/ruby_output/dyy_split_vehicle_07.json"
    test_image_path="/root/data/gvision/dataset/train/ruby_output/vehicle_07"
    dataset_test_dicts = json.load(open(test_json,"r"))
    MetadataCatalog.get("pv_train").set(thing_classes=["small car","midsize car","large car","bicycle","baby carriage","motorcycle","tricycle","electric car"], thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3,5:4,6:5,7:6,8:7})
    train_dicts_metadata = MetadataCatalog.get("pv_train")
    print("metadata----------------",train_dicts_metadata)
    print("predict-------------------start")
    "thing_classes=['visible body', 'full body', 'head', 'vehicle'], thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3}"
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, "my_predict"),exist_ok=True)
    # for j,(file_name,dict_value) in  enumerate(random.sample(dataset_test_dicts.items(),5)):
    coco_list_results=[]
    for j,(file_name,dict_value) in  enumerate(dataset_test_dicts.items()):
        print("{}\t{}------------------{}\t{}".format(file_name,j,model_weight[6:-4],len(dataset_test_dicts.keys())),flush=True)
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
            v = Visualizer(img[:, :, ::-1],metadata=train_dicts_metadata, scale=1,instance_mode=ColorMode.IMAGE)# ColorMode.SEGMENTATION or ColorMode.IMAGE_BW) 
            v = v.draw_instance_predictions(pre_instances.to("cpu"))#draw xyxy
            os.makedirs(os.path.join(cfg.OUTPUT_DIR,"d2_predict_split_visual_7_01"),exist_ok=True)
            cv2.imwrite(os.path.join(cfg.OUTPUT_DIR,"d2_predict_split_visual_7_01","visual{}_{}".format(model_weight[6:-4],file_name)),v.get_image()[:, :, ::-1])
    print(os.path.join(cfg.OUTPUT_DIR, "my_predict",f"{model_weight[6:-4]}_predict_14_01.json"))
    f1=open(os.path.join(cfg.OUTPUT_DIR, "my_predict",f"{model_weight[6:-4]}_predict_14_01.json"),'w') 
    f1.write(json.dumps(coco_list_results,cls=MyEncoder))
    print("predict----------------end")
def single_predict(filter_cate=None):
    cfg.TEST.DETECTIONS_PER_IMAGE=500
    cfg.MODEL.WEIGHTS=os.path.join(cfg.OUTPUT_DIR,"model_0030704.pth")
    img=cv2.imread("/root/data/gvision/dataset/train/ruby_output/vehicle_07/07_East_Gate_IMG_07_01___0.3___12360___1180.jpg")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(img)
    pre_instances=outputs['instances']
    # # We can use `Visualizer` to draw the predictions on the image.
    # MetadataCatalog.get("pv_train").set(thing_classes=["small car","midsize car","large car","bicycle","baby carriage","motorcycle","tricycle","electric car"], thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3,5:4,6:5,7:6,8:7})
    train_dicts_metadata = MetadataCatalog.get("pandavehicle_07")
    print(train_dicts_metadata )
    # print(train_dicts_metadata)
    # v = Visualizer(img[:, :, ::-1], train_dicts_metadata, scale=1,instance_mode=ColorMode.IMAGE)
    # v = v.draw_instance_predictions(outputs["instances"].to("cpu")) 
    # # print(outputs["instances"])
    # cv2.imwrite("/root/data/gvision/detectron2-master/workdir/output/test/ouput_2.jpg",v.get_image()[:, :, ::-1])
    bboxes=pre_instances.pred_boxes.tensor.cpu().numpy()

    category_ids=pre_instances.pred_classes.cpu().numpy()
    print(category_ids)
    # print(category_ids)
    if filter_cate!=None:
        assert filter_cate-1 in category_ids,f"category not have cls:{filter_cate }"
    score=pre_instances.scores.cpu().numpy()
    "fliter"
    if filter_cate!=None:
        bboxes=[bboxes[i] for i in range(len(bboxes)) if category_ids[i]==filter_cate-1]
        score==[score[i] for i in range(len(score)) if category_ids[i]==filter_cate-1]
        category_ids=[category_ids[i] for i in range(len(category_ids)) if category_ids[i]==filter_cate-1]
    "fliter"
    # class_names=_create_text_labels(category_ids,['visible body', 'full body', 'head', 'vehicle'],score)
    bboxes=[list(bboxes[i]) for i in range(len(bboxes))]
    score=np.resize(score,(len(score),1))
    bboxes=[old+list(new) for old,new in zip(bboxes,score)]
    # print([old+new for old,new in zip(bboxes,score)])
    # print(score)
    # category_ids=[category_ids[i]+1 for i in range(len(category_ids))]
    #  class_names=["small car","midsize car","large car","bicycle","baby carriage","motorcycle","tricycle","electric car"],
    # mmcv.imshow_bboxes(img,bboxes,top_k=10,show=False,out_file="/root/data/gvision/detectron2-master/demo/ouputmmcv_10_1.jpg")
    # mmcv.imshow_bboxes(img,bboxes,show=False,out_file="/root/data/gvision/1detectron2-master/demo/ouputmmcv2_all_1.jpg")
    # mmcv.imshow_bboxes(img,bboxes,top_k=500,show=False,out_file="/root/data/gvision/detectron2-master/demo/ouputmmcv_500_1.jpg")
    out_file=os.path.join(cfg.OUTPUT_DIR,f"ouput_111c{filter_cate}.jpg")
    print(out_file)
    mmcv.imshow_det_bboxes(
        img,
        np.array(bboxes),
        np.array(category_ids),
        show=False,
        out_file=out_file)



def main():
    "register data"
    register_dataset()
    "checkout raw dataset annotation"
    # checkout_dataset_annotation()
    "train"
    train(train_flag=False,resume_load=True)
    # single_predict()
    "check pre annotation"
    # checkout_pre_annotation(visual_num=10)
    "predict and merge_nms" 
    single_predict()
    # for i in ["model_0004093.pth"]:
    #     predict_my(model_weight=i,visual=True)

if __name__ == "__main__":
    main()




