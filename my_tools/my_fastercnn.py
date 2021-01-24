from Res2coco import res2coco 
from Post_processing import result_analysis
import ResultMerge
import numpy as np
from detectron2 import model_zoo
from detectron2.config import get_cfg
import tqdm,time,torch,logging,json,csv,os,random,cv2
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer
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
)

def get_balloon_dicts(img_dir):
    json_file = os.path.join(img_dir, "via_region_data.json")
    with open(json_file) as f:
        imgs_anns = json.load(f)

    dataset_dicts = []
    for idx, v in enumerate(imgs_anns.values()):
        record = {}
        filename = os.path.join(img_dir, v["filename"])
        height, width = cv2.imread(filename).shape[:2]
        
        record["file_name"] = filename
        record["image_id"] = idx
        record["height"] = height
        record["width"] = width
      
        annos = v["regions"]
        objs = []
        for _, anno in annos.items():
            assert not anno["region_attributes"]
            anno = anno["shape_attributes"]
            px = anno["all_points_x"]
            py = anno["all_points_y"]
            poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            poly = [p for x in poly for p in x]

            obj = {
                "bbox": [np.min(px), np.min(py), np.max(px), np.max(py)],
                "bbox_mode": BoxMode.XYXY_ABS,
                "segmentation": [poly],
                "category_id": 0,
                "iscrowd": 0
            }
            objs.append(obj)
        record["annotations"] = objs
        dataset_dicts.append(record)
    return dataset_dicts

from detectron2.data import DatasetCatalog, MetadataCatalog
for d in ["train", "val"]:
    os.path.join("/root/data/gvision/dataset/raw_data/ballon/", d)
    DatasetCatalog.register("balloon_" + d, lambda d=d: get_balloon_dicts(os.path.join("/root/data/gvision/dataset/raw_data/ballon", d)))
    MetadataCatalog.get("balloon_" + d).set(thing_classes=["balloon"])
balloon_metadata = MetadataCatalog.get("balloon_train")
print("------------------------dataset")

"""To verify the data loading is correct, let's visualize the annotations of randomly selected samples in the training set:"""
print(os.getcwd())
dataset_dicts = get_balloon_dicts("/root/data/gvision/dataset/raw_data/ballon/train")
for d in random.sample(dataset_dicts, 3):
    img = cv2.imread(d["file_name"])
    visualizer = Visualizer(img[:, :, ::-1], metadata=balloon_metadata, scale=0.5)
    vis = visualizer.draw_dataset_dict(d)
    cv2.imwrite("/root/data/gvision/dataset/raw_data/ballon/annos_in_image/ball.jpg",vis.get_image()[:, :, ::-1])


class Trainer(DefaultTrainer):
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        evaluator_list = []
        evaluator_type = MetadataCatalog.get(dataset_name).evaluator_type
        if evaluator_type in ["coco", "coco_panoptic_seg"]:
            evaluator_list.append(COCOEvaluator(dataset_name, cfg, True, output_folder))
        if len(evaluator_list) == 0:
            raise NotImplementedError(
                "no Evaluator for the dataset {} with the type {}".format(
                    dataset_name, evaluator_type
                )
            )
        elif len(evaluator_list) == 1:
            return evaluator_list[0]
        return DatasetEvaluators(evaluator_list)
    @classmethod
    def test_with_TTA(cls, cfg, model):
        logger = logging.getLogger("detectron2.trainer")
        # In the end of training, run an evaluation with TTA
        # Only support some R-CNN models.
        logger.info("Running inference with test-time augmentation ...")
        model = GeneralizedRCNNWithTTA(cfg, model)
        evaluators = [
            cls.build_evaluator(
                cfg, name, output_folder=os.path.join(cfg.OUTPUT_DIR, "inference_TTA")
            )
            for name in cfg.DATASETS.TEST
        ]
        res = cls.test(cfg, model, evaluators)
        res = OrderedDict({k + "_TTA": v for k, v in res.items()})
        return res
class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)
"""注册数据集"""
#inference
VAL_impath="/root/data/gvision/dataset/inference/VAL_split/image_train"
TRAIN_impath="/root/data/gvision/dataset/inference/TRAIN_split/image_train"
#train
train_impath="/root/data/gvision/dataset/train/s0.5_t0.8_all/image_train"
pv_raw_test_impath="/root/data/gvision/dataset/raw_data/image_test"
pv_split_test_impath="/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_test"
PREDEFINED_SPLITS_DATASET = {
    "person_TRAIN": ("/root/data/gvision/dataset/inference/TRAIN_split/image_annos/coco_person_TRAIN_hwnoi.json",TRAIN_impath),
    "vehicle_TRAIN": ("/root/data/gvision/dataset/inference/TRAIN_split/image_annos/coco_vehicle_TRAIN_hwnoi.json",TRAIN_impath),
    "pv_TRAIN": ("/root/data/gvision/dataset/inference/TRAIN_split/image_annos/coco_pv_TRAIN_hwnoi.json",TRAIN_impath),
    "person_VAL": ("/root/data/gvision/dataset/inference/VAL_split/image_annos/coco_person_VAL_hwnoi.json",VAL_impath),
    "vehicle_VAL": ("/root/data/gvision/dataset/inference/VAL_split/image_annos/coco_vehicle_VAL_hwnoi.json",VAL_impath),
    "pv_VAL": ("/root/data/gvision/dataset/inference/VAL_split/image_annos/coco_pv_VAL_hwnoi.json",VAL_impath),


    "person_train": ("/root/data/gvision/dataset/train/s0.5_t0.8_all/image_annos/coco_person_train_hwnoi.json",train_impath),
    "vehicle_train": ("/root/data/gvision/dataset/train/s0.5_t0.8_all/image_annos/coco_vehicle_train_split_hwnoi.json",train_impath),
    "pv_train": ("/root/data/gvision/dataset/train_all_annos/s0.3_t0.7_all/image_annos/coco_pv_train_bbox_hwnoi.json",
                "/root/data/gvision/dataset/train_all_annos/s0.3_t0.7_all/image_train"),
    "pv_raw_test": ("/root/data/gvision/dataset/predict/s0.5_t0.8_141517/resJSONS/coco_merge_result.json",pv_raw_test_impath),
    "pv_test": ("/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_annos/coco_person_bbox_test_141517_split.json",pv_split_test_impath),
}

def get_test_dicts(img_dir="/root/data/gvision/dataset/raw_data",annos_file="test_14_15_17.json"):
    attrDict = dict()
    attrDict["categories"] = [
            {"supercategory": "none", "id": 1, "name": 'visible body'},
            {"supercategory": "none", "id": 2, "name": 'full body'},
            {"supercategory": "none", "id": 3, "name": 'head'},
            {"supercategory": "none", "id": 4, "name": 'vehicle'}
        ]
    images = list()
    annotations=list()
    json_file = os.path.join(img_dir, "image_annos",annos_file)
    with open(json_file) as f:
        imgs_anns = json.load(f)
    for idx,(filename,image_dicts) in enumerate(imgs_anns.items()):
        record = {}
        record["file_name"] = filename
        record["id"] = image_dicts["image id"]
        record["height"] = image_dicts["image size"]["height"]
        record["width"] =image_dicts["image size"]["width"]
        images.append(record)
    attrDict["images"] = images
    attrDict["annotations"] = annotations
    attrDict["type"] = "instances"

    return attrDict

DatasetCatalog.register("test" ,lambda:get_test_dicts("/root/data/gvision/dataset/raw_data",annos_file="test_14_15_17.json"))
MetadataCatalog.get("test").set(thing_classes=['visible body','full body','head','vehicle'])

cfg = get_cfg()
cfg.merge_from_file("/root/data/gvision/detectron2-master/configs/COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml")
# cfg.merge_from_file("/root/data/gvision/detectron2-master/configs/COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml")

cfg.SOLVER.BASE_LR = 0.005  # pick a good 
# 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.SOLVER.IMS_PER_BATCH = 2*4# batch_size=2*5; iters_in_one_epoch = dataset_imgs/batch_size 22302
# ITERS_IN_ONE_EPOCH = int(9254/cfg.SOLVER.IMS_PER_BATCH )
ITERS_IN_ONE_EPOCH = int(61/cfg.SOLVER.IMS_PER_BATCH )
# 保存模型文件的命名数据减1# Save a checkpoint after every this number of iterations
cfg.SOLVER.CHECKPOINT_PERIOD =100
# cfg.SOLVER.MAX_ITER =1000
# cfg.SOLVER.MAX_ITER =ITERS_IN_ONE_EPOCH * 5+1  # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.SOLVER.MAX_ITER =400
cfg.OUTPUT_DIR="/root/data/gvision/dataset/d2_output/my_balloon_faster_lr0.02"
cfg.DATASETS.TRAIN = ("balloon_train",)
# cfg.DATASETS.TRAIN=("pv_train",) 
cfg.DATASETS.TEST=("balloon_val",) 
# cfg.DATASETS.TEST=("pv_test",) 
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1 # only has one class (ballon)
os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)
with open(os.path.join(cfg.OUTPUT_DIR,"config1.yaml"),'w') as f:    #设置文件对象
    f.write("{}".format(cfg))  
# cfg.freeze()
def register_dataset():
    """
    purpose: register all splits of dataset with PREDEFINED_SPLITS_DATASET
    """
    for key, (json_file,image_root) in PREDEFINED_SPLITS_DATASET.items():
        register_coco_instances(key,{},json_file,image_root)

def checkout_dataset_annotation(save_path="/root/data/gvision/dataset/train_all_annos/s0.3_t0.7_all/pv_image_in_annos",subdata_name="pv_train",showwidth=640):
    """
    Create configs and perform basic setups.  
    DATASETS:person(vehicle or pv)_VAL(TRAIN or train) or 查看融合的结果
    """
    os.makedirs(save_path,exist_ok=True)
    metadata_dicts= MetadataCatalog.get(subdata_name)
    print("checkout_dataset_annotation------------------------------------------------start")
    dataset_dicts = DatasetCatalog.get(subdata_name)
    # random.seed(0)
    # for d in dataset_dicts:
    for d in random.sample(dataset_dicts, 10):
        img = cv2.imread(d["file_name"])
        visualizer = Visualizer(img[:, :, ::-1], metadata=metadata_dicts, scale=1)#font sacle
        vis = visualizer.draw_dataset_dict(d)
        save_name=os.path.join(save_path,"visual_{}".format(os.path.basename(d["file_name"])))
        print(save_name)
        img=vis.get_image()[:, :, ::-1]
        imgheight, imgwidth = img.shape[:2]
        scale = showwidth / imgwidth
        img = cv2.resize(vis.get_image()[:, :, ::-1], (int(imgwidth * scale), int(imgheight * scale)),interpolation=cv2.INTER_AREA)
        cv2.imwrite(save_name,img ,[int(cv2.IMWRITE_PNG_COMPRESSION), 9])
    print("checkout_dataset{}_annotation------------------------------------------------end".format(subdata_name))
    """
    Create configs and perform basic setups.  
    DATASETS:person(vehicle or pv)_VAL(TRAIN or train)
    """
def train(train_flag,resume=False):
    # trainer= Trainer(cfg)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume)
    if train_flag:
        trainer.train()
    return trainer
def predict_ballon(): 
    cfg.MODEL.WEIGHTS=os.path.join(cfg.OUTPUT_DIR,"model_final.pth")
    dataset_dicts = get_balloon_dicts("/root/data/gvision/dataset/raw_data/ballon/val")
    predictor = DefaultPredictor(cfg)
    for d in random.sample(dataset_dicts, 3):    
        file_name=os.path.basename(d["file_name"])
        print(file_name)
        
        im = cv2.imread(d["file_name"])
        outputs = predictor(im)
        print(outputs)
        v = Visualizer(im[:, :, ::-1],metadata=balloon_metadata, scale=0.8)#, instance_mode=ColorMode.IMAGE_BW)    # remove the colors of unsegmented pixels)
        v = v.draw_instance_predictions(outputs["instances"].to("cpu"))
        os.makedirs(os.path.join(cfg.OUTPUT_DIR,"my_predict_split_visual"),exist_ok=True)
        cv2.imwrite(os.path.join(cfg.OUTPUT_DIR,"my_predict_split_visual","vis_{}".format(file_name)),v.get_image()[:, :, ::-1])

def predict(cfg,megrge_result=False,visual=False):
    """
    instances format:{'pred_boxes': Boxes(tensor([[ 732.5856, 1598.1067,  766.4857, 1633.0486]], device='cuda:0')), 
    'scores': tensor([0.9482], device='cuda:0'), 'pred_classes': tensor([2], device='cuda:0')}
    BoxMode.convert(pre_instances.pred_boxes.tensor,from_mode=BoxMode.XYXY_ABS,to_mode=BoxMode.XYWH_ABS
    print("\n"+"-" * int(i/len(dataset_test_dicts.keys())*100*50) +">"+ "{}".format(i/len(dataset_test_dicts.keys())) + "%", end='\r')
    time.sleep(0.00001)
    json.dump(coco_list_results,f,cls=MyEncoder,indent=2)# print(type(dict_value))# print(type(dict_value["image id"]))
    """
    predictor = DefaultPredictor(cfg)
    test_annos_root_dir="/root/data/gvision/dataset/predict/s0.5_t0.8_141517"
    test_json="/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_annos/person_bbox_test_141517_split.json"
    # test_annos_root_dir="/root/data/gvision/dataset/predict/s0.5_t0.8_141517"
    # test_json="/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_annos/person_bbox_test_141517_split.json"
    # print(os.getcwd())
    dataset_test_dicts = json.load(open(test_json,"r"))
    coco_list_results=[]
    print("predict-------------------start")
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, "my_predict"),exist_ok=True)
    f=open(os.path.join(cfg.OUTPUT_DIR, "my_predict","pre_result.json"),'w') 
    # for j,(file_name,dict_value) in  enumerate(dataset_test_dicts.items()):
    for j,(file_name,dict_value) in  enumerate(random.sample(dataset_test_dicts.items(),20)):
        cate=[]
        coco_dict_results={}
        id_1,id_2,id_3,id_4=0,0,0,0
        print("{}\t{}-------------------{}".format(file_name,j,len(dataset_test_dicts.keys())),flush=True)
        img=cv2.imread(os.path.join("/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_test",file_name))
        pre_output =predictor(img)
        pre_instances=pre_output['instances']
        for i in range(len(pre_instances.scores)):
            coco_dict_results["image_id"]=dict_value["image id"]
            coco_dict_results["category_id"]=pre_instances.pred_classes.cpu().numpy()[i]+1
            coco_dict_results["bbox"]=pre_instances.pred_boxes.tensor.cpu().numpy()[i]#pre_output['instances'].to("cpu")
            coco_dict_results["score"]=pre_instances.scores.cpu().numpy()[i]
            coco_list_results.append(coco_dict_results)
        # if save=FALSE:
            if visual:
                if 1:
                    # b = random.randint(0, 255)
                    # g = random.randint(0, 255)
                    # r = random.randint(0, 255)
                    cate.append(coco_dict_results["category_id"])
                    xmin, ymin, xmax , ymax = coco_dict_results["bbox"]
                    if coco_dict_results["category_id"]==1:#green
                        id_1+=1
                        img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (138,255,0), 8,lineType=8)
                        cv2.putText(img, '{}'.format(coco_dict_results["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 1.5, (138,255,0), 4)
                    if coco_dict_results["category_id"]==2:#pink
                        id_2+=1
                        img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (138,0,255), 8,lineType=8)
                        cv2.putText(img, '{}'.format(coco_dict_results["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 1.5, (138,0,255), 4)
                    if coco_dict_results["category_id"]==3:#purple
                        id_3+=1
                        img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,46,46), 8,lineType=8)
                        cv2.putText(img, '{}'.format(coco_dict_results["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255,46,46), 4)
                    if coco_dict_results["category_id"]==4:#grey
                        id_4+=1
                        img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (131,131,131), 8,lineType=8)
                        cv2.putText(img, '{}'.format(coco_dict_results["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 1.5, (131,131,131), 4)
                    cv2.putText(img, r"len{} cid:{}".format(len(pre_instances.scores),list(set(cate))[:]), (15,40), cv2.FONT_HERSHEY_COMPLEX, 1.5, (170,64,112), 4)#
                    cv2.putText(img, r"c1:{} c2:{} c3:{} c4:{}".format(id_1,id_2,id_3,id_4), (15,80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (170,64,112), 4)
                    #统计每个类别的数量
                os.makedirs(os.path.join(cfg.OUTPUT_DIR,"my_split_visual"),exist_ok=True)
                cv2.imwrite(os.path.join(cfg.OUTPUT_DIR,"my_split_visual","vis_per5555_{}".format(file_name)),img)
            if j%2000==0:
                f=open(os.path.join(cfg.OUTPUT_DIR, "my_predict","pre_result.json"),'w') 
                f.write(json.dumps("{}".format(coco_list_results),cls=MyEncoder))#
    f.write(json.dumps(coco_list_results,cls=MyEncoder))
    print("predict----------------end")
    if megrge_result:
        print("--------->>>>>>>>>merge-------------start")
        merge =ResultMerge.DetResMerge(resfile=os.path.join(cfg.OUTPUT_DIR, "my_predict","pre_result.json"), 
                                splitannofile=test_json, 
                                srcannofile="/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_annos/person_bbox_test_14_15_17.json",
                                outpath=cfg.OUTPUT_DIR,
                                outfile="resJSONS/pv_pre_merge_result.json")
        merge.mergeResults(is_nms=True)
        print("merge-------------end")
def merge():
        print("--------->>>>>>>>>merge-------------start")
        merge =ResultMerge.DetResMerge(resfile=os.path.join(cfg.OUTPUT_DIR, "my_inference","inference_results.json"), 
                                splitannofile="/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_annos/person_bbox_test_141517_split.json" ,
                                srcannofile="/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_annos/person_bbox_test_141517.json",
                                outpath=cfg.OUTPUT_DIR,
                                outfile="my_inference/pv_inference_merge_result.json")
        merge.mergeResults(is_nms=True)
        print("merge-------------end")

def main():
    "register data"
    register_dataset()
    "checkout raw dataset annotation"
    # checkout_dataset_annotation()
    "train"
    trainer=train(train_flag=True,resume=False)
    "predict and merge_nms"
    predict_ballon()
    # predict(cfg,megrge_result=False,visual=True)
    "merge"
    # merge()
    "checkout merge predict result annotation "
    # checkout_pre_annotation()
    # checkout_dataset_annotation((save_path=os.path.join(cfg.OUTPUT_DIR,"my_merge_visual"),subdata_name="pv_raw_test")
    "data analysis ,coco format save,output visual,megrgeresult visual--"
    # result_analysis()
    # evaluator = COCOEvaluator("pv_test",cfg, True, output_dir=os.path.join(cfg.OUTPUT_DIR,"my_pv_test_inference"))

    """Evaluate object proposal, instance detection/segmentation, keypoint detectionoutputs using COCO's metrics and APIs."""
    # evaluator = COCOEvaluator("pv_test",cfg, True, output_dir=os.path.join(cfg.OUTPUT_DIR,"my_test"))
    # trainer.test(cfg,trainer.model,evaluator)#与cfg.DATASETS.TEST=("pv_VAL",) 有关
    # evaluator = COCOEvaluator("pv_test",cfg, True, output_dir=os.path.join(cfg.OUTPUT_DIR,"my_inference"),result_name="pv_inference_results_11.json",pth_name="instances_predictions.pth")
    # val_loader = build_detection_test_loader(cfg,"pv_test")#与cfg.DATASETS.TEST=("pv_VAL",) 无关
    # inference_on_dataset(trainer.model, val_loader, evaluator=None)#only benchmark default evaluator

if __name__ == "__main__":
    main()




