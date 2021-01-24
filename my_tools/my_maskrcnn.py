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
cfg.SOLVER.BASE_LR = 0.02 # pick a good LR
cfg.SOLVER.IMS_PER_BATCH = 2*2# batch_size=2*5; iters_in_one_epoch = dataset_imgs/batch_size 22302
ITERS_IN_ONE_EPOCH = int(9254/cfg.SOLVER.IMS_PER_BATCH )
cfg.DATALOADER.FILTER_EMPTY_ANNOTATIONS=False
cfg.TEST.DETECTIONS_PER_IMAGE=500
# ITERS_IN_ONE_EPOCH = int(61/cfg.SOLVER.IMS_PER_BATCH )
# 保存模型文件的命名数据减1# Save a checkpoint after every this number of iterations
cfg.SOLVER.CHECKPOINT_PERIOD =ITERS_IN_ONE_EPOCH
cfg.SOLVER.MAX_ITER =ITERS_IN_ONE_EPOCH *20 # 300 iterations seems good enough for this toy dataset; you may need to train longer for a practical dataset
cfg.OUTPUT_DIR="/root/data/gvision/detectron2-master/workdir/output/my_pv_center_ms_mask"
cfg.DATASETS.TRAIN=("pv_center_train",) 
# cfg.DATASETS.TRAIN=("crowdh_all_train","crowdh_all_val") 
cfg.DATASETS.TEST=("pv_test",) 
cfg.MODEL.WEIGHTS=os.path.join(cfg.OUTPUT_DIR,"model_final.pth")
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 4 
cfg.MODEL.SEM_SEG_HEAD.NUM_CLASSES = 4
os.makedirs(cfg.OUTPUT_DIR,exist_ok=True)
with open(os.path.join(cfg.OUTPUT_DIR,"config_train.yaml"),'w') as f: 
    f.write("{}".format(cfg))  
# cfg.freeze()
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
def train(train_flag,resume_load=False):
    # trainer= Trainer(cfg)
    trainer = DefaultTrainer(cfg) 
    trainer.resume_or_load(resume_load)
    if train_flag:
        trainer.train()
    return trainer
def checkout_pre_annotation(visual_num=5):
    """
    instances format:{'pred_boxes': Boxes(tensor([[ 732.5856, 1598.1067,  766.4857, 1633.0486]], device='cuda:0')), 
    'scores': tensor([0.9482], device='cuda:0'), 'pred_classes': tensor([2], device='cuda:0')}
    BoxMode.convert(pre_instances.pred_boxes.tensor,from_mode=BoxMode.XYXY_ABS,to_mode=BoxMode.XYWH_ABS
    print("\n"+"-" * int(i/len(dataset_test_dicts.keys())*100*50) +">"+ "{}".format(i/len(dataset_test_dicts.keys())) + "%", end='\r')
    time.sleep(0.00001)
    json.dump(coco_list_results,f,cls=MyEncoder,indent=2)# print(type(dict_value))# print(type(dict_value["image id"]))
    """
    test_json="/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_annos/person_bbox_test_141517_split.json"
    test_image_path="/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_test"
    cfg.MODEL.WEIGHTS=os.path.join(cfg.OUTPUT_DIR,"model_final.pth")
    predictor = DefaultPredictor(cfg)
    dataset_test_dicts = json.load(open(test_json,"r"))
    """metadata Metadata(evaluator_type='coco', image_root='/root/data/gvision/dataset/train_all_annos/s0.3_t0.7_all/image_train', 
    json_file='/root/data/gvision/dataset/train_all_annos/s0.3_t0.7_all/image_annos/coco_pv_train_bbox_hwnoi.json', name='pv_train', 
    thing_classes=['visible body', 'full body', 'head', 'vehicle'], thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3})"""
    MetadataCatalog.get("pv_train").set(thing_colors=[(138,255,0),(138,0,255),(255,46,46),(131,131,131)])
    # MetadataCatalog.get("pv_train").set(thing_colors=[(131,131,131),(131,131,131),(131,131,131),(131,131,131)])
    """green            pink        purple    grey
      ['visible body', 'full body', 'head', 'vehicle']
      1                   2           3        4
    """
    train_dicts_metadata = MetadataCatalog.get("pv_train")
    print("metadata",train_dicts_metadata)
    print("pre_visual-------------------start")
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, "my_predict"),exist_ok=True)
    # for j,(file_name,dict_value) in  enumerate(dataset_test_dicts.items()):
    for j,(file_name,dict_value) in  enumerate(random.sample(dataset_test_dicts.items(),visual_num)):
        cate=[]
        coco_dict_results={}
        id_1,id_2,id_3,id_4=0,0,0,0
        print("{}\t{}-------------------{}".format(file_name,j,visual_num),flush=True)
        img=cv2.imread(os.path.join(test_image_path,file_name))
        pre_output =predictor(img)
        pre_instances=pre_output['instances']
        for i in range(len(pre_instances.scores)):
            coco_dict_results["category_id"]=pre_instances.pred_classes.cpu().numpy()[i]+1
            coco_dict_results["bbox"]=pre_instances.pred_boxes.tensor.cpu().numpy()[i]#pre_output['instances'].to("cpu")
            cate.append(coco_dict_results["category_id"])
            xmin, ymin, xmax , ymax = coco_dict_results["bbox"]
            if coco_dict_results["category_id"]==1:#green
                id_1+=1
                # img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (138,255,0), 8,lineType=8)
                cv2.putText(img, '{}'.format(coco_dict_results["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 1.5, (138,255,0), 4)
            if coco_dict_results["category_id"]==2:#pink
                id_2+=1
                # img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (138,0,255), 8,lineType=8)
                cv2.putText(img, '{}'.format(coco_dict_results["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 1.5, (138,0,255), 4)
            if coco_dict_results["category_id"]==3:#purple
                id_3+=1
                # img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (255,46,46), 8,lineType=8)
                cv2.putText(img, '{}'.format(coco_dict_results["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 1.5, (255,46,46), 4)
            if coco_dict_results["category_id"]==4:#grey
                id_4+=1
                # img=cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (131,131,131), 8,lineType=8)
                cv2.putText(img, '{}'.format(coco_dict_results["category_id"]), (xmin,ymin), cv2.FONT_HERSHEY_COMPLEX, 1.5, (131,131,131), 4)
        cv2.putText(img, r"len{} c1:{} c2:{} c3:{} c4:{}".format(len(pre_instances.scores),id_1,id_2,id_3,id_4), (15,80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (138,0,255), 4)
        v = Visualizer(img[:, :, ::-1],metadata=train_dicts_metadata, scale=1,instance_mode=ColorMode.IMAGE)# ColorMode.SEGMENTATION1 or ColorMode.IMAGE_BW2 ColorMode.IMAGE0
        v = v.draw_instance_predictions(pre_output["instances"].to("cpu"))
        os.makedirs(os.path.join(cfg.OUTPUT_DIR,"predict_split_visual"),exist_ok=True)
        cv2.imwrite(os.path.join(cfg.OUTPUT_DIR,"predict_split_visual","vis_00{}".format(file_name)),v.get_image())
    print("pre_visual----------------end")
def predict(save_json=False,megrge_result=False,d2_visual=True,my_visual=False):
    """
    instances format:{'pred_boxes': Boxes(tensor([[ 732.5856, 1598.1067,  766.4857, 1633.0486]], device='cuda:0')), 
    'scores': tensor([0.9482], device='cuda:0'), 'pred_classes': tensor([2], device='cuda:0')}
    BoxMode.convert(pre_instances.pred_boxes.tensor,from_mode=BoxMode.XYXY_ABS,to_mode=BoxMode.XYWH_ABS
    print("\n"+"-" * int(i/len(dataset_test_dicts.keys())*100*50) +">"+ "{}".format(i/len(dataset_test_dicts.keys())) + "%", end='\r')
    time.sleep(0.00001)
    json.dump(coco_list_results,f,cls=MyEncoder,indent=2)# print(type(dict_value))# print(type(dict_value["image id"]))
    """
    cfg.MODEL.WEIGHTS=os.path.join(cfg.OUTPUT_DIR,"model_final.pth")
    predictor = DefaultPredictor(cfg)
    # test_annos_root_dir="/root/data/gvision/dataset/predict/s0.5_t0.8_141517"
    # test_json="/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_annos/person_bbox_test_141517_split.json"
    test_image_path="/root/data/gvision/dataset/predict/s0.5_t0.9_14/image_test"
    test_json="/root/data/gvision/dataset/predict/s0.5_t0.9_14/image_annos/person_s0.5_t0.9_14_split_test.json"
    dataset_test_dicts = json.load(open(test_json,"r"))
    """metadata Metadata(evaluator_type='coco', image_root='/root/data/gvision/dataset/train_all_annos/s0.3_t0.7_all/image_train', 
    json_file='/root/data/gvision/dataset/train_all_annos/s0.3_t0.7_all/image_annos/coco_pv_train_bbox_hwnoi.json', name='pv_train', 
    thing_classes=['visible body', 'full body', 'head', 'vehicle'], thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3})"""
    MetadataCatalog.get("pv_train").set(thing_colors=[(138,255,0),(138,0,255),(255,46,46),(131,131,131)])
    # MetadataCatalog.get("pv_train").set(thing_colors=[(131,131,131),(131,131,131),(131,131,131),(131,131,131)])
    """green            pink        purple    grey
      ['visible body', 'full body', 'head', 'vehicle']
      1                   2           3        4
    """
    train_dicts_metadata = MetadataCatalog.get("pv_train")
    print("metadata",train_dicts_metadata)
    coco_list_results=[]
    print("predict-------------------start")
    os.makedirs(os.path.join(cfg.OUTPUT_DIR, "my_predict"),exist_ok=True)
    # for j,(file_name,dict_value) in  enumerate(dataset_test_dicts.items()):
    for j,(file_name,dict_value) in  enumerate(random.sample(dataset_test_dicts.items(),9)):
        cate=[]
        coco_dict_results={}
        id_1,id_2,id_3,id_4=0,0,0,0
        print("{}\t{}-------------------{}".format(file_name,j,len(dataset_test_dicts.keys())),flush=True)
        img=cv2.imread(os.path.join(test_image_path,file_name))
        pre_output =predictor(img)
        pre_instances=pre_output['instances']
        for i in range(len(pre_instances.scores)):
            coco_dict_results["image_id"]=dict_value["image id"]
            coco_dict_results["category_id"]=pre_instances.pred_classes.cpu().numpy()[i]+1
            coco_dict_results["bbox"]=pre_instances.pred_boxes.tensor.cpu().numpy()[i]#pre_output['instances'].to("cpu")
            coco_dict_results["score"]=pre_instances.scores.cpu().numpy()[i]
            coco_list_results.append(coco_dict_results)
            if my_visual:
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
                if i==len(pre_instances.scores)-1:
                    cv2.putText(img, r"len{} cid:{}".format(len(pre_instances.scores),list(set(cate))[:]), (15,40), cv2.FONT_HERSHEY_COMPLEX, 1.5, (170,64,112), 4)#
                    cv2.putText(img, r"c1:{} c2:{} c3:{} c4:{}".format(id_1,id_2,id_3,id_4), (15,80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (170,64,112), 4)
                    os.makedirs(os.path.join(cfg.OUTPUT_DIR,"my_pre_split_visual"),exist_ok=True)
                    cv2.imwrite(os.path.join(cfg.OUTPUT_DIR,"my_pre_split_visual","vis2_{}".format(file_name)),img)
        if d2_visual: 
            v = Visualizer(img[:, :, ::-1],metadata=train_dicts_metadata, scale=1,instance_mode=ColorMode.IMAGE)# ColorMode.SEGMENTATION or ColorMode.IMAGE_BW) 
            v = v.draw_instance_predictions(pre_output["instances"].to("cpu"))
            os.makedirs(os.path.join(cfg.OUTPUT_DIR,"d2_predict_split_visual"),exist_ok=True)
            cv2.imwrite(os.path.join(cfg.OUTPUT_DIR,"d2_predict_split_visual","vis2_{}".format(file_name)),v.get_image()[:, :, ::-1])
    if save_json:
        f1=open(os.path.join(cfg.OUTPUT_DIR, "my_predict","pre_result_test.json"),'w') 
        f1.write(json.dumps(coco_list_results,cls=MyEncoder))
    print("predict----------------end")
    if megrge_result:
        print("--------->>>>>>>>>merge-------------start")
        merge =ResultMerge.DetResMerge(resfile=os.path.join(cfg.OUTPUT_DIR, "my_predict","pre_result.json"), 
                                splitannofile=test_json, 
                                srcannofile="/root/data/gvision/dataset/predict/s0.5_t0.8_141517/image_annos/person_bbox_test_141517.json",
                                outpath=cfg.OUTPUT_DIR,
                                outfile="my_predict/pre_merge_result.json")
        merge.mergeResults(is_nms=True)
        print("merge-------------end")
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
    test_json="/root/data/gvision/dataset/predict/14_01/image_annos/s0.5_14_01_split_test.json"
    test_image_path="/root/data/gvision/dataset/predict/14_01/image_test"
    dataset_test_dicts = json.load(open(test_json,"r"))
    MetadataCatalog.get("pv_train").set(thing_classes=['visible body', 'full body', 'head', 'vehicle'], # 可以选择开启，但是不能显示中文，所以本人关闭
                                        thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3})
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
        if visual and up=="2048" and image_id=="01":
            print("visual-------------------------",file_name)
            cv2.putText(img, f"len:{num_instance} c1:{cid[0]} c2:{cid[1]} c3:{cid[2]} c4:{cid[3]}", (15,80), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (138,0,255), 4)
            v = Visualizer(img[:, :, ::-1],metadata=train_dicts_metadata, scale=1,instance_mode=ColorMode.IMAGE)# ColorMode.SEGMENTATION or ColorMode.IMAGE_BW) 
            v = v.draw_instance_predictions(pre_instances.to("cpu"))#draw xyxy
            os.makedirs(os.path.join(cfg.OUTPUT_DIR,"d2_predict_split_visual_14_01"),exist_ok=True)
            cv2.imwrite(os.path.join(cfg.OUTPUT_DIR,"d2_predict_split_visual_14_01","visual{}_{}".format(model_weight[6:-4],file_name)),v.get_image()[:, :, ::-1])
    print(os.path.join(cfg.OUTPUT_DIR, "my_predict",f"{model_weight[6:-4]}_predict_14_01.json"))
    f1=open(os.path.join(cfg.OUTPUT_DIR, "my_predict",f"{model_weight[6:-4]}_predict_14_01.json"),'w') 
    f1.write(json.dumps(coco_list_results,cls=MyEncoder))
    print("predict----------------end")
def single_predict(filter_cate=None):
    cfg.TEST.DETECTIONS_PER_IMAGE=200
    cfg.MODEL.WEIGHTS=os.path.join(cfg.OUTPUT_DIR,"model_final.pth")
    img=cv2.imread("/root/data/gvision/dataset/predict/1.jpg")
    predictor = DefaultPredictor(cfg)
    outputs = predictor(img)
    pre_instances=outputs['instances']
    # # We can use `Visualizer` to draw the predictions on the image.
    MetadataCatalog.get("pv_train").set(thing_classes=['visible body', 'full body', 'head', 'vehicle'], # 可以选择开启，但是不能显示中文，所以本人关闭
                                        thing_dataset_id_to_contiguous_id={1: 0, 2: 1, 3: 2, 4: 3})
    train_dicts_metadata = MetadataCatalog.get("pv_train")
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
    # mmcv.imshow_bboxes(img,bboxes,top_k=10,show=False,out_file="/root/data/gvision/detectron2-master/demo/ouputmmcv_10_1.jpg")
    # mmcv.imshow_bboxes(img,bboxes,show=False,out_file="/root/data/gvision/1detectron2-master/demo/ouputmmcv2_all_1.jpg")
    # mmcv.imshow_bboxes(img,bboxes,top_k=500,show=False,out_file="/root/data/gvision/detectron2-master/demo/ouputmmcv_500_1.jpg")
    out_file=f"/root/data/gvision/detectron2-master/workdir/output/test/ouput_san_d200_c{filter_cate}.jpg"
    mmcv.imshow_det_bboxes(
        img,
        np.array(bboxes),
        np.array(category_ids),
        class_names=['visible body', 'full body', 'head', 'vehicle'],
        show=False,
        out_file=out_file)

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
    trainer=train(train_flag=False,resume_load=False)
    # for i in[1,2,3]:
    #     single_predict(i)
    "check pre annotation"
    # checkout_pre_annotation(visual_num=10)
    "predict and merge_nms" 
    # single_predict()
    for i in ["model_0018499.pth"]:
        predict_my(visual=True)
    # "merge"
    # merge()
    "-------------"
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




