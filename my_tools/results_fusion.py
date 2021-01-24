from nms import py_cpu_nms,py_cpu_softnms,set_cpu_nms
from d2det.ops.nms.nms_wrapper import soft_nms
import numpy as np
from tqdm import tqdm 
from collections import defaultdict
import random
from concurrent.futures.thread import ThreadPoolExecutor
from ensemble_boxes import nms
import json
import os
def scene_fusion():
    """scene fusion"""
    path_result14_15_16="/root/data/gvision/CrowdDet-master/model/rcnn_emd_refinet/outputs/coco_results/full_dump-4emnms_crowdet.json"#y14 15 
    path_result14_15_16="/root/data/gvision/CrowdDet-master/model/rcnn_emd_refinet/outputs/coco_results/full_enms_flitertestalldump-4nms0.35prethre0.4.json"
    path_result17_18="/root/data/gvision/CrowdDet-master/model/rcnn_emd_refinet/outputs/coco_results/full_emnms_crowdet.json"#17 18
    path_result17_18="/root/data/gvision/CrowdDet-master/model/rcnn_emd_refinet/outputs/coco_results/full_enms_flitertestallfnms0.35prethre0.4.json"
    with open(path_result14_15_16, 'r') as load_f:
        result14_15_16= json.load(load_f)
    with open(path_result17_18, 'r') as load_f:
        result17_18= json.load(load_f)
    result14_15_16_dict=[temp for temp in result14_15_16 if 391<=temp["image_id"]<=480]
    result17_18_dict=[temp for temp in result14_15_16 if 480<=temp["image_id"]<=555]
    result=result17_18_dict+result14_15_16_dict
    f=open("/root/data/gvision/CrowdDet-master/submission/filter_dump4_14_15_16_resume_1718.json", 'w')
    dict_str = json.dumps(result, indent=2)
    f.write(dict_str)
def classes_fusion():
    """classes fusion"""
    # path_result1="/root/data/gvision/my_merge/finalsubmission/final1/testcrowdet_visible.json"
    # path_result2="/root/data/gvision/my_merge/finalsubmission/final2/2.json"
    # path_result3="/root/data/gvision/my_merge/finalsubmission/final2/m2retinaface_head.json"#17 18
    # path_result123="/root/data/gvision/final_merge/fusion_results/final_123_all.json"
    # path_result4="/root/data/gvision/my_merge/finalsubmission/fafafinal/det_results_4.json"
    # path_result4="/root/data/gvision/final_merge/fusion_results/final_4.json"
    # path_result4_14="/root/data/rubzz/fengwei/picture_proecessing/2day/submitnms.json"
    # path_result4_15="/root/data/rubzz/fengwei/picture_proecessing/2day/15submitnms0.1+0.18.json"
    path_result124="/root/data/gvision/my_merge/finalsubmission/fafafinal/det_results_4.json"
    path_result3="/root/data/gvision/final_merge/fusion_results/final_4.json"
    # path_result4="/root/data/gvision/my_merge/finalsubmission/final2/vehicle.json"
    # [
    #   {
    #     "image_id": 391,
    #     "category_id": 1,
    #     "bbox": [
    #       9028,
    #       6249,
    #       109,
    #       124
    #     ],
    #jiancha
    with open(path_result1, 'r') as load_f:
        result1= json.load(load_f)
    assert result1[0]["category_id"]==1,"1 error"
    print(type(result1[0]["category_id"]),result1[0]["category_id"])
    assert type(result1[0]["category_id"])==int,f"4 error {cls}"
    with open(path_result2, 'r') as load_f:
        result2= json.load(load_f)
    assert result2[0]["category_id"]==2,"2 error"
    assert type(result2[0]["category_id"])==int,f"4 error {cls}"
    with open(path_result3, 'r') as load_f:
        result3= json.load(load_f)
        cls=result3[0]["category_id"]
        result3=[{"image_id": i["image_id"],"category_id":3,"bbox":i["bbox"],"score":i["score"]} for i in result3]
    assert result3[0]["category_id"]==3,cls
    assert type(result3[0]["category_id"])==int,f"4 error {cls}"
    with open(path_result4, 'r') as load_f:
        result4= json.load(load_f)
        cls=result4[0]["category_id"]
    assert result4[0]["category_id"]==4,f"4 error {cls}"
    assert type(result4[1]["category_id"])==int,f"4 error {cls}"
    with open(path_result123, 'r') as load_f:
        result123= json.load(load_f)
        result123=[temp for temp in result123 if temp["category_id"]==1 or temp["category_id"]==2 or temp["category_id"]==3]
        # result123=[{"image_id": i["image_id"],"category_id":int(i["category_id"]),"bbox":i["bbox"],"score":i["score"]} for/ i in result123]
    print(type(result123))
    result123_4=result123+result4
    resultall=result1+result2+result3+result4
    with open("/root/data/gvision/my_merge/finalsubmission/final2/0.57_123_my_4.json", 'w') as f:
        dict_str = json.dumps(result123_4, indent=2)
        f.write(dict_str)
# classes_fusion()
def vehicle_fusion():
    # path_resultcar_1="/root/data/gvision/my_merge/vehicle/coco_results/detectors/car_17.json"#17 18
    # path_resultcar_2="/root/data/gvision/my_merge/vehicle/coco_results/detectors/car_without_17.json"#17 18
    # path_resulelse="/root/data/gvision/my_merge/vehicle/coco_results/detectors/else.json"
    path_result4_14="/root/data/rubzz/fengwei/picture_proecessing/2day/submitnms.json"
    path_result4_15="/root/data/rubzz/fengwei/picture_proecessing/2day/15submitnms0.1+0.18.json"
    path_result4_16="/root/data/gvision/final_merge/final2/det_results.json"
    path_result4_1718="/root/data/gvision/my_merge/finalsubmission/final2/vehicle.json"
    path_result123="/root/data/gvision/final_merge/final2/det_results.json"

    with open(path_result123, 'r') as load_f:
        result123= json.load(load_f)
        result123==[x for x in result123 if x["category_id"]==1 or x["category_id"]==2 or x["category_id"]==3]
        print(result123[0]["category_id"])
    with open(path_result4_14, 'r') as load_f:
        result4_14= json.load(load_f)

    with open(path_result4_15, 'r') as load_f:
        result4_15= json.load(load_f)

    with open(path_result4_16, 'r') as load_f:
        result4_16= json.load(load_f)
        result4_16=[x for x in result4_16 if 451<=x["image_id"]<=480]


    with open(path_result4_1718, 'r') as load_f:
        result4_1718= json.load(load_f)
        result4_1718=[x for x in result4_1718 if 481<=x["image_id"]<=555 ]
    result=result4_16+result4_1718+result4_15+result4_14+result123

    # resultall=resultcar_1+resultcar_2+resulelse
    with open("/root/data/gvision/final_merge/final4/submit_123_4.json", 'w') as f:
        dict_str = json.dumps(result, indent=2)
        f.write(dict_str)
def results_resolve(model_path,weight):
    with open(model_path, 'r') as load_f:
        model_results= json.load(load_f)
    # results1,results2,results3,results4=[],[],[],[]
    # for i in model_results:
    #     if i['category_id']==1:
    #         print(i['score'])
    #         i.update(score=i['score']*weight)
    #         results1.append(i)
    #     elif i['category_id']==2:
    #         print(i['score'])
    #         i.update(score=i['score']*weight)
    #         results2.append(i)
    #     elif i['category_id']==3:
    #         i.update(score=i['score']*weight)
    #         results3.append(i)
    #     elif i['category_id']==4:
    #         i.update(score=i['score']*weight)
    #         results4.append(i)
    # [i.update(score=i['score']*0.8) for i in a if i['category_id']==3]
    results1=[i for i in model_results if i['category_id']==1]
    results2=[i for i in model_results if i['category_id']==2]
    results3=[i for i in model_results if i['category_id']==3]
    results4=[i for i in model_results if i['category_id']==4]
    return results1,results2,results3,results4
# results1,results2,results3,results4=results_resolve("/root/data/gvision/final_merge/final6/det_results.json",weight=[1.0,1.0,1.0,1.0])
# outpath="/root/data/gvision/final_merge/final6"
# with open(os.path.join(outpath, "det_results_1.json"), 'w') as f:
#     dict_str = json.dumps(results1, indent=2)
#     f.write(dict_str)
# with open(os.path.join(outpath, "det_results_2.json"), 'w') as f:
#     dict_str = json.dumps(results2, indent=2)
#     f.write(dict_str)

# with open(os.path.join(outpath, "det_results_3.json"), 'w') as f:
#     dict_str = json.dumps(results3, indent=2)
#     f.write(dict_str)

# with open(os.path.join(outpath,"det_results_4.json"), 'w') as f:
#     dict_str = json.dumps(results4, indent=2)
#     f.write(dict_str)


def indexResults(reslist,anno):
    print("bboxex_num",len(reslist))#498
    indexedresults = defaultdict(list)
    # if test:
    #     tempannos={}
    #     imgfilters=imgfilters
    #     if imgfilters:
    #     # imgfilters=["15_24"]
    #         for imgfilter in imgfilters:
    #             tempannos.update({i:j for i,j in anno.items() if imgfilter in i })
    #         anno=tempannos
    def say(iss):
        filename, annodict=iss[0],iss[1]
        imageid = annodict['image id']
        # print("imageid",imageid)
        for resdict in reslist:
            resimageid = resdict['image_id']
            if resimageid == imageid:
                # print("1111",resdict) {'image_id': 253, 'category_id': 1, 'bbox': [981.3349609375, 322.8221435546875, 22.030517578125, 32.01666259765625], 'score': 0.16039377450942993}
                # print("2222",resimageid)
                # print("1111",type(resdict))
                # print("2222",type(resimageid))
                indexedresults[filename].append(resdict)

        return indexedresults
    # print("anno",anno)
    executor = ThreadPoolExecutor(max_workers=1)
    func_var = [[file_name,dict_value] for file_name,dict_value in anno.items()]
    pbar = tqdm(total=len(anno), ncols=50)
    for temp in executor.map(say,func_var):
        # print(temp)
        indexedresults.update(temp)
        pbar.update(1)
    pbar.close()
    results = indexedresults
    return results 
def model_fusion(outpath,outfile):
    resultsa1,resultsa2,resultsa3,resultsa4=results_resolve(model_path="/root/data/gvision/my_merge/finalsubmission/fafafinal/det_results.json",weight=1)
    resultsb1,resultsb2,resultsb3,resultsb4=results_resolve(model_path="/root/data/gvision/my_merge/finalsubmission/final2/all.json",weight=0.6)
    annopath="/root/data/gvision/dataset/raw_data/image_annos/person_bbox_test.json"
    # for i in zip(list(resulta1,resulta2,resulta3,resulta4),list(resultb1,resultb2,resultb3,resultb4)):
    results1=resultsa1+resultsb1
    results2=resultsa2+resultsb2
    results3=resultsa3+resultsb3
    results4=resultsa4+resultsb4
    print('Loading split annotation json file: {}'.format(annopath))
    with open(annopath, 'r') as load_f:
        srcanno = json.load(load_f)
    indexedresults=indexResults(results1,anno=srcanno)
    mergedresults = defaultdict(list)
    for (filename, objlist) in indexedresults.items():
        # print("filename",filename)
        # print("srcfile, paras",srcfile, paras )
        srcfile = filename.replace('_IMG', '/IMG')#02_Xili_Crossroad_IMG_02_01___0.5__0__0.jpg
        srcimageid = srcanno[srcfile]['image id']
        for objdict in objlist:
            mergedresults[srcimageid].append([objdict['bbox'][0],objdict['bbox'][1],objdict['bbox'][2],objdict['bbox'][3],objdict['score'], objdict['category_id']])
    for (imageid, objlist) in mergedresults.items():
        print(imageid,objlist)
        # masxlist=[i[2]*i[3] for i in objlist]
        # max_wh=np.max(masxlist)
        # objlist=[[i[0],i[1],i[2],i[3],i[4]*0.05+i[3]*i[2]*0.95/max_wh,i[5],i[6]] for i in objlist ]
        keep = py_cpu_nms(np.array(objlist),0.5)
        outdets = []
        for index in keep:
            outdets.append(objlist[index])
        mergedresults[imageid] = outdets
    savelist = []
    def say2(iss):
        imageid, objlist=iss[0],iss[1]
        # print(imageid, objlist)
        templist=[]
        for obj in objlist:#obj [22528, 1270, 24576, 1, 1.0, 4]
            # print(obj)
            templist.append({
                "image_id": imageid,
                "category_id": obj[5],
                "bbox": obj[:4],
                # "bbox": tlbr2tlwh(obj[:4]),
                "score": obj[4]
            })
        return templist
    executor = ThreadPoolExecutor(max_workers=80)
    func_var = [[file_name,dict_value] for file_name,dict_value in mergedresults.items()]

    print("fusion bbox into self'image start ")
    pbar2= tqdm(total=len(mergedresults), ncols=50)
    for temp in executor.map(say2,func_var):
        savelist+=temp
        pbar2.update(1)
    pbar2.close()
    with open(os.path.join(outpath, outfile), 'w') as f:
        dict_str = json.dumps(savelist, indent=2)
        f.write(dict_str)
        print(f"save ***results*** json :{os.path.join(outpath, outfile)}")
# model_fusion(outpath="/root/data/gvision/my_merge/fusionresults", outfile="fafaxue_final2.json")
# vehicle_fusion()
#
# classes_fusion()
# scene_fusion()
# vehicle_fusion()
