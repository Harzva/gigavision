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
try:
    import xml.etree.cElementTree as ET  #解析xml的c语言版的模块
except ImportError:
    import xml.etree.ElementTree as ET
def sceneid2list(selected):
    selected_dict={"14":list(range(391,421)),"15":list(range(421,451)),"16":list(range(451,481)),"17":list(range(481,511)),"18":list(range(511,556)),"no":list(range(1000,1200)),}
    temp_list=[]
    for i in selected:
        temp_list=temp_list+selected_dict[i]
    return temp_list
def results_resolve(model_path,weight,selected1,selected2,selected3,selected4,model_id):
    selected4=sceneid2list(selected4)
    selected3=sceneid2list(selected3)
    selected2=sceneid2list(selected2)
    selected1=sceneid2list(selected1)
    with open(model_path, 'r') as load_f:
        model_results= json.load(load_f)
    results1,results2,results3,results4=[],[],[],[]
    for i in model_results:
        if i['category_id']==1 and i['image_id'] in selected1:
            new_score=i['score']*weight[0]
            if new_score>1:
                new_score=float(1)
            i.update(category_id=1,score=new_score,number=i['image_id']+model_id)
            assert isinstance(i['category_id'],int),f"the  results must is 1" 
            results1.append(i)
        elif i['category_id']==2 and i['image_id'] in selected2:
            new_score=i['score']*weight[0]
            if new_score>1:
                new_score=float(1)
            i.update(category_id=2,score=new_score,number=i['image_id']+model_id)
            assert isinstance(i['category_id'],int),f"the  results must is 2" 
            results2.append(i)
        elif i['category_id']==3 and i['image_id'] in selected3:
            new_score=i['score']*weight[0]
            if new_score>1:
                new_score=float(1)
            i.update(category_id=3,score=new_score,number=i['image_id']+model_id)
            assert isinstance(i['category_id'],int),f"the  results must is 3" 
            results3.append(i)
        elif i['category_id']==4 and i['image_id'] in selected4:
            new_score=i['score']*weight[0]
            if new_score>1:
                new_score=float(1)
            i.update(category_id=4,score=new_score,number=i['image_id']+model_id)
            assert isinstance(i['category_id'],int),f"the  results must is 4" 
            results4.append(i)
    print("cid1",len(results1),"cid2",len(results2),"cid3",len(results3),"cid4",len(results4))
    return results1,results2,results3,results4

def indexResults(reslist,annopath=""):
    annopath="/root/data/gvision/dataset/raw_data/image_annos/person_bbox_test.json"
    # print('Loading test annotation json file: {}'.format(annopath))
    with open(annopath, 'r') as load_f:
        anno= json.load(load_f)
    # print("bboxex_num",len(reslist))#498
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
        for resdict in reslist:
            resimageid = resdict['image_id']
            if resimageid == imageid:
                indexedresults[imageid ].append(resdict)
        return indexedresults
    executor = ThreadPoolExecutor(max_workers=10)
    func_var = [[file_name,dict_value] for file_name,dict_value in anno.items()]
    pbar = tqdm(total=len(anno), ncols=50)
    for temp in executor.map(say,func_var):
        indexedresults.update(temp)
        pbar.update(1)
    pbar.close()
    results = indexedresults
    print("index bbox to self image")
    return results 
def GetAnnotBoxLoc(AnotPath,rectange_list):#AnotPath VOC标注文件路径
    tree = ET.ElementTree(file=AnotPath)  #打开文件，解析成一棵树型结构
    root = tree.getroot()#获取树型结构的根
    ObjectSet=root.findall('object')#找到文件中所有含有object关键字的地方，这些地方含有标注目标
    backlist=[]
    # print(f"forbid zone before {len(rectange_list)}")
    
    for a in rectange_list:
        i=a["bbox"]
        imageid=a["image_id"]
        # print(imageid)
        left,up,right,down=i[0],i[1],i[0]+i[2],i[3]+i[1]
        templist=[]
        inter_xml=np.zeros(len(ObjectSet),dtype=float)
        for k,Object in enumerate(ObjectSet):
            BndBox=Object.find('bndbox')
            xmin= int(BndBox.find('xmin').text)#-1 #-1是因为程序是按0作为起始位置的
            ymin= int(BndBox.find('ymin').text)#-1
            xmax= int(BndBox.find('xmax').text)#-1
            ymax= int(BndBox.find('ymax').text)#-1
            templist.append({
                    "image_id":imageid,
                    "category_id":5,
                    "bbox": [xmin,ymin,xmax-xmin,ymax-ymin],
                    "score":0
                })

            if xmax <= left or right <= xmin or ymax <= up or down <= ymin:
                intersection = 0
            else:
                lens = min(xmax, right) - max(xmin, left)
                wide = min(ymax, down) - max(ymin, up)
                intersection = lens * wide
                # print("*"*60,intersection)
                # print(i[2]*i[3])
            inter_xml[k]=intersection/(i[2]*i[3]+0.00001)
        if np.where(inter_xml<0.05)[0].shape[0]==len(ObjectSet):#则没有与bbox相交的xmlforbidzone < param or ==0
            backlist.append(a)
        # else:
        #     print(np.where(inter_xml==0)[0].shape[0]==len(ObjectSet))
        #     print("del")
    # print(f"forbid zone after {len(backlist)}")
    return backlist#+templist#17_newzhongguan
    # return backlist,templist
#
def wnms(results,outpath,outfile,iouthresh,savejson=1,nmsname="nms"):
    indexedresults=indexResults(results)
    mergedresults = defaultdict(list)
    for (imageid, objlist) in indexedresults.items():
        for objdict in objlist:
            mergedresults[imageid].append([objdict['bbox'][0],objdict['bbox'][1],objdict['bbox'][2],objdict['bbox'][3],objdict['score'], objdict['category_id'],objdict["number"]])
        objlist=mergedresults[imageid]
        # masxlist=[i[2]*i[3] for i in objlist]
        # max_wh=np.max(masxlist)
        # objlist=[[i[0],i[1],i[2],i[3],i[4]*0.05+i[3]*i[2]*0.95/max_wh,i[5],i[6]] for i in objlist ]
        if nmsname=="softnms":
            newdets,keep=soft_nms(np.array(objlist),iou_thr=iouthresh, method='linear',sigma=0.5, min_score=1e-3)#'gaussian''linear',
            # keep =py_cpu_softnms(np.array(objlist),thresh=nms_thresh, Nt=0.02, sigma=0.5, method=1)
            outdets = []
            for index in keep:
                outdets.append(objlist[index])
            mergedresults[imageid] = outdets
        elif nmsname=="setnms":
            print(objlist[0])
            print(len(objlist[0]))
            keep=np.array(objlist)[set_cpu_nms(np.array(objlist),iouthresh)].tolist()
            mergedresults[imageid] = keep
        elif nmsname==False:
            print("no nms")
        else:
            raise ValueError('nmsname must is softnms or nms')
    savelist = []
    def say2(iss):
        imageid, objlist=iss[0],iss[1]
        templist=[]

        for obj in objlist:#obj [22528, 1270, 24576, 1, 1.0, 4]
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
        # print(temp)
        savelist+=temp
        pbar2.update(1)
    pbar2.close()
    # assert len(savelist)==0,f"error{savelist} error"
    if savejson:
        assert isinstance(savelist[0], dict),f"the  results must is not {savelist[0]}" 
        # if  not isinstance(savelist[0], dict):
        #     raise f"the  results must is not {savelist[0]}" 
        # print(savelist[0]['category_id'])
        outfile=outfile[:-5].replace("all",f"{savelist[1]['category_id']}")+".json"
        with open(os.path.join(outpath, outfile), 'w') as f:
            dict_str = json.dumps(savelist, indent=2)
            f.write(dict_str)
            print(f"save ***{len(savelist)} results*** json :{os.path.join(outpath, outfile)}")
    return savelist
    
def model_fusion( outpath="/root/data/gvision/my_merge/fusion_results",
    outfile="fafaxue_final_wnms_all.json",model_num=2):
    results_all=[[],[],[],[]]
    a=["14","15","16","17","18"]
    b=["15","16","17"]
    no=["no"]
    "selected1,selected2,selected3,selected4 不同类别选择不同场景"
    
    json_root="/root/data/gvision/final_merge/"
    for idj,i in enumerate(zip(
                list(results_resolve(model_path="/root/data/gvision/my_merge/finalsubmission/fafafinal/det_results.json",weight=[1,1,1,1],
                selected1=a,selected2=a,selected3=a,selected4=a,model_id=0)),

                list(results_resolve(model_path="/root/data/gvision/my_merge/finalsubmission/fafafinal/det_results.json",weight=[1,1,1,1],
                selected1=a,selected2=a,selected3=a,selected4=a,model_id=1))
                )):
        a=[]
        for j in range(model_num):         
            a+=i[j]
        results_all[idj]=a

    results1,results2,results3,results4=results_all[0],results_all[1],results_all[2],results_all[3]
    # print(results_all)
    assert results1[0]["category_id"]==1,f'4 error {results1[0]["category_id"]}'
    assert results2[0]["category_id"]==2,f'4 error {results2[0]["category_id"]}'
    assert results3[0]["category_id"]==3,f'4 error {results3[0]["category_id"]}'
    assert results4[0]["category_id"]==4,f'4 error {results4[0]["category_id"]}'
    assert type(results1[0]["category_id"])==int,f'4 error '
    assert type(results2[0]["category_id"])==int,f'4 error '
    assert type(results3[0]["category_id"])==int,f'4 error '
    assert type(results4[0]["category_id"])==int,f'4 error '


    afternms_results1=wnms(results1,outpath,outfile=outfile,iouthresh=0.1,savejson=0,nmsname="setnms")#"softnms")#

    afternms_results2=wnms(results2,outpath,outfile=outfile,iouthresh=0.1,savejson=0,nmsname="setnms")#"softnms")#

    afternms_results3=wnms(results3,outpath,outfile=outfile,iouthresh=0.1,savejson=0,nmsname="setnms")#"nms")#

    afternms_results4=wnms(results4,outpath,outfile=outfile,iouthresh=0.1,savejson=0,nmsname="setnms")#"nms")#

    afternms_results_all=afternms_results1+afternms_results2+afternms_results3+afternms_results4
    with open(os.path.join(outpath, outfile), 'w') as f:
        dict_str = json.dumps(afternms_results_all, indent=2)
        f.write(dict_str)
        print(f"save ***{len(afternms_results_all)} results*** json :{os.path.join(outpath, outfile)}")




def main():
    global test,isdel_inter,isfliter,fliterscore
    fliterscore={"14_OCT":0,"15_nanshan":0,"1601_shool":0,"1602_shool":0,"17_newzhongguan":0,"1801_xilin":0,"1802_xilin":0}
    isfliter=0#fliter score xml
    isdel_inter=0
    test=0

    if test:
        outpath="/root/data/gvision/my_merge/fusion_results"
    else:
        outpath="/root/data/gvision/final_merge/fusion_results"
    model_fusion(outpath=outpath,
            outfile="final_det_det7755_all.json")#must all

if __name__ == "__main__":
    main()